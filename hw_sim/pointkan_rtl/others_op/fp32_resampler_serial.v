

module fp32_lerp_pipe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] y0,
    input  wire [31:0] y1,
    input  wire [31:0] t,
    output wire [31:0] result,
    output wire        valid_out
);

    wire [31:0] neg_y0 = {~y0[31], y0[30:0]};

    wire [31:0] diff;
    wire        diff_vld;

    fp32_add u_add_sub (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in),
        .a        (y1),
        .b        (neg_y0),
        .result   (diff),
        .valid_out(diff_vld)
    );

    reg [31:0] y0_d1, t_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y0_d1 <= 32'b0;
            t_d1  <= 32'b0;
        end else begin
            y0_d1 <= y0;
            t_d1  <= t;
        end
    end

    wire [31:0] prod;
    wire        prod_vld;

    fp32_mul u_mul (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (diff_vld),
        .a        (diff),
        .b        (t_d1),
        .result   (prod),
        .valid_out(prod_vld)
    );

    reg [31:0] y0_d2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            y0_d2 <= 32'b0;
        else
            y0_d2 <= y0_d1;
    end

    fp32_add u_add_final (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (prod_vld),
        .a        (y0_d2),
        .b        (prod),
        .result   (result),
        .valid_out(valid_out)
    );

endmodule

module fp32_resampler_down_sample #(
    parameter N_IN_MAX  = 51,
    parameter N_OUT_MAX = 32,
    parameter ADDR_W    = 10,
    parameter FRAC_W    = 16
)(
    input  wire                        clk,
    input  wire                        rstn,
    input  wire [ADDR_W-1:0]           cfg_n_in,
    input  wire [ADDR_W-1:0]           cfg_n_out,
    input  wire [ADDR_W+FRAC_W-1:0]   cfg_step,
    input  wire                        wr_en,
    input  wire [ADDR_W-1:0]           wr_addr,
    input  wire [31:0]                 wr_data,
    input  wire                        wr_t_en,
    input  wire [ADDR_W-1:0]           wr_t_addr,
    input  wire [31:0]                 wr_t_data,
    input  wire                        start,
    output wire [31:0]                 out_data,
    output wire                        out_valid,
    output reg                         busy,
    output reg                         done
);

    reg [31:0] mem [0:N_IN_MAX-1];
    always @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    reg [31:0] t_lut [0:N_OUT_MAX-1];
    always @(posedge clk)
        if (wr_t_en) t_lut[wr_t_addr] <= wr_t_data;

    reg [ADDR_W-1:0]        n_in_r, n_out_r;
    reg [ADDR_W+FRAC_W-1:0] step_r;

    reg  [ADDR_W+FRAC_W-1:0] pos;
    reg  [ADDR_W-1:0]        feed_cnt;
    wire [ADDR_W-1:0]        idx     = pos[ADDR_W+FRAC_W-1 : FRAC_W];
    wire [ADDR_W-1:0]        n_in_m1 = n_in_r - 1'b1;

    wire is_last_point = (feed_cnt == n_out_r - 1'b1);
    wire [ADDR_W-1:0] idx_eff  = is_last_point ? n_in_m1 : idx;
    wire [ADDR_W-1:0] idx1_eff = (idx_eff >= n_in_m1) ? n_in_m1 : (idx_eff + 1'b1);

    wire [31:0] y0_w   = mem[idx_eff];
    wire [31:0] y1_w   = mem[idx1_eff];
    wire [31:0] t_fp32 = t_lut[feed_cnt];

    reg         lerp_valid_in;
    wire [31:0] lerp_result;
    wire        lerp_valid_out;

    fp32_lerp_pipe u_lerp (
        .clk      (clk),
        .rst_n    (rstn),
        .valid_in (lerp_valid_in),
        .y0       (y0_w),
        .y1       (y1_w),
        .t        (t_fp32),
        .result   (lerp_result),
        .valid_out(lerp_valid_out)
    );

    assign out_data  = lerp_result;
    assign out_valid = lerp_valid_out;

    reg [ADDR_W-1:0] out_cnt;

    localparam [1:0] F_IDLE  = 2'd0,
                     F_RUN   = 2'd1,
                     F_DRAIN = 2'd2;

    reg [1:0] fsm;
    reg [1:0] drain_cnt;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            fsm           <= F_IDLE;
            pos           <= {(ADDR_W+FRAC_W){1'b0}};
            feed_cnt      <= {ADDR_W{1'b0}};
            out_cnt       <= {ADDR_W{1'b0}};
            drain_cnt     <= 2'b0;
            lerp_valid_in <= 1'b0;
            busy          <= 1'b0;
            done          <= 1'b0;
            n_in_r        <= {ADDR_W{1'b0}};
            n_out_r       <= {ADDR_W{1'b0}};
            step_r        <= {(ADDR_W+FRAC_W){1'b0}};
        end else begin
            done          <= 1'b0;
            lerp_valid_in <= 1'b0;

            if (lerp_valid_out && busy)
                out_cnt <= out_cnt + 1'b1;

            case (fsm)
                F_IDLE: begin
                    if (start) begin
                        n_in_r    <= cfg_n_in;
                        n_out_r   <= cfg_n_out;
                        step_r    <= cfg_step;
                        pos       <= {(ADDR_W+FRAC_W){1'b0}};
                        feed_cnt  <= {ADDR_W{1'b0}};
                        out_cnt   <= {ADDR_W{1'b0}};
                        busy      <= 1'b1;
                        fsm       <= F_RUN;
                    end
                end

                F_RUN: begin

                    lerp_valid_in <= 1'b1;

                    if (feed_cnt >= n_out_r - 1'b1) begin

                        drain_cnt <= 2'd0;
                        fsm       <= F_DRAIN;
                    end else begin
                        pos      <= pos + step_r;
                        feed_cnt <= feed_cnt + 1'b1;
                    end
                end

                F_DRAIN: begin

                    if (drain_cnt >= 2'd2) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        fsm  <= F_IDLE;
                    end else begin
                        drain_cnt <= drain_cnt + 1'b1;
                    end
                end

                default: fsm <= F_IDLE;
            endcase
        end
    end

endmodule