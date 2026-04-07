module pe #(
    parameter DATA_WIDTH = 32,
    parameter ACC_WIDTH  = 80
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   q_valid_in,
    input  wire [DATA_WIDTH-1:0]  q_coord_in,
    output reg                    q_valid_out,
    output reg  [DATA_WIDTH-1:0]  q_coord_out,
    input  wire [DATA_WIDTH-1:0]  d_coord,
    output reg  [ACC_WIDTH-1:0]   dist_sq_out,
    output reg                    result_valid
);

    reg [1:0] dim_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dim_cnt <= 2'd0;
        else if (q_valid_in) begin
            if (dim_cnt == 2'd2)
                dim_cnt <= 2'd0;
            else
                dim_cnt <= dim_cnt + 2'd1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_valid_out <= 1'b0;
            q_coord_out <= {DATA_WIDTH{1'b0}};
        end else begin
            q_valid_out <= q_valid_in;
            q_coord_out <= q_coord_in;
        end
    end

    wire signed [DATA_WIDTH-1:0]   diff;
    wire signed [2*DATA_WIDTH-1:0] diff_sq;

    assign diff    = $signed(q_coord_in) - $signed(d_coord);
    assign diff_sq = $signed(diff) * $signed(diff);

    reg [ACC_WIDTH-1:0] acc_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg      <= {ACC_WIDTH{1'b0}};
            dist_sq_out  <= {ACC_WIDTH{1'b0}};
            result_valid <= 1'b0;
        end else begin
            result_valid <= 1'b0;
            if (q_valid_in) begin
                case (dim_cnt)
                    2'd0: acc_reg <= {{(ACC_WIDTH-2*DATA_WIDTH){1'b0}}, diff_sq};
                    2'd1: acc_reg <= acc_reg + {{(ACC_WIDTH-2*DATA_WIDTH){1'b0}}, diff_sq};
                    2'd2: begin
                        dist_sq_out  <= acc_reg + {{(ACC_WIDTH-2*DATA_WIDTH){1'b0}}, diff_sq};
                        result_valid <= 1'b1;
                    end
                    default: acc_reg <= {ACC_WIDTH{1'b0}};
                endcase
            end
        end
    end

endmodule


module distance_engine #(
    parameter PE_NUM     = 64,
    parameter DATA_WIDTH = 32,
    parameter ACC_WIDTH  = 80
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             start,
    input  wire [DATA_WIDTH-1:0]            q_x,
    input  wire [DATA_WIDTH-1:0]            q_y,
    input  wire [DATA_WIDTH-1:0]            q_z,
    input  wire [PE_NUM*DATA_WIDTH-1:0]     db_x_packed,
    input  wire [PE_NUM*DATA_WIDTH-1:0]     db_y_packed,
    input  wire [PE_NUM*DATA_WIDTH-1:0]     db_z_packed,
    output wire [ACC_WIDTH-1:0]             dist_sq_out,
    output wire                             result_valid,
    output wire [$clog2(PE_NUM)-1:0]        result_idx,
    output wire                             all_done
);

    wire                   q_valid_chain [0:PE_NUM];
    wire [DATA_WIDTH-1:0]  q_coord_chain [0:PE_NUM];
    wire [ACC_WIDTH-1:0]   pe_dist_sq    [0:PE_NUM-1];
    wire                   pe_result_valid[0:PE_NUM-1];

    reg  [7:0]  cycle_cnt;
    reg         running;
    reg [DATA_WIDTH-1:0] qx_r, qy_r, qz_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_cnt <= 8'd0;
            running   <= 1'b0;
            qx_r      <= {DATA_WIDTH{1'b0}};
            qy_r      <= {DATA_WIDTH{1'b0}};
            qz_r      <= {DATA_WIDTH{1'b0}};
        end else begin
            if (start && !running) begin
                running   <= 1'b1;
                cycle_cnt <= 8'd1;
                qx_r      <= q_x;
                qy_r      <= q_y;
                qz_r      <= q_z;
            end else if (running) begin
                if (cycle_cnt == PE_NUM + 2)
                    running <= 1'b0;
                else
                    cycle_cnt <= cycle_cnt + 8'd1;
            end
        end
    end

    reg                   q0_valid;
    reg [DATA_WIDTH-1:0]  q0_coord;

    always @(*) begin
        case (cycle_cnt)
            8'd1:    begin q0_valid = running; q0_coord = qx_r; end
            8'd2:    begin q0_valid = running; q0_coord = qy_r; end
            8'd3:    begin q0_valid = running; q0_coord = qz_r; end
            default: begin q0_valid = 1'b0;    q0_coord = {DATA_WIDTH{1'b0}}; end
        endcase
    end

    assign q_valid_chain[0] = q0_valid;
    assign q_coord_chain[0] = q0_coord;

    genvar gi;
    generate
        for (gi = 0; gi < PE_NUM; gi = gi + 1) begin : gen_pe_chain
            wire        pe_d_sel_valid;
            wire [1:0]  pe_dim;
            wire [DATA_WIDTH-1:0] pe_d_coord;

            assign pe_d_sel_valid = running &&
                                    (cycle_cnt >= (gi + 1)) &&
                                    (cycle_cnt <= (gi + 3));

            assign pe_dim = cycle_cnt - gi - 1;

            assign pe_d_coord = (pe_dim == 2'd0) ? db_x_packed[gi*DATA_WIDTH +: DATA_WIDTH] :
                                (pe_dim == 2'd1) ? db_y_packed[gi*DATA_WIDTH +: DATA_WIDTH] :
                                                   db_z_packed[gi*DATA_WIDTH +: DATA_WIDTH];

            pe #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH (ACC_WIDTH)
            ) u_pe (
                .clk          (clk),
                .rst_n        (rst_n),
                .q_valid_in   (q_valid_chain[gi]),
                .q_coord_in   (q_coord_chain[gi]),
                .q_valid_out  (q_valid_chain[gi+1]),
                .q_coord_out  (q_coord_chain[gi+1]),
                .d_coord      (pe_d_coord),
                .dist_sq_out  (pe_dist_sq[gi]),
                .result_valid (pe_result_valid[gi])
            );
        end
    endgenerate

    reg [ACC_WIDTH-1:0]          dist_sq_out_r;
    reg                          result_valid_r;
    reg [$clog2(PE_NUM)-1:0]     result_idx_r;

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_sq_out_r  <= {ACC_WIDTH{1'b0}};
            result_valid_r <= 1'b0;
            result_idx_r   <= {$clog2(PE_NUM){1'b0}};
        end else begin
            result_valid_r <= 1'b0;
            for (k = 0; k < PE_NUM; k = k + 1) begin
                if (pe_result_valid[k]) begin
                    dist_sq_out_r  <= pe_dist_sq[k];
                    result_valid_r <= 1'b1;
                    result_idx_r   <= k[$clog2(PE_NUM)-1:0];
                end
            end
        end
    end

    assign dist_sq_out  = dist_sq_out_r;
    assign result_valid = result_valid_r;
    assign result_idx   = result_idx_r;

    reg done_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_r <= 1'b0;
        else
            done_r <= pe_result_valid[PE_NUM-1];
    end
    assign all_done = done_r;

endmodule