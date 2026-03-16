

module affine_norm #(
    parameter DATA_W   = 32,
    parameter P        = 32,
    parameter D        = 32,
    parameter S        = 512,
    parameter K        = 24,
    parameter [31:0] FP_INV_L = 32'h3489_7595,
    parameter ADDR_W   = $clog2(S * K * D / P)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    start,
    output reg                     done,
    output reg  [1:0]              phase,

    output reg  [ADDR_W-1:0]      gp_rd_addr,
    output reg                     gp_rd_req,
    input  wire [P*DATA_W-1:0]    gp_rd_data,
    input  wire                    gp_rd_valid,

    input  wire [P*DATA_W-1:0]    anc_data,

    input  wire [P*DATA_W-1:0]    alpha_data,
    input  wire [P*DATA_W-1:0]    beta_data,

    output wire [ADDR_W-1:0]      out_wr_addr,
    output wire [P*DATA_W-1:0]    out_wr_data,
    output wire                    out_wr_en
);

    localparam L             = S * K * D;
    localparam BEATS_TOTAL   = L / P;
    localparam P_W           = P * DATA_W;
    localparam TREE_DEPTH    = $clog2(P);
    localparam DRAIN1_CY     = TREE_DEPTH + 3;
    localparam DRAIN2_CY     = 4;
    localparam [31:0] FP_EPS  = 32'h3727_C5AC;
    localparam [31:0] FP_ZERO = 32'h0000_0000;

    localparam [2:0]
        ST_IDLE   = 3'd0,
        ST_ACCUM  = 3'd1,
        ST_DRAIN1 = 3'd2,
        ST_GAP    = 3'd3,
        ST_NORM   = 3'd4,
        ST_DRAIN2 = 3'd5,
        ST_DONE   = 3'd6;

    reg [2:0]  state;
    reg [4:0]  gap_step;
    reg [3:0]  drain_cnt;
    reg [ADDR_W-1:0] beat_cnt;

    wire beat_last  = (beat_cnt == BEATS_TOTAL - 1);
    wire beat_valid = gp_rd_valid && (state == ST_ACCUM || state == ST_NORM);

    wire [P_W-1:0] anc_neg;
    genvar ni;
    generate
        for (ni = 0; ni < P; ni = ni + 1) begin : gen_anc_neg
            assign anc_neg[ni*32 +: 32] = {~anc_data[ni*32+31], anc_data[ni*32 +: 31]};
        end
    endgenerate

    reg [P_W-1:0] alpha_d1, alpha_d2;
    reg [P_W-1:0] beta_d1, beta_d2, beta_d3;

    always @(posedge clk) begin
        alpha_d1 <= alpha_data;
        alpha_d2 <= alpha_d1;
        beta_d1  <= beta_data;
        beta_d2  <= beta_d1;
        beta_d3  <= beta_d2;
    end

    reg [31:0] gap_add_a, gap_add_b;
    reg        gap_add_v;
    reg [31:0] gap_mul_a, gap_mul_b;
    reg        gap_mul_v;

    reg [31:0] sum_r_final, sum_r2_final;
    reg [31:0] mu_reg, mean_r2_reg, mu2_reg, var_reg, var_eps_reg;
    reg [31:0] inv_std_reg;

    wire [P_W-1:0] sub_r_bus;
    wire [P-1:0]   sub_d_bus;

    genvar gi;
    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : gen_sub
            wire [31:0] sa = (gi == 0 && phase == 2'd1) ? gap_add_a : gp_rd_data[gi*32 +: 32];
            wire [31:0] sb = (gi == 0 && phase == 2'd1) ? gap_add_b : anc_neg[gi*32 +: 32];
            wire        sv = (gi == 0 && phase == 2'd1) ? gap_add_v : beat_valid;

            fp32_add u_sub (
                .clk(clk), .rst_n(rst_n),
                .a(sa), .b(sb), .valid_in(sv),
                .result(sub_r_bus[gi*32 +: 32]),
                .valid_out(sub_d_bus[gi])
            );
        end
    endgenerate

    wire [P_W-1:0] mul_r_bus;
    wire [P-1:0]   mul_d_bus;

    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : gen_mul
            wire [31:0] ma = (gi == 0 && phase == 2'd1) ? gap_mul_a : sub_r_bus[gi*32 +: 32];
            wire [31:0] mb;
            wire        mv = (gi == 0 && phase == 2'd1) ? gap_mul_v : sub_d_bus[gi];

            assign mb = (gi == 0 && phase == 2'd1) ? gap_mul_b
                      : (phase == 2'd2)            ? inv_std_reg
                      :                              sub_r_bus[gi*32 +: 32];

            fp32_mul u_mul (
                .clk(clk), .rst_n(rst_n),
                .a(ma), .b(mb), .valid_in(mv),
                .result(mul_r_bus[gi*32 +: 32]),
                .valid_out(mul_d_bus[gi])
            );
        end
    endgenerate

    wire [31:0] tree_r_sum, tree_r2_sum;
    wire        tree_r_valid, tree_r2_valid;

    fp32_reduce_tree u_tree_r (
        .clk(clk), .rst_n(rst_n),
        .x_in(sub_r_bus),
        .valid_in(sub_d_bus[0]),
        .sum_out(tree_r_sum),
        .valid_out(tree_r_valid)
    );

    fp32_reduce_tree u_tree_r2 (
        .clk(clk), .rst_n(rst_n),
        .x_in(mul_r_bus),
        .valid_in(mul_d_bus[0]),
        .sum_out(tree_r2_sum),
        .valid_out(tree_r2_valid)
    );

    reg        acc_sel_r, acc_sel_r2;
    reg [31:0] acc_r_even, acc_r_odd;
    reg [31:0] acc_r2_even, acc_r2_odd;

    wire [31:0] acc_re_new; wire acc_re_v;
    fp32_add u_acc_re (.clk(clk), .rst_n(rst_n),
        .a(acc_r_even), .b(tree_r_sum),
        .valid_in(tree_r_valid & ~acc_sel_r & (phase == 2'd0)),
        .result(acc_re_new), .valid_out(acc_re_v));

    wire [31:0] acc_ro_new; wire acc_ro_v;
    fp32_add u_acc_ro (.clk(clk), .rst_n(rst_n),
        .a(acc_r_odd), .b(tree_r_sum),
        .valid_in(tree_r_valid & acc_sel_r & (phase == 2'd0)),
        .result(acc_ro_new), .valid_out(acc_ro_v));

    wire [31:0] acc_r2e_new; wire acc_r2e_v;
    fp32_add u_acc_r2e (.clk(clk), .rst_n(rst_n),
        .a(acc_r2_even), .b(tree_r2_sum),
        .valid_in(tree_r2_valid & ~acc_sel_r2 & (phase == 2'd0)),
        .result(acc_r2e_new), .valid_out(acc_r2e_v));

    wire [31:0] acc_r2o_new; wire acc_r2o_v;
    fp32_add u_acc_r2o (.clk(clk), .rst_n(rst_n),
        .a(acc_r2_odd), .b(tree_r2_sum),
        .valid_in(tree_r2_valid & acc_sel_r2 & (phase == 2'd0)),
        .result(acc_r2o_new), .valid_out(acc_r2o_v));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_r_even <= FP_ZERO; acc_r_odd <= FP_ZERO;
            acc_r2_even <= FP_ZERO; acc_r2_odd <= FP_ZERO;
            acc_sel_r <= 1'b0; acc_sel_r2 <= 1'b0;
        end else begin
            if (acc_re_v)  acc_r_even  <= acc_re_new;
            if (acc_ro_v)  acc_r_odd   <= acc_ro_new;
            if (acc_r2e_v) acc_r2_even <= acc_r2e_new;
            if (acc_r2o_v) acc_r2_odd  <= acc_r2o_new;
            if (tree_r_valid  && phase == 2'd0) acc_sel_r  <= ~acc_sel_r;
            if (tree_r2_valid && phase == 2'd0) acc_sel_r2 <= ~acc_sel_r2;
            if (state == ST_IDLE && start) begin
                acc_r_even <= FP_ZERO; acc_r_odd <= FP_ZERO;
                acc_r2_even <= FP_ZERO; acc_r2_odd <= FP_ZERO;
                acc_sel_r <= 1'b0; acc_sel_r2 <= 1'b0;
            end
        end
    end

    wire [P_W-1:0] aff_mul_r_bus, aff_add_r_bus;
    wire [P-1:0]   aff_mul_d_bus, aff_add_d_bus;

    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : gen_aff
            fp32_mul u_aff_mul (
                .clk(clk), .rst_n(rst_n),
                .a(alpha_d2[gi*32 +: 32]),
                .b(mul_r_bus[gi*32 +: 32]),
                .valid_in(mul_d_bus[gi] & (phase == 2'd2)),
                .result(aff_mul_r_bus[gi*32 +: 32]),
                .valid_out(aff_mul_d_bus[gi])
            );
            fp32_add u_aff_add (
                .clk(clk), .rst_n(rst_n),
                .a(aff_mul_r_bus[gi*32 +: 32]),
                .b(beta_d3[gi*32 +: 32]),
                .valid_in(aff_mul_d_bus[gi]),
                .result(aff_add_r_bus[gi*32 +: 32]),
                .valid_out(aff_add_d_bus[gi])
            );
        end
    endgenerate

    reg  [31:0] rsqrt_x;
    reg         rsqrt_v;
    wire [31:0] rsqrt_y;
    wire        rsqrt_done;

    fp32_rsqrt u_rsqrt (
        .clk(clk), .rst_n(rst_n),
        .x_in(rsqrt_x), .valid_in(rsqrt_v),
        .y_out(rsqrt_y), .valid_out(rsqrt_done)
    );

    reg [DRAIN2_CY-1:0] wr_valid_pipe;
    reg [ADDR_W-1:0]    wr_addr_d [0:DRAIN2_CY-1];
    integer wi;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_valid_pipe <= 0;
            for (wi = 0; wi < DRAIN2_CY; wi = wi + 1)
                wr_addr_d[wi] <= 0;
        end else begin
            wr_valid_pipe <= {wr_valid_pipe[DRAIN2_CY-2:0],
                              beat_valid & (phase == 2'd2)};
            wr_addr_d[0] <= beat_cnt;
            for (wi = 1; wi < DRAIN2_CY; wi = wi + 1)
                wr_addr_d[wi] <= wr_addr_d[wi-1];
        end
    end

    assign out_wr_en   = wr_valid_pipe[DRAIN2_CY-1];
    assign out_wr_data = aff_add_r_bus;
    assign out_wr_addr = wr_addr_d[DRAIN2_CY-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_IDLE;
            phase      <= 2'd0;
            done       <= 1'b0;
            beat_cnt   <= 0;
            gap_step   <= 5'd0;
            drain_cnt  <= 4'd0;
            gp_rd_addr <= 0;
            gp_rd_req  <= 1'b0;
            gap_add_v  <= 1'b0;
            gap_mul_v  <= 1'b0;
            rsqrt_v    <= 1'b0;
            inv_std_reg <= FP_ZERO;
        end else begin
            done      <= 1'b0;
            gap_add_v <= 1'b0;
            gap_mul_v <= 1'b0;
            rsqrt_v   <= 1'b0;
            gp_rd_req <= 1'b0;

            case (state)

                ST_IDLE: begin
                    if (start) begin
                        state      <= ST_ACCUM;
                        phase      <= 2'd0;
                        beat_cnt   <= 0;
                        gp_rd_req  <= 1'b1;
                        gp_rd_addr <= 0;
                    end
                end

                ST_ACCUM: begin
                    if (gp_rd_valid) begin
                        beat_cnt <= beat_cnt + 1;
                        if (beat_last) begin
                            state     <= ST_DRAIN1;
                            drain_cnt <= 0;
                            gp_rd_req <= 1'b0;
                        end else begin
                            gp_rd_addr <= beat_cnt + 1;
                            gp_rd_req  <= 1'b1;
                        end
                    end
                end

                ST_DRAIN1: begin
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == DRAIN1_CY[3:0] - 1) begin
                        state    <= ST_GAP;
                        phase    <= 2'd1;
                        gap_step <= 5'd0;
                    end
                end

                ST_GAP: begin
                    case (gap_step)
                        5'd0: begin
                            gap_add_a <= acc_r_even; gap_add_b <= acc_r_odd;
                            gap_add_v <= 1'b1; gap_step <= 5'd1;
                        end
                        5'd1: if (sub_d_bus[0]) begin
                            sum_r_final <= sub_r_bus[0*32 +: 32]; gap_step <= 5'd2;
                        end
                        5'd2: begin
                            gap_add_a <= acc_r2_even; gap_add_b <= acc_r2_odd;
                            gap_add_v <= 1'b1; gap_step <= 5'd3;
                        end
                        5'd3: if (sub_d_bus[0]) begin
                            sum_r2_final <= sub_r_bus[0*32 +: 32]; gap_step <= 5'd4;
                        end
                        5'd4: begin
                            gap_mul_a <= sum_r_final; gap_mul_b <= FP_INV_L;
                            gap_mul_v <= 1'b1; gap_step <= 5'd5;
                        end
                        5'd5: if (mul_d_bus[0]) begin
                            mu_reg <= mul_r_bus[0*32 +: 32]; gap_step <= 5'd6;
                        end
                        5'd6: begin
                            gap_mul_a <= sum_r2_final; gap_mul_b <= FP_INV_L;
                            gap_mul_v <= 1'b1; gap_step <= 5'd7;
                        end
                        5'd7: if (mul_d_bus[0]) begin
                            mean_r2_reg <= mul_r_bus[0*32 +: 32]; gap_step <= 5'd8;
                        end
                        5'd8: begin
                            gap_mul_a <= mu_reg; gap_mul_b <= mu_reg;
                            gap_mul_v <= 1'b1; gap_step <= 5'd9;
                        end
                        5'd9: if (mul_d_bus[0]) begin
                            mu2_reg <= mul_r_bus[0*32 +: 32]; gap_step <= 5'd10;
                        end
                        5'd10: begin
                            gap_add_a <= mean_r2_reg;
                            gap_add_b <= {~mu2_reg[31], mu2_reg[30:0]};
                            gap_add_v <= 1'b1; gap_step <= 5'd11;
                        end
                        5'd11: if (sub_d_bus[0]) begin
                            var_reg <= sub_r_bus[0*32 +: 32]; gap_step <= 5'd12;
                        end
                        5'd12: begin
                            gap_add_a <= var_reg; gap_add_b <= FP_EPS;
                            gap_add_v <= 1'b1; gap_step <= 5'd13;
                        end
                        5'd13: if (sub_d_bus[0]) begin
                            var_eps_reg <= sub_r_bus[0*32 +: 32]; gap_step <= 5'd14;
                        end
                        5'd14: begin
                            rsqrt_x <= var_eps_reg; rsqrt_v <= 1'b1;
                            gap_step <= 5'd15;
                        end
                        5'd15: if (rsqrt_done) begin
                            inv_std_reg <= rsqrt_y; gap_step <= 5'd16;
                        end
                        5'd16: begin
                            state      <= ST_NORM;
                            phase      <= 2'd2;
                            beat_cnt   <= 0;
                            gp_rd_req  <= 1'b1;
                            gp_rd_addr <= 0;
                        end
                        default: gap_step <= 5'd0;
                    endcase
                end

                ST_NORM: begin
                    if (gp_rd_valid) begin
                        beat_cnt <= beat_cnt + 1;
                        if (beat_last) begin
                            state     <= ST_DRAIN2;
                            drain_cnt <= 0;
                            gp_rd_req <= 1'b0;
                        end else begin
                            gp_rd_addr <= beat_cnt + 1;
                            gp_rd_req  <= 1'b1;
                        end
                    end
                end

                ST_DRAIN2: begin
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == DRAIN2_CY[3:0] - 1)
                        state <= ST_DONE;
                end

                ST_DONE: begin
                    done  <= 1'b1;
                    state <= ST_IDLE;
                    phase <= 2'd0;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule