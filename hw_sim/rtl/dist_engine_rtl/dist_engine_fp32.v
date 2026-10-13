`timescale 1ns/1ps

module interp3nn_fp32_mul (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);
    wire        a_sign = a[31];
    wire [7:0]  a_exp  = a[30:23];
    wire [22:0] a_mant = a[22:0];
    wire        b_sign = b[31];
    wire [7:0]  b_exp  = b[30:23];
    wire [22:0] b_mant = b[22:0];
    wire a_is_zero = (a_exp == 8'd0) && (a_mant == 23'd0);
    wire b_is_zero = (b_exp == 8'd0) && (b_mant == 23'd0);
    wire a_is_inf  = (a_exp == 8'hFF) && (a_mant == 23'd0);
    wire b_is_inf  = (b_exp == 8'hFF) && (b_mant == 23'd0);
    wire        res_sign = a_sign ^ b_sign;
    wire [23:0] a_mant_ext = (a_exp == 8'd0) ? {1'b0, a_mant} : {1'b1, a_mant};
    wire [23:0] b_mant_ext = (b_exp == 8'd0) ? {1'b0, b_mant} : {1'b1, b_mant};
    wire [47:0] mant_product = a_mant_ext * b_mant_ext;
    wire [9:0]  exp_sum = {2'b0, a_exp} + {2'b0, b_exp};
    reg [31:0] result_comb; reg [9:0] res_exp_raw;
    always @(*) begin
        result_comb = 32'd0; res_exp_raw = 10'd0;
        if (a_is_zero || b_is_zero)      result_comb = {res_sign, 31'd0};
        else if (a_is_inf || b_is_inf)   result_comb = {res_sign, 8'hFF, 23'd0};
        else if (mant_product[47]) begin
            res_exp_raw = exp_sum - 10'd127 + 10'd1;
            if (res_exp_raw[9] || res_exp_raw == 10'd0)  result_comb = {res_sign, 31'd0};
            else if (res_exp_raw >= 10'd255)              result_comb = {res_sign, 8'hFF, 23'd0};
            else result_comb = {res_sign, res_exp_raw[7:0], mant_product[46:24]};
        end else begin
            res_exp_raw = exp_sum - 10'd127;
            if (res_exp_raw[9] || res_exp_raw == 10'd0)  result_comb = {res_sign, 31'd0};
            else if (res_exp_raw >= 10'd255)              result_comb = {res_sign, 8'hFF, 23'd0};
            else result_comb = {res_sign, res_exp_raw[7:0], mant_product[45:23]};
        end
    end
    assign result = result_comb;
endmodule

module interp3nn_fp32_add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);
    wire a_sign=a[31]; wire [7:0] a_exp=a[30:23]; wire [22:0] a_mant=a[22:0];
    wire b_sign=b[31]; wire [7:0] b_exp=b[30:23]; wire [22:0] b_mant=b[22:0];
    wire a_is_zero=(a_exp==8'd0)&&(a_mant==23'd0);
    wire b_is_zero=(b_exp==8'd0)&&(b_mant==23'd0);
    wire a_is_inf =(a_exp==8'hFF)&&(a_mant==23'd0);
    wire b_is_inf =(b_exp==8'hFF)&&(b_mant==23'd0);
    wire [23:0] a_mant_ext=(a_exp==8'd0)?{1'b0,a_mant}:{1'b1,a_mant};
    wire [23:0] b_mant_ext=(b_exp==8'd0)?{1'b0,b_mant}:{1'b1,b_mant};
    wire a_exp_ge=(a[30:0]>=b[30:0]);
    wire [7:0] exp_diff=a_exp_ge?(a_exp-b_exp):(b_exp-a_exp);
    wire [4:0] shift_amt=(exp_diff>8'd24)?5'd24:exp_diff[4:0];
    wire [24:0] large_mant=a_exp_ge?{1'b0,a_mant_ext}:{1'b0,b_mant_ext};
    wire [24:0] small_mant_shifted=a_exp_ge?
        ({1'b0,b_mant_ext}>>shift_amt):({1'b0,a_mant_ext}>>shift_amt);
    wire large_sign=a_exp_ge?a_sign:b_sign;
    wire small_sign=a_exp_ge?b_sign:a_sign;
    wire [7:0] aligned_exp=a_exp_ge?a_exp:b_exp;
    wire effective_sub=(large_sign^small_sign);
    wire [25:0] mant_sum=effective_sub?
        ({1'b0,large_mant}-{1'b0,small_mant_shifted}):
        ({1'b0,large_mant}+{1'b0,small_mant_shifted});
    wire res_sign=large_sign;
    wire [25:0] mant_abs=mant_sum;
    reg [4:0] lzc; reg [7:0] norm_exp; reg [22:0] norm_mant;
    reg [31:0] result_comb; reg [23:0] shifted_mant;
    always @(*) begin
        lzc=5'd0; norm_exp=8'd0; norm_mant=23'd0; result_comb=32'd0; shifted_mant=24'd0;
        if (a_is_zero&&b_is_zero) result_comb=32'd0;
        else if (a_is_zero) result_comb=b;
        else if (b_is_zero) result_comb=a;
        else if (a_is_inf)  result_comb=a;
        else if (b_is_inf)  result_comb=b;
        else if (mant_abs==26'd0) result_comb=32'd0;
        else if (mant_abs[24]) begin
            norm_exp=aligned_exp+8'd1; norm_mant=mant_abs[23:1];
            result_comb=(norm_exp==8'hFF)?{res_sign,8'hFF,23'd0}:{res_sign,norm_exp,norm_mant};
        end else begin
            if      (mant_abs[23]) lzc=0;  else if (mant_abs[22]) lzc=1;
            else if (mant_abs[21]) lzc=2;  else if (mant_abs[20]) lzc=3;
            else if (mant_abs[19]) lzc=4;  else if (mant_abs[18]) lzc=5;
            else if (mant_abs[17]) lzc=6;  else if (mant_abs[16]) lzc=7;
            else if (mant_abs[15]) lzc=8;  else if (mant_abs[14]) lzc=9;
            else if (mant_abs[13]) lzc=10; else if (mant_abs[12]) lzc=11;
            else if (mant_abs[11]) lzc=12; else if (mant_abs[10]) lzc=13;
            else if (mant_abs[9])  lzc=14; else if (mant_abs[8])  lzc=15;
            else if (mant_abs[7])  lzc=16; else if (mant_abs[6])  lzc=17;
            else if (mant_abs[5])  lzc=18; else if (mant_abs[4])  lzc=19;
            else if (mant_abs[3])  lzc=20; else if (mant_abs[2])  lzc=21;
            else if (mant_abs[1])  lzc=22; else lzc=23;
            if (aligned_exp<={3'd0,lzc}) result_comb=32'd0;
            else begin
                norm_exp=aligned_exp-{3'd0,lzc};
                shifted_mant=mant_abs[23:0]<<lzc;
                norm_mant=shifted_mant[22:0];
                result_comb={res_sign,norm_exp,norm_mant};
            end
        end
    end
    assign result = result_comb;
endmodule

module interp3nn_fp32_sub (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);
    wire [31:0] b_neg = {~b[31], b[30:0]};
    interp3nn_fp32_add u_add (.a(a), .b(b_neg), .result(result));
endmodule


// =============================================================================
// pe_fp32_pipe3 — 3-stage pipelined PE with cross-dimension overlap
//
// Pipeline stages run EVERY cycle on DIFFERENT dimensions simultaneously:
//
//   Cycle 1: STG1=SUB(x)
//   Cycle 2: STG1=SUB(y)   STG2=MUL(x)
//   Cycle 3: STG1=SUB(z)   STG2=MUL(y)   STG3=ADD(x)
//   Cycle 4:                STG2=MUL(z)   STG3=ADD(y)
//   Cycle 5:                               STG3=ADD(z) → result!
//
// Valid and dimension tags flow through the pipeline alongside data.
// q_valid/q_coord forwarding: 1-cycle delay (systolic spacing = 1).
// =============================================================================
module pe_fp32_pipe3 (
    input  wire        clk,
    input  wire        rst_n,

    // systolic chain input
    input  wire        q_valid_in,
    input  wire [31:0] q_coord_in,

    // systolic chain output (1-cycle delay)
    output reg         q_valid_out,
    output reg  [31:0] q_coord_out,

    // database point coordinates (fixed per PE)
    input  wire [31:0] d_x,
    input  wire [31:0] d_y,
    input  wire [31:0] d_z,

    // result
    output reg  [31:0] dist_sq_out,
    output reg         result_valid
);

    // ---- SUB stage dimension counter ----
    reg [1:0] sub_dim;    // 0=x, 1=y, 2=z

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            sub_dim <= 2'd0;
        else if (q_valid_in)
            sub_dim <= (sub_dim == 2'd2) ? 2'd0 : sub_dim + 2'd1;
    end

    // ---- d_sel MUX: select database coordinate for current SUB dimension ----
    wire [31:0] d_sel = (sub_dim == 2'd0) ? d_x :
                        (sub_dim == 2'd1) ? d_y : d_z;

    // ---- Systolic chain forwarding: 1-cycle delay ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_valid_out <= 1'b0;
            q_coord_out <= 32'h0;
        end else begin
            q_valid_out <= q_valid_in;
            q_coord_out <= q_coord_in;
        end
    end

    // =========================================================================
    //  Stage 1: SUB (combinational: diff = q - d)
    // =========================================================================
    wire [31:0] diff;
    interp3nn_fp32_sub u_sub (.a(q_coord_in), .b(d_sel), .result(diff));

    // Stage 1 output registers
    reg [31:0] diff_r;
    reg        stg1_valid;
    reg [1:0]  stg1_dim;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff_r     <= 32'h0;
            stg1_valid <= 1'b0;
            stg1_dim   <= 2'd0;
        end else begin
            stg1_valid <= q_valid_in;
            stg1_dim   <= sub_dim;
            if (q_valid_in)
                diff_r <= diff;
        end
    end

    // =========================================================================
    //  Stage 2: MUL (combinational: diff_sq = diff * diff)
    // =========================================================================
    wire [31:0] diff_sq;
    interp3nn_fp32_mul u_mul (.a(diff_r), .b(diff_r), .result(diff_sq));

    // Stage 2 output registers
    reg [31:0] diff_sq_r;
    reg        stg2_valid;
    reg [1:0]  stg2_dim;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff_sq_r  <= 32'h0;
            stg2_valid <= 1'b0;
            stg2_dim   <= 2'd0;
        end else begin
            stg2_valid <= stg1_valid;
            stg2_dim   <= stg1_dim;
            if (stg1_valid)
                diff_sq_r <= diff_sq;
        end
    end

    // =========================================================================
    //  Stage 3: ADD (combinational: acc = acc_reg + diff_sq)
    // =========================================================================
    reg [31:0] acc_reg;
    wire [31:0] acc_plus;
    interp3nn_fp32_add u_add (.a(acc_reg), .b(diff_sq_r), .result(acc_plus));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg      <= 32'h0;
            dist_sq_out  <= 32'h0;
            result_valid <= 1'b0;
        end else begin
            result_valid <= 1'b0;
            if (stg2_valid) begin
                case (stg2_dim)
                    2'd0: acc_reg     <= diff_sq_r;   // x: initialize
                    2'd1: acc_reg     <= acc_plus;     // y: accumulate
                    2'd2: begin                        // z: output final
                        dist_sq_out  <= acc_plus;
                        result_valid <= 1'b1;
                    end
                    default: acc_reg  <= 32'h0;
                endcase
            end
        end
    end

endmodule


// =============================================================================
// distance_engine_fp32 — Systolic engine with overlapped 3-stage PEs
//
// Coordinator injects qx/qy/qz on consecutive cycles (cnt=1,2,3), same as
// the original 1-stage design. Systolic spacing = 1 cycle.
//
// Timing:
//   PE[i] engine output:  cycle i + 7
//   Batch latency:        (PE_NUM - 1) + 7 cycles
//   PE_NUM=64:            70 cycles/batch
//   Critical path:        ~4ns (single FP32 op)
// =============================================================================
module dist_engine_fp32 #(
    parameter PE_NUM = 64
)(
    input  wire                             clk,
    input  wire                             rst_n,

    input  wire                             start,

    input  wire [31:0]                      q_x,
    input  wire [31:0]                      q_y,
    input  wire [31:0]                      q_z,

    input  wire [PE_NUM*32-1:0]             db_x_packed,
    input  wire [PE_NUM*32-1:0]             db_y_packed,
    input  wire [PE_NUM*32-1:0]             db_z_packed,

    output wire [31:0]                      dist_sq_out,
    output wire                             result_valid,
    output wire [$clog2(PE_NUM)-1:0]        result_idx,
    output wire                             all_done
);

    // ---- Systolic chain wires ----
    wire                   q_valid_chain [0:PE_NUM];
    wire [31:0]            q_coord_chain [0:PE_NUM];
    wire [31:0]            pe_dist_sq    [0:PE_NUM-1];
    wire                   pe_result_valid[0:PE_NUM-1];

    // =========================================================================
    //  Block A: Scheduler FSM
    //  Injects qx/qy/qz on 3 consecutive cycles (spacing = 1, same as original)
    // =========================================================================
    reg  [7:0]  cycle_cnt;
    reg         running;
    reg  [31:0] qx_r, qy_r, qz_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_cnt <= 8'd0;
            running   <= 1'b0;
            qx_r      <= 32'h0;
            qy_r      <= 32'h0;
            qz_r      <= 32'h0;
        end else begin
            if (start && !running) begin
                running   <= 1'b1;
                cycle_cnt <= 8'd1;
                qx_r      <= q_x;
                qy_r      <= q_y;
                qz_r      <= q_z;
            end else if (running) begin
                cycle_cnt <= cycle_cnt + 8'd1;
            end
            // stop when done
            if (all_done)
                running <= 1'b0;
        end
    end

    // =========================================================================
    //  Block B: Coordinate injector (consecutive: cnt=1 qx, cnt=2 qy, cnt=3 qz)
    // =========================================================================
    reg                   q0_valid;
    reg [31:0]            q0_coord;

    always @(*) begin
        q0_valid = 1'b0;
        q0_coord = 32'h0;
        if (running) begin
            case (cycle_cnt)
                8'd1: begin q0_valid = 1'b1; q0_coord = qx_r; end
                8'd2: begin q0_valid = 1'b1; q0_coord = qy_r; end
                8'd3: begin q0_valid = 1'b1; q0_coord = qz_r; end
                default: ;
            endcase
        end
    end

    assign q_valid_chain[0] = q0_valid;
    assign q_coord_chain[0] = q0_coord;

    // =========================================================================
    //  Block C: PE systolic array (overlapped 3-stage pipeline PEs)
    // =========================================================================
    genvar gi;
    generate
        for (gi = 0; gi < PE_NUM; gi = gi + 1) begin : gen_pe

            pe_fp32_pipe3 u_pe (
                .clk          (clk),
                .rst_n        (rst_n),
                .q_valid_in   (q_valid_chain[gi]),
                .q_coord_in   (q_coord_chain[gi]),
                .q_valid_out  (q_valid_chain[gi+1]),
                .q_coord_out  (q_coord_chain[gi+1]),
                .d_x          (db_x_packed[gi*32 +: 32]),
                .d_y          (db_y_packed[gi*32 +: 32]),
                .d_z          (db_z_packed[gi*32 +: 32]),
                .dist_sq_out  (pe_dist_sq[gi]),
                .result_valid (pe_result_valid[gi])
            );
        end
    endgenerate

    // =========================================================================
    //  Block D: Output arbiter
    // =========================================================================
    reg [31:0]                  dist_sq_out_r;
    reg                         result_valid_r;
    reg [$clog2(PE_NUM)-1:0]    result_idx_r;

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_sq_out_r  <= 32'h0;
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

    // ---- all_done: 1 cycle after last PE's result_valid ----
    reg done_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_r <= 1'b0;
        else
            done_r <= pe_result_valid[PE_NUM-1];
    end
    assign all_done = done_r;

endmodule