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

    reg [31:0] result_comb;
    reg [9:0]  res_exp_raw;

    always @(*) begin
        result_comb = 32'd0;
        if (a_is_zero || b_is_zero) begin
            result_comb = {res_sign, 31'd0};
        end else if (a_is_inf || b_is_inf) begin
            result_comb = {res_sign, 8'hFF, 23'd0};
        end else if (mant_product[47]) begin
            res_exp_raw = exp_sum - 10'd127 + 10'd1;
            if (res_exp_raw[9] || res_exp_raw == 10'd0)
                result_comb = {res_sign, 31'd0};
            else if (res_exp_raw >= 10'd255)
                result_comb = {res_sign, 8'hFF, 23'd0};
            else
                result_comb = {res_sign, res_exp_raw[7:0], mant_product[46:24]};
        end else begin
            res_exp_raw = exp_sum - 10'd127;
            if (res_exp_raw[9] || res_exp_raw == 10'd0)
                result_comb = {res_sign, 31'd0};
            else if (res_exp_raw >= 10'd255)
                result_comb = {res_sign, 8'hFF, 23'd0};
            else
                result_comb = {res_sign, res_exp_raw[7:0], mant_product[45:23]};
        end
    end

    assign result = result_comb;
endmodule


module interp3nn_fp32_add (
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

    wire [23:0] a_mant_ext = (a_exp == 8'd0) ? {1'b0, a_mant} : {1'b1, a_mant};
    wire [23:0] b_mant_ext = (b_exp == 8'd0) ? {1'b0, b_mant} : {1'b1, b_mant};

    wire        a_exp_ge = (a[30:0] >= b[30:0]);
    wire [7:0]  exp_diff = a_exp_ge ? (a_exp - b_exp) : (b_exp - a_exp);
    wire [4:0]  shift_amt = (exp_diff > 8'd24) ? 5'd24 : exp_diff[4:0];

    wire [24:0] large_mant = a_exp_ge ? {1'b0, a_mant_ext} : {1'b0, b_mant_ext};
    wire [24:0] small_mant_shifted = a_exp_ge ?
                    ({1'b0, b_mant_ext} >> shift_amt) :
                    ({1'b0, a_mant_ext} >> shift_amt);

    wire        large_sign = a_exp_ge ? a_sign : b_sign;
    wire        small_sign = a_exp_ge ? b_sign : a_sign;
    wire [7:0]  aligned_exp = a_exp_ge ? a_exp : b_exp;

    wire        effective_sub = (large_sign ^ small_sign);
    wire [25:0] mant_sum = effective_sub ?
                    ({1'b0, large_mant} - {1'b0, small_mant_shifted}) :
                    ({1'b0, large_mant} + {1'b0, small_mant_shifted});

    wire        res_sign = large_sign;
    wire [25:0] mant_abs = mant_sum;

    reg [4:0]  leading_zero_count;
    reg [7:0]  norm_exp;
    reg [22:0] norm_mant;
    reg [31:0] result_comb;
    reg [23:0] shifted_mant;

    always @(*) begin
        leading_zero_count = 5'd0;
        norm_exp  = 8'd0;
        norm_mant = 23'd0;
        result_comb  = 32'd0;
        shifted_mant = 24'd0;

        if (a_is_zero && b_is_zero) begin
            result_comb = 32'd0;
        end else if (a_is_zero) begin
            result_comb = b;
        end else if (b_is_zero) begin
            result_comb = a;
        end else if (a_is_inf) begin
            result_comb = a;
        end else if (b_is_inf) begin
            result_comb = b;
        end else if (mant_abs == 26'd0) begin
            result_comb = 32'd0;
        end else if (mant_abs[24]) begin
            norm_exp  = aligned_exp + 8'd1;
            norm_mant = mant_abs[23:1];
            if (norm_exp == 8'hFF)
                result_comb = {res_sign, 8'hFF, 23'd0};
            else
                result_comb = {res_sign, norm_exp, norm_mant};
        end else begin
            if      (mant_abs[23]) leading_zero_count = 5'd0;
            else if (mant_abs[22]) leading_zero_count = 5'd1;
            else if (mant_abs[21]) leading_zero_count = 5'd2;
            else if (mant_abs[20]) leading_zero_count = 5'd3;
            else if (mant_abs[19]) leading_zero_count = 5'd4;
            else if (mant_abs[18]) leading_zero_count = 5'd5;
            else if (mant_abs[17]) leading_zero_count = 5'd6;
            else if (mant_abs[16]) leading_zero_count = 5'd7;
            else if (mant_abs[15]) leading_zero_count = 5'd8;
            else if (mant_abs[14]) leading_zero_count = 5'd9;
            else if (mant_abs[13]) leading_zero_count = 5'd10;
            else if (mant_abs[12]) leading_zero_count = 5'd11;
            else if (mant_abs[11]) leading_zero_count = 5'd12;
            else if (mant_abs[10]) leading_zero_count = 5'd13;
            else if (mant_abs[9])  leading_zero_count = 5'd14;
            else if (mant_abs[8])  leading_zero_count = 5'd15;
            else if (mant_abs[7])  leading_zero_count = 5'd16;
            else if (mant_abs[6])  leading_zero_count = 5'd17;
            else if (mant_abs[5])  leading_zero_count = 5'd18;
            else if (mant_abs[4])  leading_zero_count = 5'd19;
            else if (mant_abs[3])  leading_zero_count = 5'd20;
            else if (mant_abs[2])  leading_zero_count = 5'd21;
            else if (mant_abs[1])  leading_zero_count = 5'd22;
            else                   leading_zero_count = 5'd23;

            if (aligned_exp <= {3'd0, leading_zero_count}) begin
                result_comb = 32'd0;
            end else begin
                norm_exp     = aligned_exp - {3'd0, leading_zero_count};
                shifted_mant = mant_abs[23:0] << leading_zero_count;
                norm_mant    = shifted_mant[22:0];
                result_comb  = {res_sign, norm_exp, norm_mant};
            end
        end
    end

    assign result = result_comb;
endmodule


module fp32_div (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire        valid_in,
    output reg  [31:0] result,
    output reg         valid_out
);
    localparam S_IDLE     = 3'd0;
    localparam S_SPECIAL  = 3'd1;
    localparam S_ITER_MUL = 3'd2;
    localparam S_ITER_UPD = 3'd3;
    localparam S_QUOT     = 3'd4;
    localparam S_NORM     = 3'd5;

    reg [2:0]  state;
    reg [1:0]  iter_count;
    reg [31:0] a_reg;
    reg [31:0] b_reg;
    reg        sign_reg;
    reg [9:0]  exp_reg;
    reg [24:0] a_norm;
    reg [24:0] b_norm;
    reg [24:0] r_est;
    reg [24:0] f_reg;

    wire a_is_zero = (a_reg[30:23] == 8'd0)  && (a_reg[22:0] == 23'd0);
    wire b_is_zero = (b_reg[30:23] == 8'd0)  && (b_reg[22:0] == 23'd0);
    wire a_is_inf  = (a_reg[30:23] == 8'hFF) && (a_reg[22:0] == 23'd0);
    wire b_is_inf  = (b_reg[30:23] == 8'hFF) && (b_reg[22:0] == 23'd0);
    wire a_is_nan  = (a_reg[30:23] == 8'hFF) && (a_reg[22:0] != 23'd0);
    wire b_is_nan  = (b_reg[30:23] == 8'hFF) && (b_reg[22:0] != 23'd0);

    wire [49:0] mul_br = {25'b0, b_norm} * {25'b0, r_est};
    wire [49:0] mul_ar = {25'b0, a_norm} * {25'b0, r_est};
    wire [49:0] mul_fr = {25'b0, f_reg}  * {25'b0, r_est};

    wire [25:0] two_fp   = 26'h2000000;
    wire [25:0] br_trunc = {1'b0, mul_br[48:24]};
    wire [25:0] f_val    = two_fp - br_trunc;

    reg [9:0]  final_exp;
    reg [22:0] final_mant;

    always @(*) begin
        if (a_norm[24]) begin
            final_exp  = exp_reg;
            final_mant = a_norm[23:1];
        end else if (a_norm[23]) begin
            final_exp  = exp_reg - 10'd1;
            final_mant = a_norm[22:0];
        end else begin
            final_exp  = exp_reg - 10'd2;
            final_mant = {a_norm[21:0], 1'b0};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            result     <= 32'd0;
            valid_out  <= 1'b0;
            iter_count <= 2'd0;
            a_reg      <= 32'd0;
            b_reg      <= 32'd0;
            sign_reg   <= 1'b0;
            exp_reg    <= 10'd0;
            a_norm     <= 25'd0;
            b_norm     <= 25'd0;
            r_est      <= 25'd0;
            f_reg      <= 25'd0;
        end else begin
            valid_out <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (valid_in) begin
                        a_reg    <= a;
                        b_reg    <= b;
                        sign_reg <= a[31] ^ b[31];
                        state    <= S_SPECIAL;
                    end
                end

                S_SPECIAL: begin
                    if (a_is_nan || b_is_nan || (a_is_inf && b_is_inf)) begin
                        result    <= {sign_reg, 8'hFF, 23'h400000};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (a_is_zero && b_is_zero) begin
                        result    <= 32'd0;
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (a_is_zero) begin
                        result    <= {sign_reg, 31'd0};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (b_is_zero) begin
                        result    <= {sign_reg, 8'hFF, 23'd0};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (a_is_inf) begin
                        result    <= {sign_reg, 8'hFF, 23'd0};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (b_is_inf) begin
                        result    <= {sign_reg, 31'd0};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else begin
                        exp_reg    <= {2'b0, a_reg[30:23]} - {2'b0, b_reg[30:23]} + 10'd127;
                        a_norm     <= {1'b1, a_reg[22:0], 1'b0};
                        b_norm     <= {1'b1, b_reg[22:0], 1'b0};
                        r_est      <= 25'h1000000 - {2'b00, b_reg[22:0]};
                        iter_count <= 2'd0;
                        state      <= S_ITER_MUL;
                    end
                end

                S_ITER_MUL: begin
                    f_reg <= f_val[24:0];
                    state <= S_ITER_UPD;
                end

                S_ITER_UPD: begin
                    r_est <= mul_fr[48:24];
                    if (iter_count == 2'd2)
                        state <= S_QUOT;
                    else begin
                        iter_count <= iter_count + 2'd1;
                        state      <= S_ITER_MUL;
                    end
                end

                S_QUOT: begin
                    a_norm <= mul_ar[48:24];
                    state  <= S_NORM;
                end

                S_NORM: begin
                    if (final_exp[9] || final_exp == 10'd0)
                        result <= {sign_reg, 31'd0};
                    else if (final_exp >= 10'd255)
                        result <= {sign_reg, 8'hFF, 23'd0};
                    else
                        result <= {sign_reg, final_exp[7:0], final_mant};

                    valid_out <= 1'b1;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule


module Sum3Adder (
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    output wire [31:0] sum
);
    wire [31:0] ab;
    interp3nn_fp32_add u_ab  (.a(a),  .b(b), .result(ab));
    interp3nn_fp32_add u_abc (.a(ab), .b(c), .result(sum));
endmodule


module ReciprocalUnit #(
    parameter DIV_LAT = 14
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [31:0] dist_sq_0,
    input  wire [31:0] dist_sq_1,
    input  wire [31:0] dist_sq_2,
    input  wire [31:0] eps_fp32,
    output reg         busy,
    output reg         valid_out,
    output reg  [31:0] weight_0,
    output reg  [31:0] weight_1,
    output reg  [31:0] weight_2
);
    localparam ONE = 32'h3f800000;

    reg [1:0] phase;

    wire [31:0] de0, de1, de2;
    interp3nn_fp32_add u_e0 (.a(dist_sq_0), .b(eps_fp32), .result(de0));
    interp3nn_fp32_add u_e1 (.a(dist_sq_1), .b(eps_fp32), .result(de1));
    interp3nn_fp32_add u_e2 (.a(dist_sq_2), .b(eps_fp32), .result(de2));

    reg        dv0, dv1, dv2;
    wire       dq0, dq1, dq2;
    wire [31:0] rc0, rc1, rc2;

    fp32_div u_r0 (.clk(clk), .rst_n(rst_n), .valid_in(dv0),
                   .a(ONE), .b(de0), .valid_out(dq0), .result(rc0));
    fp32_div u_r1 (.clk(clk), .rst_n(rst_n), .valid_in(dv1),
                   .a(ONE), .b(de1), .valid_out(dq1), .result(rc1));
    fp32_div u_r2 (.clk(clk), .rst_n(rst_n), .valid_in(dv2),
                   .a(ONE), .b(de2), .valid_out(dq2), .result(rc2));

    reg [31:0] r_hold0, r_hold1, r_hold2;
    reg [2:0]  got_rcp;

    wire [31:0] rsum;
    Sum3Adder u_sum (.a(r_hold0), .b(r_hold1), .c(r_hold2), .sum(rsum));

    reg        dvs;
    wire       dqs;
    wire [31:0] inv_sum;
    fp32_div u_inv (.clk(clk), .rst_n(rst_n), .valid_in(dvs),
                    .a(ONE), .b(rsum), .valid_out(dqs), .result(inv_sum));

    wire [31:0] w0_c, w1_c, w2_c;
    interp3nn_fp32_mul u_w0 (.a(r_hold0), .b(inv_sum), .result(w0_c));
    interp3nn_fp32_mul u_w1 (.a(r_hold1), .b(inv_sum), .result(w1_c));
    interp3nn_fp32_mul u_w2 (.a(r_hold2), .b(inv_sum), .result(w2_c));

    wire all_rcp_done = (got_rcp[0] | dq0) & (got_rcp[1] | dq1) & (got_rcp[2] | dq2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase     <= 2'd0;
            busy      <= 1'b0;
            valid_out <= 1'b0;
            dv0 <= 0; dv1 <= 0; dv2 <= 0; dvs <= 0;
            r_hold0 <= 0; r_hold1 <= 0; r_hold2 <= 0;
            got_rcp  <= 3'b0;
            weight_0 <= 0; weight_1 <= 0; weight_2 <= 0;
        end else begin
            dv0 <= 1'b0;
            dv1 <= 1'b0;
            dv2 <= 1'b0;
            dvs <= 1'b0;
            valid_out <= 1'b0;

            case (phase)
                2'd0: begin
                    if (start) begin
                        busy    <= 1'b1;
                        got_rcp <= 3'b0;
                        dv0     <= 1'b1;
                        dv1     <= 1'b1;
                        dv2     <= 1'b1;
                        phase   <= 2'd1;
                    end
                end

                2'd1: begin
                    if (dq0) begin
                        r_hold0    <= rc0;
                        got_rcp[0] <= 1'b1;
                    end
                    if (dq1) begin
                        r_hold1    <= rc1;
                        got_rcp[1] <= 1'b1;
                    end
                    if (dq2) begin
                        r_hold2    <= rc2;
                        got_rcp[2] <= 1'b1;
                    end
                    if (all_rcp_done) begin
                        dvs   <= 1'b1;
                        phase <= 2'd2;
                    end
                end

                2'd2: begin
                    if (dqs) begin
                        weight_0  <= w0_c;
                        weight_1  <= w1_c;
                        weight_2  <= w2_c;
                        valid_out <= 1'b1;
                        busy      <= 1'b0;
                        phase     <= 2'd0;
                    end
                end

                default: phase <= 2'd0;
            endcase
        end
    end
endmodule


module ScalarVecMul #(
    parameter FEATURE_C = 8
)(
    input  wire [31:0]                    scalar,
    input  wire [FEATURE_C*32-1:0]        vec_in,
    output wire [FEATURE_C*32-1:0]        vec_out
);
    genvar ch;
    generate
        for (ch = 0; ch < FEATURE_C; ch = ch + 1) begin : gen_smul
            interp3nn_fp32_mul u_m (
                .a(scalar),
                .b(vec_in[ch*32 +: 32]),
                .result(vec_out[ch*32 +: 32])
            );
        end
    endgenerate
endmodule


module VecReduce3 #(
    parameter FEATURE_C = 8
)(
    input  wire [FEATURE_C*32-1:0]        v0,
    input  wire [FEATURE_C*32-1:0]        v1,
    input  wire [FEATURE_C*32-1:0]        v2,
    output wire [FEATURE_C*32-1:0]        vec_out
);
    genvar ch;
    generate
        for (ch = 0; ch < FEATURE_C; ch = ch + 1) begin : gen_addtree
            wire [31:0] s01;
            interp3nn_fp32_add u_a0 (
                .a(v0[ch*32 +: 32]),
                .b(v1[ch*32 +: 32]),
                .result(s01)
            );
            interp3nn_fp32_add u_a1 (
                .a(s01),
                .b(v2[ch*32 +: 32]),
                .result(vec_out[ch*32 +: 32])
            );
        end
    endgenerate
endmodule


module interp_3nn #(
    parameter FEATURE_C   = 8,
    parameter DIV_LAT     = 14
)(
    input  wire                                    clk,
    input  wire                                    rst_n,

    input  wire                                    start,
    output wire                                    busy,
    output reg                                     interp_valid,

    input  wire [31:0]                             dist_sq_0,
    input  wire [31:0]                             dist_sq_1,
    input  wire [31:0]                             dist_sq_2,

    input  wire [FEATURE_C*32-1:0]                 feat_0,
    input  wire [FEATURE_C*32-1:0]                 feat_1,
    input  wire [FEATURE_C*32-1:0]                 feat_2,

    output wire                                    feat_req,

    input  wire [31:0]                             eps_fp32,

    output reg  [FEATURE_C*32-1:0]                 feat_interp
);

    reg running;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            running <= 1'b0;
        else if (start && !running)
            running <= 1'b1;
        else if (interp_valid)
            running <= 1'b0;
    end
    assign busy = running;

    reg [31:0] d0_r, d1_r, d2_r;

    always @(posedge clk) begin
        if (start && !running) begin
            d0_r <= dist_sq_0;
            d1_r <= dist_sq_1;
            d2_r <= dist_sq_2;
        end
    end

    reg         recip_start;
    wire        recip_busy;
    wire        recip_valid;
    wire [31:0] w0, w1, w2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            recip_start <= 1'b0;
        else
            recip_start <= (start && !running);
    end

    ReciprocalUnit #(
        .DIV_LAT (DIV_LAT)
    ) u_rcp (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (recip_start),
        .dist_sq_0 (d0_r),
        .dist_sq_1 (d1_r),
        .dist_sq_2 (d2_r),
        .eps_fp32  (eps_fp32),
        .busy      (recip_busy),
        .valid_out (recip_valid),
        .weight_0  (w0),
        .weight_1  (w1),
        .weight_2  (w2)
    );

    reg [31:0] w0_r, w1_r, w2_r;
    reg        feat_req_r;
    reg        s4a_go;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w0_r       <= 32'd0;
            w1_r       <= 32'd0;
            w2_r       <= 32'd0;
            feat_req_r <= 1'b0;
            s4a_go     <= 1'b0;
        end else begin
            feat_req_r <= 1'b0;
            s4a_go     <= feat_req_r;
            if (recip_valid) begin
                w0_r       <= w0;
                w1_r       <= w1;
                w2_r       <= w2;
                feat_req_r <= 1'b1;
            end
        end
    end

    assign feat_req = feat_req_r;

    wire [FEATURE_C*32-1:0] wf0_comb, wf1_comb, wf2_comb;

    ScalarVecMul #(.FEATURE_C(FEATURE_C)) u_sv0 (
        .scalar  (w0_r),
        .vec_in  (feat_0),
        .vec_out (wf0_comb)
    );
    ScalarVecMul #(.FEATURE_C(FEATURE_C)) u_sv1 (
        .scalar  (w1_r),
        .vec_in  (feat_1),
        .vec_out (wf1_comb)
    );
    ScalarVecMul #(.FEATURE_C(FEATURE_C)) u_sv2 (
        .scalar  (w2_r),
        .vec_in  (feat_2),
        .vec_out (wf2_comb)
    );

    reg [FEATURE_C*32-1:0] wf0_r, wf1_r, wf2_r;
    reg                     s4a_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wf0_r     <= {(FEATURE_C*32){1'b0}};
            wf1_r     <= {(FEATURE_C*32){1'b0}};
            wf2_r     <= {(FEATURE_C*32){1'b0}};
            s4a_valid <= 1'b0;
        end else begin
            s4a_valid <= 1'b0;
            if (s4a_go) begin
                wf0_r     <= wf0_comb;
                wf1_r     <= wf1_comb;
                wf2_r     <= wf2_comb;
                s4a_valid <= 1'b1;
            end
        end
    end

    wire [FEATURE_C*32-1:0] sum_comb;

    VecReduce3 #(.FEATURE_C(FEATURE_C)) u_vr (
        .v0      (wf0_r),
        .v1      (wf1_r),
        .v2      (wf2_r),
        .vec_out (sum_comb)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_interp  <= {(FEATURE_C*32){1'b0}};
            interp_valid <= 1'b0;
        end else begin
            interp_valid <= 1'b0;
            if (s4a_valid) begin
                feat_interp  <= sum_comb;
                interp_valid <= 1'b1;
            end
        end
    end

endmodule