module fp32_adder (
    input  wire clk,           // Clock
    input  wire rstn,          // Reset (active low)
    input  wire ena,           // Input data valid
    input  wire [31:0] a,      // First operand (IEEE 754 single-precision)
    input  wire [31:0] b,      // Second operand (IEEE 754 single-precision)
    input  wire [1:0]  rm,     // Rounding mode (00: nearest, 01: zero, 10: +inf, 11: -inf)
    input  wire        sel,    // Operation: 1 = subtract, 0 = add
    output reg [31:0] s,       // Result (IEEE 754 single-precision)
    output reg valid
);

    // ---------------------------
    // Stage 1 - inputs registers (kept) and compute selection / small helpers
    // ---------------------------
    reg [31:0] a_s1, b_s1;
    reg [1:0]  rm_s1;
    reg        sel_s1;
    reg        valid_s1;

    // ----- original reg outputs from S1 that downstream stages use -----
    reg        exchange_s1;
    reg [31:0] fp_large_s1;
    reg [31:0] fp_small_s1;
    reg        fp_large_hidden_bit_s1;
    reg        fp_small_hidden_bit_s1;
    reg [23:0] large_frac24_s1;
    reg [23:0] small_frac24_s1;
    reg [7:0]  temp_exp_s1;
    reg        sign_s1;
    reg        op_sub_s1;
    reg        fp_large_expo_is_ff_s1;
    reg        fp_small_expo_is_ff_s1;
    reg        fp_large_frac_is_00_s1;
    reg        fp_small_frac_is_00_s1;
    reg        fp_large_is_inf_s1;
    reg        fp_small_is_inf_s1;
    reg        s_is_inf_s1;
    reg        s_is_nan_s1;
    reg [22:0] nan_frac_s1;
    reg [22:0] inf_nan_frac_s1;

    // Align helpers (registered outputs from S1)
    reg [7:0]  exp_diff_s1;
    reg        small_den_only_s1;
    reg [7:0]  shift_amount_s1;
    reg [49:0] small_frac50_s1;
    reg [26:0] small_frac27_s1;

    // ---------------------------
    // Stage1: combinational computations -> wires (avoid driving regs in combinational block)
    // ---------------------------
    wire        exchange_s1_c;
    wire [31:0] fp_large_s1_c;
    wire [31:0] fp_small_s1_c;
    wire        fp_large_hidden_bit_s1_c;
    wire        fp_small_hidden_bit_s1_c;
    wire [23:0] large_frac24_s1_c;
    wire [23:0] small_frac24_s1_c;
    wire [7:0]  temp_exp_s1_c;
    wire        sign_s1_c;
    wire        op_sub_s1_c;
    wire        fp_large_expo_is_ff_s1_c;
    wire        fp_small_expo_is_ff_s1_c;
    wire        fp_large_frac_is_00_s1_c;
    wire        fp_small_frac_is_00_s1_c;
    wire        fp_large_is_inf_s1_c;
    wire        fp_small_is_inf_s1_c;
    wire        s_is_inf_s1_c;
    wire        s_is_nan_s1_c;
    wire [22:0] nan_frac_s1_c;
    wire [22:0] inf_nan_frac_s1_c;
    wire [7:0]  exp_diff_s1_c;
    wire        small_den_only_s1_c;
    wire [7:0]  shift_amount_s1_c;
    wire [49:0] small_frac50_s1_c;
    wire [26:0] small_frac27_s1_c;

    // compute combinationally using assigns (preserve original expressions)
    assign exchange_s1_c = ({1'b0, b[30:0]} > {1'b0, a[30:0]});
    assign fp_large_s1_c = exchange_s1_c ? b : a;
    assign fp_small_s1_c = exchange_s1_c ? a : b;

    assign fp_large_hidden_bit_s1_c = |fp_large_s1_c[30:23];
    assign fp_small_hidden_bit_s1_c = |fp_small_s1_c[30:23];
    assign large_frac24_s1_c = {fp_large_hidden_bit_s1_c, fp_large_s1_c[22:0]};
    assign small_frac24_s1_c = {fp_small_hidden_bit_s1_c, fp_small_s1_c[22:0]};

    assign temp_exp_s1_c = fp_large_s1_c[30:23];
    assign sign_s1_c = exchange_s1_c ? (sel ^ b[31]) : a[31];
    assign op_sub_s1_c = sel ^ fp_large_s1_c[31] ^ fp_small_s1_c[31];

    assign fp_large_expo_is_ff_s1_c = &fp_large_s1_c[30:23];
    assign fp_small_expo_is_ff_s1_c = &fp_small_s1_c[30:23];
    assign fp_large_frac_is_00_s1_c = ~|fp_large_s1_c[22:0];
    assign fp_small_frac_is_00_s1_c = ~|fp_small_s1_c[22:0];
    assign fp_large_is_inf_s1_c = fp_large_expo_is_ff_s1_c & fp_large_frac_is_00_s1_c;
    assign fp_small_is_inf_s1_c = fp_small_expo_is_ff_s1_c & fp_small_frac_is_00_s1_c;
    assign s_is_inf_s1_c = fp_large_is_inf_s1_c | fp_small_is_inf_s1_c;
    assign s_is_nan_s1_c = (fp_large_expo_is_ff_s1_c & ~fp_large_frac_is_00_s1_c)
                          | (fp_small_expo_is_ff_s1_c & ~fp_small_frac_is_00_s1_c)
                          | (((sel ^ fp_small_s1_c[31] ^ fp_large_s1_c[31]) & fp_large_is_inf_s1_c & fp_small_is_inf_s1_c));

    assign nan_frac_s1_c = ({1'b0, a[22:0]} > {1'b0, b[22:0]}) ? {1'b1, a[21:0]} : {1'b1, b[21:0]};
    assign inf_nan_frac_s1_c = s_is_nan_s1_c ? nan_frac_s1_c : 23'h0;

    assign exp_diff_s1_c = fp_large_s1_c[30:23] - fp_small_s1_c[30:23];
    assign small_den_only_s1_c = (fp_large_s1_c[30:23] != 0) & (fp_small_s1_c[30:23] == 0);
    assign shift_amount_s1_c = small_den_only_s1_c ? (exp_diff_s1_c - 8'h1) : exp_diff_s1_c;
    assign small_frac50_s1_c = (shift_amount_s1_c >= 8'd26) ? {26'h0, small_frac24_s1_c} : ({small_frac24_s1_c, 26'h0} >> shift_amount_s1_c);
    assign small_frac27_s1_c = {small_frac50_s1_c[49:24], |small_frac50_s1_c[23:0]};

    // ---------------------------
    // register S1 outputs into S1 flops (to freeze values for next stage)
    // ---------------------------
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            a_s1 <= 32'b0;
            b_s1 <= 32'b0;
            rm_s1 <= 2'b00;
            sel_s1 <= 1'b0;
            valid_s1 <= 1'b0;

            // zero other regs to safe default
            exchange_s1 <= 1'b0;
            fp_large_s1 <= 32'b0;
            fp_small_s1 <= 32'b0;
            fp_large_hidden_bit_s1 <= 1'b0;
            fp_small_hidden_bit_s1 <= 1'b0;
            large_frac24_s1 <= 24'b0;
            small_frac24_s1 <= 24'b0;
            temp_exp_s1 <= 8'b0;
            sign_s1 <= 1'b0;
            op_sub_s1 <= 1'b0;
            fp_large_expo_is_ff_s1 <= 1'b0;
            fp_small_expo_is_ff_s1 <= 1'b0;
            fp_large_frac_is_00_s1 <= 1'b0;
            fp_small_frac_is_00_s1 <= 1'b0;
            fp_large_is_inf_s1 <= 1'b0;
            fp_small_is_inf_s1 <= 1'b0;
            s_is_inf_s1 <= 1'b0;
            s_is_nan_s1 <= 1'b0;
            nan_frac_s1 <= 23'b0;
            inf_nan_frac_s1 <= 23'b0;
            exp_diff_s1 <= 8'b0;
            small_den_only_s1 <= 1'b0;
            shift_amount_s1 <= 8'b0;
            small_frac50_s1 <= 50'b0;
            small_frac27_s1 <= 27'b0;
        end else begin
            if (ena) begin
                a_s1 <= a;
                b_s1 <= b;
                rm_s1 <= rm;
                sel_s1 <= sel;
            end
            valid_s1 <= ena;

            // snapshot the combinational results computed above into regs
            exchange_s1 <= exchange_s1_c;
            fp_large_s1 <= fp_large_s1_c;
            fp_small_s1 <= fp_small_s1_c;
            fp_large_hidden_bit_s1 <= fp_large_hidden_bit_s1_c;
            fp_small_hidden_bit_s1 <= fp_small_hidden_bit_s1_c;
            large_frac24_s1 <= large_frac24_s1_c;
            small_frac24_s1 <= small_frac24_s1_c;
            temp_exp_s1 <= temp_exp_s1_c;
            sign_s1 <= sign_s1_c;
            op_sub_s1 <= op_sub_s1_c;
            fp_large_expo_is_ff_s1 <= fp_large_expo_is_ff_s1_c;
            fp_small_expo_is_ff_s1 <= fp_small_expo_is_ff_s1_c;
            fp_large_frac_is_00_s1 <= fp_large_frac_is_00_s1_c;
            fp_small_frac_is_00_s1 <= fp_small_frac_is_00_s1_c;
            fp_large_is_inf_s1 <= fp_large_is_inf_s1_c;
            fp_small_is_inf_s1 <= fp_small_is_inf_s1_c;
            s_is_inf_s1 <= s_is_inf_s1_c;
            s_is_nan_s1 <= s_is_nan_s1_c;
            nan_frac_s1 <= nan_frac_s1_c;
            inf_nan_frac_s1 <= inf_nan_frac_s1_c;
            exp_diff_s1 <= exp_diff_s1_c;
            small_den_only_s1 <= small_den_only_s1_c;
            shift_amount_s1 <= shift_amount_s1_c;
            small_frac50_s1 <= small_frac50_s1_c;
            small_frac27_s1 <= small_frac27_s1_c;
        end
    end

    // ---------------------------
    // Stage 2 - alignment and add/sub (register S1 -> S2 values first)
    // ---------------------------
    // S2 registers (inputs to stage 2)
    reg [7:0]  temp_exp_s2;
    reg        sign_s2;
    reg        op_sub_s2;
    reg [1:0]  rm_s2;
    reg        s_is_inf_s2;
    reg        s_is_nan_s2;
    reg [22:0] inf_nan_frac_s2;
    reg [23:0] large_frac24_s2;
    reg [26:0] small_frac27_s2;
    reg        valid_s2;

    // aligned fractions and cal_frac computed in S2 (then forwarded to S3)
    reg [27:0] aligned_large_frac_s2;
    reg [27:0] aligned_small_frac_s2;
    reg [27:0] cal_frac_s2; // result of add/sub

    // register S1 -> S2
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            temp_exp_s2 <= 8'b0;
            sign_s2 <= 1'b0;
            op_sub_s2 <= 1'b0;
            rm_s2 <= 2'b00;
            s_is_inf_s2 <= 1'b0;
            s_is_nan_s2 <= 1'b0;
            inf_nan_frac_s2 <= 23'b0;
            large_frac24_s2 <= 24'b0;
            small_frac27_s2 <= 27'b0;
            valid_s2 <= 1'b0;

            aligned_large_frac_s2 <= 28'b0;
            aligned_small_frac_s2 <= 28'b0;
            cal_frac_s2 <= 28'b0;
        end else begin
            // forward control and metadata
            temp_exp_s2 <= temp_exp_s1;
            sign_s2 <= sign_s1;
            op_sub_s2 <= op_sub_s1;
            rm_s2 <= rm_s1;
            s_is_inf_s2 <= s_is_inf_s1;
            s_is_nan_s2 <= s_is_nan_s1;
            inf_nan_frac_s2 <= inf_nan_frac_s1;
            large_frac24_s2 <= large_frac24_s1;
            small_frac27_s2 <= small_frac27_s1;
            valid_s2 <= valid_s1;

            // alignment (do per original logic)
            aligned_large_frac_s2 <= {1'b0, large_frac24_s1, 3'b000};
            aligned_small_frac_s2 <= {1'b0, small_frac27_s1};

            // perform add/sub exactly as original
            if (op_sub_s1)
                cal_frac_s2 <= ({1'b0, large_frac24_s1, 3'b000} - {1'b0, small_frac27_s1});
            else
                cal_frac_s2 <= ({1'b0, large_frac24_s1, 3'b000} + {1'b0, small_frac27_s1});
        end
    end

    // ---------------------------
    // Stage 3 - normalization + rounding + final assembly
    // ---------------------------
    // S3 inputs (registered from S2)
    reg [27:0] cal_frac_s3;
    reg [7:0]  temp_exp_s3;
    reg        sign_s3;
    reg [1:0]  rm_s3;
    reg        s_is_inf_s3;
    reg        s_is_nan_s3;
    reg [22:0] inf_nan_frac_s3;
    reg        valid_s3;

    // snapshot S2 -> S3
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            cal_frac_s3 <= 28'b0;
            temp_exp_s3 <= 8'b0;
            sign_s3 <= 1'b0;
            rm_s3 <= 2'b00;
            s_is_inf_s3 <= 1'b0;
            s_is_nan_s3 <= 1'b0;
            inf_nan_frac_s3 <= 23'b0;
            valid_s3 <= 1'b0;
        end else begin
            cal_frac_s3 <= cal_frac_s2;
            temp_exp_s3 <= temp_exp_s2;
            sign_s3 <= sign_s2;
            rm_s3 <= rm_s2;
            s_is_inf_s3 <= s_is_inf_s2;
            s_is_nan_s3 <= s_is_nan_s2;
            inf_nan_frac_s3 <= inf_nan_frac_s2;
            valid_s3 <= valid_s2;
        end
    end

    // Normalization tree (same structure & semantics as original)
    wire [26:0] f4_s3, f3_s3, f2_s3, f1_s3, f0_s3;
    wire [4:0]  zeros_s3;
    assign zeros_s3[4] = ~|cal_frac_s3[26:11];
    assign f4_s3 = zeros_s3[4] ? {cal_frac_s3[10:0], 16'b0} : cal_frac_s3[26:0];
    assign zeros_s3[3] = ~|f4_s3[26:19];
    assign f3_s3 = zeros_s3[3] ? {f4_s3[18:0], 8'b0} : f4_s3;
    assign zeros_s3[2] = ~|f3_s3[26:23];
    assign f2_s3 = zeros_s3[2] ? {f3_s3[22:0], 4'b0} : f3_s3;
    assign zeros_s3[1] = ~|f2_s3[26:25];
    assign f1_s3 = zeros_s3[1] ? {f2_s3[24:0], 2'b0} : f2_s3;
    assign zeros_s3[0] = ~f1_s3[26];
    assign f0_s3 = zeros_s3[0] ? {f1_s3[25:0], 1'b0} : f1_s3;

    // regs for normalized results computed in combinational block
    reg [7:0]  exp0_s3;
    reg [26:0] frac0_s3;

    always @(*) begin
        // follow original always @(*) semantics for normalization
        if (cal_frac_s3[27]) begin // overflow: 1x.x
            frac0_s3 = cal_frac_s3[27:1];
            exp0_s3 = temp_exp_s3 + 8'h1;
        end else begin
            if ((temp_exp_s3 > zeros_s3) && (f0_s3[26])) begin
                exp0_s3 = temp_exp_s3 - zeros_s3;
                frac0_s3 = f0_s3;
            end else begin
                exp0_s3 = 8'b0;
                if (temp_exp_s3 != 0)
                    frac0_s3 = cal_frac_s3[26:0] << (temp_exp_s3 - 8'h1);
                else
                    frac0_s3 = cal_frac_s3[26:0];
            end
        end
    end

    // Rounding (exactly original)
    wire frac_plus_3_s3 = (~rm_s3[1] & ~rm_s3[0] & frac0_s3[2] & (frac0_s3[1] | frac0_s3[0])) |
                         (~rm_s3[1] & ~rm_s3[0] & frac0_s3[2] & ~frac0_s3[1] & ~frac0_s3[0] & frac0_s3[3]) |
                         (~rm_s3[1] &  rm_s3[0] & (frac0_s3[2] | frac0_s3[1] | frac0_s3[0]) & sign_s3) |
                         ( rm_s3[1] & ~rm_s3[0] & (frac0_s3[2] | frac0_s3[1] | frac0_s3[0]) & ~sign_s3);
    wire [24:0] frac_round_s3 = {1'b0, frac0_s3[26:3]} + {24'h0, frac_plus_3_s3};
    wire [7:0]  exponent_s3 = frac_round_s3[24] ? exp0_s3 + 8'h1 : exp0_s3;
    wire        overflow_s3 = &exp0_s3 | &exponent_s3;

    // final assembly (same final_result function semantics)
    wire [31:0] s_wire_s3;
    assign s_wire_s3 = final_result(overflow_s3, rm_s3, sign_s3, s_is_nan_s3, s_is_inf_s3, exponent_s3, frac_round_s3[22:0], inf_nan_frac_s3);

    // Output register: align with valid_s3
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            s <= 32'h0000_0000;
            valid <= 1'b0;
        end else begin
            if (valid_s3) begin
                s <= s_wire_s3;
            end
            valid <= valid_s3;
        end
    end

    // final_result function preserved from original (no change)
    function [31:0] final_result;
        input        overflow;
        input [1:0]  rm;
        input        sign;
        input        is_nan;
        input        is_inf;
        input [7:0]  exponent;
        input [22:0] fraction;
        input [22:0] inf_nan_frac;
        casex ({overflow, rm, sign, is_nan, is_inf})
            6'b1_00_x_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
            6'b1_01_0_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max (round to zero)
            6'b1_01_1_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
            6'b1_10_0_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
            6'b1_10_1_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max
            6'b1_11_x_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max
            6'b0_xx_x_0_0 : final_result = {sign, exponent, fraction}; // Normal result
            6'bx_xx_x_1_x : final_result = {1'b1, 8'hff, inf_nan_frac}; // NaN
            6'bx_xx_x_0_1 : final_result = {sign, 8'hff, inf_nan_frac}; // Infinity
            default       : final_result = {sign, 8'h00, 23'h000000}; // Zero
        endcase
    endfunction

endmodule
