// // Floating-point multiplier module
// `include "wallace_tree_24x24.v"
// module fp32_mul (
//     input  wire clk,           // Clock  (New addition for pipeling)
//     input  wire rstn,          // Reset
//     input  wire ena,           // Input data valid
//     input  wire [31:0] a,      // First operand (IEEE 754 single-precision)
//     input  wire [31:0] b,      // Second operand (IEEE 754 single-precision)
//     input  wire [1:0]  rm,     // Rounding mode (00: nearest, 01: zero, 10: +inf, 11: -inf)
//     output reg [31:0] s,       // Result (IEEE 754 single-precision)
//     output reg valid           // Output data valid
// );
//     // Detect special cases for inputs
//     wire        a_expo_is_00 = ~|a[30:23];     // Exponent of a is 0 (denormalized)
//     wire        b_expo_is_00 = ~|b[30:23];     // Exponent of b is 0 (denormalized)
//     wire        a_expo_is_ff = &a[30:23];      // Exponent of a is 0xff (Inf/NaN)
//     wire        b_expo_is_ff = &b[30:23];      // Exponent of b is 0xff (Inf/NaN)
//     wire        a_frac_is_00 = ~|a[22:0];      // Fraction of a is 0
//     wire        b_frac_is_00 = ~|b[22:0];      // Fraction of b is 0

//     // Identify special values
//     wire        a_is_inf = a_expo_is_ff & a_frac_is_00;     // a is infinity
//     wire        b_is_inf = b_expo_is_ff & b_frac_is_00;     // b is infinity
//     wire        a_is_nan = a_expo_is_ff & ~a_frac_is_00;    // a is NaN
//     wire        b_is_nan = b_expo_is_ff & ~b_frac_is_00;    // b is NaN
//     wire        a_is_0 = a_expo_is_00 & a_frac_is_00;       // a is zero
//     wire        b_is_0 = b_expo_is_00 & b_frac_is_00;       // b is zero
//     wire        s_is_inf = a_is_inf | b_is_inf;             // Result is infinity
//     wire        s_is_nan = a_is_nan | b_is_nan | (a_is_inf & b_is_0) | (b_is_inf & a_is_0); // Result is NaN

//     // NaN fraction: select larger fraction with MSB set to 1 // why?
//     wire [22:0] nan_frac = (a[21:0] > b[21:0]) ? {1'b1, a[21:0]} : {1'b1, b[21:0]};
//     wire [22:0] inf_nan_frac = s_is_nan ? nan_frac : 23'h0; // Fraction for Inf/NaN result

//     // Compute sign of result
//     wire        sign = a[31] ^ b[31]; // XOR of input signs

//     // Exponent calculation with bias adjustment
//     wire [9:0]  exp10 = {2'h0, a[30:23]} + {2'h0, b[30:23]} - 10'h7f +
//                         {9'h0, a_expo_is_00} + {9'h0, b_expo_is_00}; // Add 1 for each denormalized input

//     // Prepare fractions with hidden bit
//     wire [23:0] a_frac24 = {~a_expo_is_00, a[22:0]}; // Hidden bit: 1 for normalized, 0 for denormalized
//     wire [23:0] b_frac24 = {~b_expo_is_00, b[22:0]}; // Hidden bit: 1 for normalized, 0 for denormalized

//     // Wallace tree multiplication
//     wire [47:0] z;              // 48-bit product
//     wire [47:8] z_sum;          // Sum from Wallace tree
//     wire [47:8] z_carry;        // Carry from Wallace tree

//     // Multiplication 1: Generate carry and sum using Wallace tree
//     wallace_tree_24x24 wt24 (
//         .a(a_frac24),
//         .b(b_frac24),
//         .x(z_sum),
//         .y(z_carry),
//         .z(z[7:0])
//     );

//     // Multiplication 2: Compute product by adding sum and carry
//     assign z[47:8] = z_sum + z_carry; // xx.xxxxxxxx...

//     // Normalization
//     wire [46:0] z5, z4, z3, z2, z1, z0; // Intermediate normalization steps
//     wire        zero5, zero4, zero3, zero2, zero1, zero0; // Leading zero flags
//     wire [5:0]  zeros = {zero5, zero4, zero3, zero2, zero1, zero0}; // Leading zero count
//     assign zero5 = ~|z[46:15];   // Check for 32-bit zero
//     assign z5 = zero5 ? {z[14:0], 32'b0} : z[46:0];
//     assign zero4 = ~|z5[46:31];  // Check for 16-bit zero
//     assign z4 = zero4 ? {z5[30:0], 16'b0} : z5;
//     assign zero3 = ~|z4[46:39];  // Check for 8-bit zero
//     assign z3 = zero3 ? {z4[38:0], 8'b0} : z4;
//     assign zero2 = ~|z3[46:43];  // Check for 4-bit zero
//     assign z2 = zero2 ? {z3[42:0], 4'b0} : z3;
//     assign zero1 = ~|z2[46:45];  // Check for 2-bit zero
//     assign z1 = zero1 ? {z2[44:0], 2'b0} : z2;
//     assign zero0 = ~|z1[46];     // Check for 1-bit zero
//     assign z0 = zero0 ? {z1[45:0], 1'b0} : z1;

//     // Adjust exponent and fraction
//     reg [46:0] frac0; // Temporary fraction
//     reg [9:0]  exp0;  // Temporary exponent
//     always @(*) begin
//         if (z[47]) begin // Result is 1x.xxxxx... (overflow)
//             exp0 = exp10 + 10'h1; // Increment exponent
//             frac0 = z[47:1];      // Remove overflow bit
//         end else begin
//             if (!exp10[9] && (exp10[8:0] > {3'b0, zeros}) && z0[46]) begin // Normalized number
//                 exp0 = exp10 - {4'b0, zeros}; // Adjust exponent
//                 frac0 = z0;                   // Use normalized fraction
//             end else begin // Denormalized number or zero
//                 exp0 = 10'b0; // Set exponent to zero
//                 if (!exp10[9] && (exp10 != 0)) // Positive exponent
//                     frac0 = z[46:0] << (exp10 - 10'h1); // Shift for denormalized
//                 else
//                     frac0 = z[46:0] >> (10'h1 - exp10); // Shift for zero or negative exponent
//             end
//         end
//     end

//     // Rounding
//     wire [26:0] frac = {frac0[46:21], |frac0[20:0]}; // Fraction with guard, round, and sticky bits
//     wire        frac_plus_1 = // Rounding decision based on mode
//         (~rm[1] & ~rm[0] & frac0[2] & (frac0[1] | frac0[0])) |
//         (~rm[1] & ~rm[0] & frac0[2] & ~frac0[1] & ~frac0[0] & frac0[3]) |
//         (~rm[1] & rm[0] & (frac0[2] | frac0[1] | frac0[0]) & sign) |
//         (rm[1] & ~rm[0] & (frac0[2] | frac0[1] | frac0[0]) & ~sign);
//     wire [24:0] frac_round = {1'b0, frac[26:3]} + {24'h0, frac_plus_1}; // Round fraction
//     wire [9:0]  exp1 = frac_round[24] ? exp0 + 10'h1 : exp0; // Adjust exponent for rounding
//     wire        overflow = (exp0 >= 10'h0ff) | (exp1 >= 10'h0ff); // Check for overflow

//     // Final result assembly
//     wire [31:0] s_wire;
//     assign s_wire = final_result(overflow, rm, sign, s_is_inf, s_is_nan, exp1[7:0], frac_round[22:0], inf_nan_frac);

//     always @(posedge clk or negedge rstn) begin
//         if (!rstn) begin
//             s <= 32'h0000_0000;
//             valid <= 1'b0;
//         end else begin
//             if (ena) begin
//                 s <= s_wire;
//                 valid <= 1'b1;  // 这个输出有效处理不合理 需要在修改流水版本
//             end else begin
//                 valid <= 1'b0;
//             end
//         end
//     end

//     // Function to handle special cases and compute final result
//     function [31:0] final_result;
//         input        overflow;
//         input [1:0]  rm;
//         input        sign, s_is_inf, s_is_nan;
//         input [7:0]  exponent;
//         input [22:0] fraction, inf_nan_frac;
//         /* verilator lint_off CASEX */
//         /* verilator lint_off CASEOVERLAP */
//         casex ({overflow, rm, sign, s_is_nan, s_is_inf})
//             6'b1_00_x_0_x : final_result = {sign, 8'hff, 23'h000000}; // inf
//             6'b1_01_0_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // max
//             6'b1_01_1_0_x : final_result = {sign, 8'hff, 23'h000000}; // inf
//             6'b1_10_0_0_x : final_result = {sign, 8'hff, 23'h000000}; // inf
//             6'b1_10_1_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // max
//             6'b1_11_x_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // max
//             6'b0_xx_x_0_0 : final_result = {sign, exponent, fraction}; // nor
//             6'bx_xx_x_1_x : final_result = {1'b1, 8'hff, inf_nan_frac}; // nan
//             6'bx_xx_x_0_x : final_result = {sign, 8'hff, inf_nan_frac}; // inf
//             default       : final_result = {sign, 8'h00, 23'h000000}; // 0
//         endcase
//         /* verilator lint_on CASEX */
//         /* verilator lint_on CASEOVERLAP */
//     endfunction
// endmodule


// Floating-point multiplier module (3-stage pipeline)
`include "wallace_tree_24x24.v"
// 三级流水：
// stage-1： 输入--> wallace tree 部分积
// stage-2： wallace tree --> 部分积结果 输出48b尾数z
// stage-3： 对z规格化、舍入、特殊值处理 --> 输出结果
module fp32_mul (
    input  wire        clk,           // Clock
    input  wire        rstn,          // Active-low Reset
    input  wire        ena,           // Input data valid (accept new operands)
    input  wire [31:0] a,             // IEEE754 single-precision
    input  wire [31:0] b,             // IEEE754 single-precision
    input  wire [1:0]  rm,            // Rounding mode: 00 nearest-even, 01 toward 0, 10 +inf, 11 -inf
    output reg  [31:0] s,             // Result
    output reg         valid          // Output data valid (aligned with s)
);

    // =========================================================================
    // Stage 1: 输入寄存、特殊值解析、指数/符号/尾数准备
    // =========================================================================
    reg        valid_s1;
    reg [31:0] a_s1, b_s1;
    reg [1:0]  rm_s1;

    // 常用解析信号（S1）
    wire        a_expo_is_00_s1 = ~|a_s1[30:23];
    wire        b_expo_is_00_s1 = ~|b_s1[30:23];
    wire        a_expo_is_ff_s1 =  &a_s1[30:23];
    wire        b_expo_is_ff_s1 =  &b_s1[30:23];
    wire        a_frac_is_00_s1 = ~|a_s1[22:0];
    wire        b_frac_is_00_s1 = ~|b_s1[22:0];

    wire        a_is_inf_s1 = a_expo_is_ff_s1 &  a_frac_is_00_s1;
    wire        b_is_inf_s1 = b_expo_is_ff_s1 &  b_frac_is_00_s1;
    wire        a_is_nan_s1 = a_expo_is_ff_s1 & ~a_frac_is_00_s1;
    wire        b_is_nan_s1 = b_expo_is_ff_s1 & ~b_frac_is_00_s1;
    wire        a_is_0_s1   = a_expo_is_00_s1 & a_frac_is_00_s1;
    wire        b_is_0_s1   = b_expo_is_00_s1 & b_frac_is_00_s1;

    // 结果特殊情况
    wire        s_is_inf_s1 = a_is_inf_s1 | b_is_inf_s1;
    wire        s_is_nan_s1 = a_is_nan_s1 | b_is_nan_s1 | (a_is_inf_s1 & b_is_0_s1) | (b_is_inf_s1 & a_is_0_s1);

    // NaN fraction：取较大的payload并强制最高bit为1（quiet NaN）
    wire [22:0] nan_frac_s1 = (a_s1[21:0] > b_s1[21:0]) ? {1'b1, a_s1[21:0]} : {1'b1, b_s1[21:0]};
    wire [22:0] inf_nan_frac_s1 = s_is_nan_s1 ? nan_frac_s1 : 23'h0;

    // 符号
    wire        sign_s1 = a_s1[31] ^ b_s1[31];

    // 指数（带去偏与非规数修正：非规数隐含位为0，相当于指数再+1来弥补）
    wire [9:0]  exp10_s1 = {2'h0, a_s1[30:23]} + {2'h0, b_s1[30:23]} - 10'h07f
                          + {9'h0, a_expo_is_00_s1} + {9'h0, b_expo_is_00_s1};

    // 尾数（带隐藏位）
    wire [23:0] a_frac24_s1 = {~a_expo_is_00_s1, a_s1[22:0]};
    wire [23:0] b_frac24_s1 = {~b_expo_is_00_s1, b_s1[22:0]};

    // S1 → S2 寄存
    reg        sign_s2;
    reg [9:0]  exp10_s2;
    reg [1:0]  rm_s2;
    reg        s_is_inf_s2, s_is_nan_s2;
    reg [22:0] inf_nan_frac_s2;
    reg [23:0] a_frac24_s2, b_frac24_s2;
    reg        valid_s2;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            valid_s1 <= 1'b0;
            valid_s2 <= 1'b0;
        end else begin
            // S1：锁存输入
            if (ena) begin
                a_s1  <= a;
                b_s1  <= b;
                rm_s1 <= rm;
            end
            valid_s1 <= ena;

            // S1 → S2
            if (valid_s1) begin
                sign_s2         <= sign_s1;
                exp10_s2        <= exp10_s1;
                rm_s2           <= rm_s1;
                s_is_inf_s2     <= s_is_inf_s1;
                s_is_nan_s2     <= s_is_nan_s1;
                inf_nan_frac_s2 <= inf_nan_frac_s1;
                a_frac24_s2     <= a_frac24_s1;
                b_frac24_s2     <= b_frac24_s1;
            end
            valid_s2 <= valid_s1;
        end
    end

    // =========================================================================
    // Stage 2: Wallace tree 乘法（得到 sum/carry/低8位）
    // =========================================================================
    wire [47:8] z_sum_s2;
    wire [47:8] z_carry_s2;
    wire [7:0]  z_low8_s2;

    // 壁虎树（部分积压缩）
    wallace_tree_24x24 wt24 (
        .a(a_frac24_s2),
        .b(b_frac24_s2),
        .x(z_sum_s2),
        .y(z_carry_s2),
        .z(z_low8_s2)
    );

    // S2 → S3 寄存
    reg        sign_s3;
    reg [9:0]  exp10_s3;
    reg [1:0]  rm_s3;
    reg        s_is_inf_s3, s_is_nan_s3;
    reg [22:0] inf_nan_frac_s3;
    reg [47:8] z_sum_s3, z_carry_s3;
    reg [7:0]  z_low8_s3;
    reg        valid_s3;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            valid_s3 <= 1'b0;
        end else begin
            if (valid_s2) begin
                sign_s3         <= sign_s2;
                exp10_s3        <= exp10_s2;
                rm_s3           <= rm_s2;
                s_is_inf_s3     <= s_is_inf_s2;
                s_is_nan_s3     <= s_is_nan_s2;
                inf_nan_frac_s3 <= inf_nan_frac_s2;
                z_sum_s3        <= z_sum_s2;
                z_carry_s3      <= z_carry_s2;
                z_low8_s3       <= z_low8_s2;
            end
            valid_s3 <= valid_s2;
        end
    end

    // =========================================================================
    // Stage 3: 合并乘积 → 规格化 → 舍入 → 特殊值/溢出处理 → 输出
    // =========================================================================
    // 合并 Wallace tree sum/carry
    wire [47:0] z_s3;
    assign z_s3[7:0]  = z_low8_s3;
    assign z_s3[47:8] = z_sum_s3 + z_carry_s3; // 48-bit product

    // —— 规格化（与你原始算法一致）——
    wire [46:0] z5, z4, z3n, z2n, z1n, z0n;  // 注意：命名避免与 z_s3 混淆
    wire        zero5, zero4, zero3, zero2, zero1, zero0;
    wire [5:0]  zeros = {zero5, zero4, zero3, zero2, zero1, zero0};

    assign zero5 = ~|z_s3[46:15];
    assign z5    = zero5 ? {z_s3[14:0], 32'b0} : z_s3[46:0];

    assign zero4 = ~|z5[46:31];
    assign z4    = zero4 ? {z5[30:0], 16'b0} : z5;

    assign zero3 = ~|z4[46:39];
    assign z3n   = zero3 ? {z4[38:0], 8'b0}  : z4;

    assign zero2 = ~|z3n[46:43];
    assign z2n   = zero2 ? {z3n[42:0], 4'b0} : z3n;

    assign zero1 = ~|z2n[46:45];
    assign z1n   = zero1 ? {z2n[44:0], 2'b0} : z2n;

    assign zero0 = ~|z1n[46];
    assign z0n   = zero0 ? {z1n[45:0], 1'b0} : z1n;

    // 指数/尾数调整（与你原始逻辑一致）
    reg [46:0] frac0_s3;
    reg [9:0]  exp0_s3;

    always @(*) begin
        if (z_s3[47]) begin
            // 1x.xxxxx... 溢出到左边
            exp0_s3  = exp10_s3 + 10'h1;
            frac0_s3 = z_s3[47:1];
        end else begin
            if (!exp10_s3[9] && (exp10_s3[8:0] > {3'b0, zeros}) && z0n[46]) begin
                // 规格数
                exp0_s3  = exp10_s3 - {4'b0, zeros};
                frac0_s3 = z0n;
            end else begin
                // 非规格数或0
                exp0_s3 = 10'b0;
                if (!exp10_s3[9] && (exp10_s3 != 10'b0))
                    // 正指数但不够成规格数 → 向右对齐为非规
                    frac0_s3 = z_s3[46:0] << (exp10_s3 - 10'h1);
                else
                    // 零或负指数
                    frac0_s3 = z_s3[46:0] >> (10'h1 - exp10_s3);
            end
        end
    end

    // —— 舍入（与你原始逻辑一致）——
    wire [26:0] frac_s3 = {frac0_s3[46:21], |frac0_s3[20:0]}; // 带 sticky
    wire        frac_plus_1_s3 =
        (~rm_s3[1] & ~rm_s3[0] & frac0_s3[2] & (frac0_s3[1] | frac0_s3[0])) |
        (~rm_s3[1] & ~rm_s3[0] & frac0_s3[2] & ~frac0_s3[1] & ~frac0_s3[0] & frac0_s3[3]) |
        (~rm_s3[1] &  rm_s3[0] & (frac0_s3[2] | frac0_s3[1] | frac0_s3[0]) &  sign_s3) |
        ( rm_s3[1] & ~rm_s3[0] & (frac0_s3[2] | frac0_s3[1] | frac0_s3[0]) & ~sign_s3);

    wire [24:0] frac_round_s3 = {1'b0, frac_s3[26:3]} + {24'h0, frac_plus_1_s3};
    wire [9:0]  exp1_s3       = frac_round_s3[24] ? (exp0_s3 + 10'h1) : exp0_s3;
    wire        overflow_s3   = (exp0_s3 >= 10'h0ff) | (exp1_s3 >= 10'h0ff);

    // 最终装配
    wire [31:0] s_wire_s3;
    assign s_wire_s3 = final_result(
        overflow_s3, rm_s3, sign_s3, s_is_inf_s3, s_is_nan_s3,
        exp1_s3[7:0], frac_round_s3[22:0], inf_nan_frac_s3
    );

    // 输出寄存（S3 → 输出）
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            s     <= 32'h0000_0000;
            valid <= 1'b0;
        end else begin
            if (valid_s3) begin
                s <= s_wire_s3;
            end
            valid <= valid_s3;  // 与 s 对齐
        end
    end

    // =========================================================================
    // 特殊情况/溢出处理函数（保持你的写法）
    // =========================================================================
    function [31:0] final_result;
        input        overflow;
        input [1:0]  rm;
        input        sign, s_is_inf, s_is_nan;
        input [7:0]  exponent;
        input [22:0] fraction, inf_nan_frac;
        /* verilator lint_off CASEX */
        /* verilator lint_off CASEOVERLAP */
        casex ({overflow, rm, sign, s_is_nan, s_is_inf})
            6'b1_00_x_0_x : final_result = {sign, 8'hff, 23'h000000}; // inf (round to nearest)
            6'b1_01_0_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // max  (toward 0, +)
            6'b1_01_1_0_x : final_result = {sign, 8'hff, 23'h000000}; // inf  (toward 0, -)
            6'b1_10_0_0_x : final_result = {sign, 8'hff, 23'h000000}; // inf  (+inf)
            6'b1_10_1_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // max  (+inf, negative)
            6'b1_11_x_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // max  (-inf)
            6'b0_xx_x_0_0 : final_result = {sign, exponent, fraction}; // normal/subnormal
            6'bx_xx_x_1_x : final_result = {1'b1, 8'hff, inf_nan_frac}; // NaN
            6'bx_xx_x_0_x : final_result = {sign, 8'hff, inf_nan_frac}; // Inf
            default       : final_result = {sign, 8'h00, 23'h000000};   // Zero
        endcase
        /* verilator lint_on CASEX */
        /* verilator lint_on CASEOVERLAP */
    endfunction

endmodule
