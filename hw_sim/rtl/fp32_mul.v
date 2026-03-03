
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
