
`timescale 1ns/1ps

// fdiv_newton (pipelined, fixed)
// - 修复了之前版本中的未声明局部变量／在过程块中声明 wire/reg 的问题
// - 将 "calc" 的组合计算改为基于当前 newton 输出 q、exp10、rm、sign 的组合逻辑（calc_next），
//   并在时钟沿将其寄存为 e1_calc --> e2_calc --> e3_calc
// - 在时序域中使用 e2_* 作为输入来计算 E3 的 final_result，从而避免在同一时钟边沿读取刚更新的寄存器值
// - 顶层输出 s 使用纯寄存的 e3_sign/e3_exponent/e3_fraction，减轻组合路径
// 注意：本模块依赖于外部模块 shift_to_msb_equ_1 与 newton24，请确保它们已在工程中。



`include "wallace_tree_24x26.v"
`include "wallace_tree_26x26.v"

module fdiv_newton (
    input  wire [31:0] a,         // Dividend (IEEE 754 single-precision)
    input  wire [31:0] b,         // Divisor (IEEE 754 single-precision)
    input  wire [1:0]  rm,        // Rounding mode (00: nearest, 01: zero, 10: +inf, 11: -inf)
    input  wire        fdiv,      // Division instruction signal (ID stage)
    input  wire        ena,       // Enable signal for pipeline
    input  wire        clk,       // Clock signal
    input  wire        clrn,      // Active-low reset signal
    output wire [31:0] s,         // Quotient (IEEE 754 single-precision)
    output wire [25:0] reg_x,     // Current approximation (for simulation)
    output wire [4:0]  count,     // Iteration counter for Newton-Raphson
    output wire        busy,      // Busy signal for pipeline control
    output wire        stall,     // Stall signal for pipeline
    output reg         valid      // Output valid signal (E3 stage)
);

    // 32-bit constants (keep full 32-bit and slice [30:0] when needed)
    localparam [31:0] ZERO32 = 32'h00000000;
    localparam [31:0] INF32  = 32'h7f800000;
    localparam [31:0] NaN32  = 32'h7fc00000;
    localparam [31:0] MAX32  = 32'h7f7fffff;

    // Input analysis for special cases (combinational)
    wire        a_expo_is_00 = ~|a[30:23];
    wire        b_expo_is_00 = ~|b[30:23];
    wire        a_expo_is_ff = &a[30:23];
    wire        b_expo_is_ff = &b[30:23];
    wire        a_frac_is_00 = ~|a[22:0];
    wire        b_frac_is_00 = ~|b[22:0];

    // Sign calculation
    wire        sign = a[31] ^ b[31];

    // Exponent calculation with bias (127)
    wire [9:0]  exp_10 = {2'h0, a[30:23]} - {2'h0, b[30:23]} + 10'h7f;

    // Normalize fractions to 1.xxxx format
    wire [23:0] a_temp24 = a_expo_is_00 ? {a[22:0], 1'b0} : {1'b1, a[22:0]};
    wire [23:0] b_temp24 = b_expo_is_00 ? {b[22:0], 1'b0} : {1'b1, b[22:0]};
    wire [23:0] a_frac24, b_frac24;        // Normalized fractions
    wire [4:0]  shamt_a, shamt_b;          // Shift amounts for normalization

    // Normalize fractions to have MSB = 1
    shift_to_msb_equ_1 shift_a (.a(a_temp24), .b(a_frac24), .sa(shamt_a));
    shift_to_msb_equ_1 shift_b (.a(b_temp24), .b(b_frac24), .sa(shamt_b));

    // Adjust exponent based on normalization shifts
    wire [9:0]  exp10 = exp_10 - shamt_a + shamt_b;

    // Pipeline registers for three stages (e1, e2, e3)
    reg         e1_sign, e2_sign, e3_sign;              // Sign bit
    reg [1:0]   e1_rm, e2_rm, e3_rm;                    // Rounding mode
    reg [9:0]   e1_exp10, e2_exp10, e3_exp10;           // Exponent
    reg         e1_ae00, e2_ae00, e3_ae00;             // a exponent = 00
    reg         e1_aeff, e2_aeff, e3_aeff;             // a exponent = ff
    reg         e1_af00, e2_af00, e3_af00;             // a fraction = 00
    reg         e1_be00, e2_be00, e3_be00;             // b exponent = 00
    reg         e1_beff, e2_beff, e3_beff;             // b exponent = ff
    reg         e1_bf00, e2_bf00, e3_bf00;             // b fraction = 00

    // Sample q from newton unit into our pipeline stages
    reg [31:0]  e1_q, e2_q, e3_q;

    // Small internal regs for staging the 'calc' (exp+fraction) before special-case handling
    reg [30:0]  e1_calc, e2_calc, e3_calc; // {exp[7:0], frac[22:0]}

    // Final registered exponent/fraction (reduce combinational load at output)
    reg [7:0]   e3_exponent;
    reg [22:0]  e3_fraction;

    // combinational 'next' value to be captured into e1_calc
    reg [30:0]  calc_next;

    // temporaries for combinational calc_next (declared at module scope for Verilog)
    reg         z0_msb;
    reg [31:0]  z0_local;
    reg [9:0]   exp_adj_local;
    reg [31:0]  frac0_local;
    reg [9:0]   exp0_local;
    reg [26:0]  frac_for_round;
    reg         frac_plus_1_local;
    reg [24:0]  frac_round_local;
    reg [9:0]   exp1_local;
    reg         overflow_local;
    reg [1:0]   rm_local;
    reg         sign_local;

    // Newton-Raphson division for fractions (kept as black box)
    wire [31:0] q; // raw fractional result from newton unit
    newton24 frac_newton (
        .a(a_frac24), .b(b_frac24), .fdiv(fdiv), .ena(ena),
        .clk(clk), .clrn(clrn), .q(q), .busy(busy),
        .count(count), .reg_x(reg_x), .stall(stall)
    );

    // --- Pipeline register update logic ---
    always @(negedge clrn or posedge clk) begin
        if (!clrn) begin
            // Reset pipeline registers
            e1_sign <= 1'b0; e2_sign <= 1'b0; e3_sign <= 1'b0;
            e1_rm <= 2'b0; e2_rm <= 2'b0; e3_rm <= 2'b0;
            e1_exp10 <= 10'b0; e2_exp10 <= 10'b0; e3_exp10 <= 10'b0;
            e1_ae00 <= 1'b0; e2_ae00 <= 1'b0; e3_ae00 <= 1'b0;
            e1_aeff <= 1'b0; e2_aeff <= 1'b0; e3_aeff <= 1'b0;
            e1_af00 <= 1'b0; e2_af00 <= 1'b0; e3_af00 <= 1'b0;
            e1_be00 <= 1'b0; e2_be00 <= 1'b0; e3_be00 <= 1'b0;
            e1_beff <= 1'b0; e2_beff <= 1'b0; e3_beff <= 1'b0;
            e1_bf00 <= 1'b0; e2_bf00 <= 1'b0; e3_bf00 <= 1'b0;
            e1_q <= 32'b0; e2_q <= 32'b0; e3_q <= 32'b0;
            e1_calc <= 31'b0; e2_calc <= 31'b0; e3_calc <= 31'b0;
            e3_exponent <= 8'b0; e3_fraction <= 23'b0;
        end else if (ena) begin
            // Stage propagation (non-blocking, pipeline style)
            e1_sign <= sign;      e2_sign <= e1_sign;      e3_sign <= e2_sign;
            e1_rm   <= rm;        e2_rm   <= e1_rm;        e3_rm   <= e2_rm;
            e1_exp10<= exp10;     e2_exp10<= e1_exp10;     e3_exp10<= e2_exp10;
            e1_ae00 <= a_expo_is_00; e2_ae00 <= e1_ae00;    e3_ae00 <= e2_ae00;
            e1_aeff <= a_expo_is_ff; e2_aeff <= e1_aeff;    e3_aeff <= e2_aeff;
            e1_af00 <= a_frac_is_00; e2_af00 <= e1_af00;    e3_af00 <= e2_af00;
            e1_be00 <= b_expo_is_00; e2_be00 <= e1_be00;    e3_be00 <= e2_be00;
            e1_beff <= b_expo_is_ff; e2_beff <= e1_beff;    e3_beff <= e2_beff;
            e1_bf00 <= b_frac_is_00; e2_bf00 <= e1_bf00;    e3_bf00 <= e2_bf00;

            // sample q and calc into pipeline
            e1_q   <= q;   e2_q   <= e1_q;   e3_q   <= e2_q;
            e1_calc<= calc_next; e2_calc<= e1_calc; e3_calc<= e2_calc;

            // compute final_result for the value that becomes E3 this cycle
            // use e2_* (old) because e3_* will receive e2_* after nonblocking update
            {e3_exponent, e3_fraction} <= final_result_reg(
                e2_calc,
                e2_rm, e2_sign,
                e2_ae00, e2_aeff, e2_af00,
                e2_be00, e2_beff, e2_bf00
            );
        end
    end

    // --- Combinational computation feeding pipeline (calc_next) ---
    // calc_next is computed from current newton output q and current exp10/rm/sign
    always @(*) begin
        // defaults
        calc_next = 31'b0;
        z0_msb = 1'b0; z0_local = 32'b0; exp_adj_local = 10'b0;
        frac0_local = 32'b0; exp0_local = 10'b0;
        frac_for_round = 27'b0; frac_plus_1_local = 1'b0; frac_round_local = 25'b0;
        exp1_local = 10'b0; overflow_local = 1'b0; rm_local = 2'b0; sign_local = 1'b0;

        // use current-stage signals (combinational)
        rm_local = rm;
        sign_local = sign;

        // normalize z0 and compute exp_adj
        z0_msb = q[31];
        if (z0_msb)
            z0_local = q;
        else
            z0_local = {q[30:0], 1'b0};

        exp_adj_local = z0_msb ? exp10 : (exp10 - 10'b1);

        // compute frac0 and exp0 (mirrors original logic)
        if (exp_adj_local[9]) begin
            // negative exponent -> denorm
            exp0_local = 10'b0;
            if (z0_local[31])
                // shift right by (1 - exp_adj_local)
                frac0_local = z0_local >> (10'b1 - exp_adj_local);
            else
                frac0_local = 32'b0;
        end else if (exp_adj_local == 10'b0) begin
            exp0_local = 10'b0;
            frac0_local = {1'b0, z0_local[31:2], |z0_local[1:0]};
        end else begin
            if (exp_adj_local > 10'd254) begin
                exp0_local = 10'hff;
                frac0_local = 32'b0;
            end else begin
                exp0_local = exp_adj_local;
                frac0_local = z0_local;
            end
        end

        // rounding preparation
        frac_for_round = {frac0_local[31:6], |frac0_local[5:0]}; // 27 bits

        // rounding decision (recreates original logic)
        frac_plus_1_local =
            (~rm_local[1] & ~rm_local[0] & frac_for_round[3] & frac_for_round[2] & ~frac_for_round[1] & ~frac_for_round[0]) |
            (~rm_local[1] & ~rm_local[0] & frac_for_round[2] & (frac_for_round[1] | frac_for_round[0])) |
            (~rm_local[1] & rm_local[0] & (|frac_for_round[2:0]) & sign_local) |
            (rm_local[1] & ~rm_local[0] & (|frac_for_round[2:0]) & ~sign_local);

        frac_round_local = {1'b0, frac_for_round[26:3]} + frac_plus_1_local;
        exp1_local = frac_round_local[24] ? (exp0_local + 10'h1) : exp0_local;
        overflow_local = (exp1_local >= 10'h0ff);

        // pack result as {exp1[7:0], frac_round[22:0]}
        calc_next = {exp1_local[7:0], frac_round_local[22:0]};
    end

    // final_result_reg: combinational helper returning {exponent[7:0], fraction[22:0]} given registered inputs
    function [30:0] final_result_reg;
        input [30:0] calc_in;                 // {exp[7:0], frac[22:0]}
        input [1:0]  e_rm;
        input        e_sign;
        input        a_e00, a_eff, a_f00, b_e00, b_eff, b_f00;
        begin
            casex ({ (calc_in[30:23] == 8'hff), e_rm, e_sign, a_e00, a_eff, a_f00, b_e00, b_eff, b_f00 })
                10'b100x_xxx_xxx : final_result_reg = INF32[30:0]; // Overflow to infinity
                10'b1010_xxx_xxx : final_result_reg = MAX32[30:0]; // Overflow, round to max
                10'b1011_xxx_xxx : final_result_reg = INF32[30:0]; // Overflow
                10'b1100_xxx_xxx : final_result_reg = INF32[30:0]; // Overflow
                10'b1101_xxx_xxx : final_result_reg = MAX32[30:0]; // Overflow
                10'b111x_xxx_xxx : final_result_reg = MAX32[30:0]; // Overflow
                10'b0xxx_010_xxx : final_result_reg = NaN32[30:0]; // NaN / any
                10'b0xxx_011_010 : final_result_reg = NaN32[30:0]; // Inf / NaN
                10'b0xxx_100_010 : final_result_reg = NaN32[30:0]; // Den / NaN
                10'b0xxx_101_010 : final_result_reg = NaN32[30:0]; // 0 / NaN
                10'b0xxx_00x_010 : final_result_reg = NaN32[30:0]; // Nor / NaN
                10'b0xxx_011_011 : final_result_reg = NaN32[30:0]; // Inf / Inf
                10'b0xxx_100_011 : final_result_reg = ZERO32[30:0]; // Den / Inf
                10'b0xxx_101_011 : final_result_reg = ZERO32[30:0]; // 0 / Inf
                10'b0xxx_00x_011 : final_result_reg = ZERO32[30:0]; // Nor / Inf
                10'b0xxx_011_101 : final_result_reg = INF32[30:0]; // Inf / 0
                10'b0xxx_100_101 : final_result_reg = INF32[30:0]; // Den / 0
                10'b0xxx_101_101 : final_result_reg = NaN32[30:0]; // 0 / 0
                10'b0xxx_00x_101 : final_result_reg = INF32[30:0]; // Nor / 0
                10'b0xxx_011_100 : final_result_reg = INF32[30:0]; // Inf / Den
                10'b0xxx_100_100 : final_result_reg = calc_in; // Den / Den
                10'b0xxx_101_100 : final_result_reg = ZERO32[30:0]; // 0 / Den
                10'b0xxx_00x_100 : final_result_reg = calc_in; // Nor / Den
                10'b0xxx_011_00x : final_result_reg = INF32[30:0]; // Inf / Nor
                10'b0xxx_100_00x : final_result_reg = calc_in; // Den / Nor
                10'b0xxx_101_00x : final_result_reg = ZERO32[30:0]; // 0 / Nor
                10'b0xxx_00x_00x : final_result_reg = calc_in; // Nor / Nor
                default : final_result_reg = ZERO32[30:0]; // Default to zero
            endcase
        end
    endfunction

    // Connect top-level output from registered E3 fields
    // assign s = {e3_sign, e3_exponent, e3_fraction};
    reg [31:0] s_reg;
    assign s = s_reg;

    // valid generation (keeps original semantic)
    always @(posedge clk or negedge clrn) begin
        if (!clrn) begin
            valid <= 1'b0;
            s_reg <= 32'b0;
        end
        else begin
            if (fdiv & (count==5'b0)) 
                valid <= 1'b0;          // 新输入清除 valid
                
            else if (!busy && (count != 5'b0)) begin
                s_reg <= {e3_sign, e3_exponent, e3_fraction};
                valid <= 1'b1;          // 迭代完成输出有效
            end
            else
                valid <= 1'b0;          // 计算中或 stall
        end
    end

endmodule


// Module to shift input until MSB is 1
module shift_to_msb_equ_1 (
    input  wire [23:0] a,   // Input fraction
    output wire [23:0] b,   // Normalized fraction (1.xxxx)
    output wire [4:0]  sa   // Shift amount
);
    wire [23:0] a5, a4, a3, a2, a1, a0;

    assign a5 = a;
    assign sa[4] = ~|a5[23:8];  // Check for 16-bit zero
    assign a4 = sa[4] ? {a5[7:0], 16'b0} : a5;
    assign sa[3] = ~|a4[23:16]; // Check for 8-bit zero
    assign a3 = sa[3] ? {a4[15:0], 8'b0} : a4;
    assign sa[2] = ~|a3[23:20]; // Check for 4-bit zero
    assign a2 = sa[2] ? {a3[19:0], 4'b0} : a3;
    assign sa[1] = ~|a2[23:22]; // Check for 2-bit zero
    assign a1 = sa[1] ? {a2[21:0], 2'b0} : a2;
    assign sa[0] = ~a1[23];     // Check for 1-bit zero
    assign a0 = sa[0] ? {a1[22:0], 1'b0} : a1;
    assign b = a0;
endmodule

// Newton-Raphson division for 24-bit fractions
module newton24 (
    input  wire [23:0] a,      // Dividend (0.1xxx)
    input  wire [23:0] b,      // Divisor (0.1xxx)
    input  wire        fdiv,   // Division instruction
    input  wire        ena,    // Enable signal
    input  wire        clk,    // Clock signal
    input  wire        clrn,   // Active-low reset
    output reg  [31:0] q,      // Quotient (x.xxxxx)
    output reg         busy,   // Busy signal
    output reg  [4:0]  count, // Iteration counter
    output reg  [25:0] reg_x, // Current approximation
    output wire        stall   // Pipeline stall signal
);
    // Internal registers
    reg [23:0] reg_a;         // Dividend register
    reg [23:0] reg_b;         // Divisor register
    reg [25:0] reg_de_x;      // Pipeline register for x (ID to E1)
    reg [23:0] reg_de_a;      // Pipeline register for a (ID to E1)
    reg [49:0] a_s;           // Sum register (E1 to E2)
    reg [49:8] a_c;           // Carry register (E1 to E2)

    // ROM for initial approximation
    function [7:0] rom;
        input [3:0] b;
        case (b)
            4'h0: rom = 8'hff; 4'h1: rom = 8'hdf;
            4'h2: rom = 8'hc3; 4'h3: rom = 8'haa;
            4'h4: rom = 8'h93; 4'h5: rom = 8'h7f;
            4'h6: rom = 8'h6d; 4'h7: rom = 8'h5c;
            4'h8: rom = 8'h4d; 4'h9: rom = 8'h3f;
            4'ha: rom = 8'h33; 4'hb: rom = 8'h27;
            4'hc: rom = 8'h1c; 4'hd: rom = 8'h12;
            4'he: rom = 8'h08; 4'hf: rom = 8'h00;
            default: rom = 8'h00;
        endcase
    endfunction

    // Newton-Raphson iteration signals
    wire [7:0]  x0 = rom(b[22:19]); // Initial approximation from ROM
    wire [49:0] bxi;                // xi * b
    wire [51:0] x52;                // xi * (2 - xi * b)
    wire [49:0] d_x;                // Final quotient sum
    wire [31:0] e2p;                // Quotient with sticky bit
    wire [49:0] m_s;                // Wallace tree sum
    wire [49:8] m_c;                // Wallace tree carry
    wire [7:0]  m_s_low;            // Wallace tree low bits

    // Wallace tree multiplications
    wallace_26x24_product bxxi (.a(reg_b), .b(reg_x), .z(bxi));
    wire [25:0] b26 = ~bxi[48:23] + 1'b1; // 2 - xi * b
    wallace_26x26_product xip1 (.a(reg_x), .b(b26), .z(x52));
    wallace_tree_24x26 wt (.a(reg_de_a), .b(reg_de_x), .x(m_s[49:8]), .y(m_c), .z(m_s[7:0]));

    // Final quotient calculation
    assign d_x = {1'b0, a_s} + {a_c, 8'b0};
    assign e2p = {d_x[48:18], |d_x[17:0]}; // Sticky bit
    assign stall = fdiv & (count == 5'b0) | busy;

    // Newton-Raphson state machine
    always @(posedge clk or negedge clrn) begin
        if (!clrn) begin
            busy <= 1'b0;
            count <= 5'b0;
            reg_x <= 26'b0;
            reg_a <= 24'b0;
            reg_b <= 24'b0;
            reg_de_x <= 26'b0;
            reg_de_a <= 24'b0;
            a_s <= 50'b0;
            a_c <= 42'b0;
            q <= 32'b0;
        end else begin
            if (fdiv & (count == 5'b0)) begin
                count <= 5'b1; // Start iteration
                busy <= 1'b1;  // Set busy
            end else begin
                if (count == 5'h01) begin
                    reg_a <= a;                // Store dividend
                    reg_b <= b;                // Store divisor
                    reg_x <= {2'b1, x0, 16'b0}; // Initialize x0
                end
                if (count != 5'b0) count <= count + 5'b1; // Increment counter
                if (count == 5'h0f) busy <= 1'b0;        // Clear busy
                if (count == 5'h10) count <= 5'b0;       // Reset counter
                if (count == 5'h06 || count == 5'h0b || count == 5'h10)
                    reg_x <= x52[50:25]; // Update approximation
            end
            if (ena) begin
                reg_de_x <= x52[50:25]; // Pipeline x
                reg_de_a <= reg_a;      // Pipeline a
                a_s <= m_s;             // Pipeline sum
                a_c <= m_c;             // Pipeline carry
                q <= e2p;               // Store quotient
            end
        end
    end
endmodule

// Wallace tree multiplier (26x24 bits)
module wallace_26x24_product (
    input  wire [23:0] a, // 24-bit input
    input  wire [25:0] b, // 26-bit input
    output wire [49:0] z  // 50-bit product
);
    wire [49:8] x;        // Sum high bits
    wire [49:8] y;        // Carry high bits
    wire [7:0]  z_low;    // Product low bits
    wire [49:8] z_high;   // Product high bits

    wallace_tree_24x26 wt_partial (.a(a), .b(b), .x(x), .y(y), .z(z_low));
    assign z_high = x + y;
    assign z = {z_high, z_low};
endmodule

// Wallace tree multiplier (26x26 bits)
module wallace_26x26_product (
    input  wire [25:0] a, // 26-bit input
    input  wire [25:0] b, // 26-bit input
    output wire [51:0] z  // 52-bit product
);
    wire [51:8] x;        // Sum high bits
    wire [51:8] y;        // Carry high bits
    wire [7:0]  z_low;    // Product low bits
    wire [51:8] z_high;   // Product high bits

    wallace_tree_26x26 wt_partial (.a(a), .b(b), .x(x), .y(y), .z(z_low));
    assign z_high = x + y;
    assign z = {z_high, z_low};
endmodule



