// // Floating-point adder/subtractor module
// module fp32_adder (
//     input  wire clk,           // Clock
//     input  wire rstn,          // Reset
//     input  wire ena,           // Input data valid
//     input  wire [31:0] a,      // First operand (IEEE 754 single-precision)
//     input  wire [31:0] b,      // Second operand (IEEE 754 single-precision)
//     input  wire [1:0]  rm,     // Rounding mode (00: nearest, 01: zero, 10: +inf, 11: -inf)
//     input  wire        sel,    // Operation: 1 = subtract, 0 = add
//     output reg [31:0] s,       // Result (IEEE 754 single-precision)
//     output reg valid
// );
//     // Compare operands to determine which is larger
//     wire        exchange = ({1'b0, b[30:0]} > {1'b0, a[30:0]}); // Compare magnitude (exponent + fraction)
//     wire [31:0] fp_large = exchange ? b : a;                    // Larger operand
//     wire [31:0] fp_small = exchange ? a : b;                    // Smaller operand

//     // Extract hidden bits for normalized numbers
//     wire        fp_large_hidden_bit = |fp_large[30:23];         // Hidden bit: 1 if exponent != 0
//     wire        fp_small_hidden_bit = |fp_small[30:23];         // Hidden bit: 1 if exponent != 0
//     wire [23:0] large_frac24 = {fp_large_hidden_bit, fp_large[22:0]}; // 24-bit fraction with hidden bit
//     wire [23:0] small_frac24 = {fp_small_hidden_bit, fp_small[22:0]}; // 24-bit fraction with hidden bit

//     // Control signals for operation and special cases
//     wire [7:0]  temp_exp = fp_large[30:23];                    // Exponent of larger operand
//     wire        sign = exchange ? (sel ^ b[31]) : a[31];       // Result sign
//     wire        op_sub = sel ^ fp_large[31] ^ fp_small[31];    // Effective operation (add or subtract)
//     wire        fp_large_expo_is_ff = &fp_large[30:23];        // Exponent of larger operand = 0xff (Inf/NaN)
//     wire        fp_small_expo_is_ff = &fp_small[30:23];        // Exponent of smaller operand = 0xff (Inf/NaN)
//     wire        fp_large_frac_is_00 = ~|fp_large[22:0];        // Fraction of larger operand = 0
//     wire        fp_small_frac_is_00 = ~|fp_small[22:0];        // Fraction of smaller operand = 0
//     wire        fp_large_is_inf = fp_large_expo_is_ff & fp_large_frac_is_00; // Larger operand is infinity
//     wire        fp_small_is_inf = fp_small_expo_is_ff & fp_small_frac_is_00; // Smaller operand is infinity
//     wire        fp_large_is_nan = fp_large_expo_is_ff & ~fp_large_frac_is_00; // Larger operand is NaN
//     wire        fp_small_is_nan = fp_small_expo_is_ff & ~fp_small_frac_is_00; // Smaller operand is NaN
//     wire        s_is_inf = fp_large_is_inf | fp_small_is_inf;   // Result is infinity
//     wire        s_is_nan = fp_large_is_nan | fp_small_is_nan | // Result is NaN
//                          ((sel ^ fp_small[31] ^ fp_large[31]) & fp_large_is_inf & fp_small_is_inf); // Inf - Inf case
//     wire [22:0] nan_frac = ({1'b0, a[22:0]} > {1'b0, b[22:0]}) ? {1'b1, a[21:0]} : {1'b1, b[21:0]}; // NaN fraction
//     wire [22:0] inf_nan_frac = s_is_nan ? nan_frac : 23'h0;    // Fraction for Inf/NaN result

//     // Align smaller operand's fraction
//     wire [7:0]  exp_diff = fp_large[30:23] - fp_small[30:23];  // Exponent difference
//     wire        small_den_only = (fp_large[30:23] != 0) & (fp_small[30:23] == 0); // Smaller operand is denormalized
//     wire [7:0]  shift_amount = small_den_only ? exp_diff - 8'h1 : exp_diff; // Adjust shift for denormalized case
//     wire [49:0] small_frac50 = (shift_amount >= 26) ? {26'h0, small_frac24} : ({small_frac24, 26'h0} >> shift_amount); // Shifted fraction
//     wire [26:0] small_frac27 = {small_frac50[49:24], |small_frac50[23:0]}; // 27-bit fraction with sticky bit

//     // Prepare fractions for addition/subtraction
//     wire [27:0] aligned_large_frac = {1'b0, large_frac24, 3'b000}; // Extend larger fraction
//     wire [27:0] aligned_small_frac = {1'b0, small_frac27};        // Extend smaller fraction
//     wire [27:0] cal_frac = op_sub ? aligned_large_frac - aligned_small_frac : aligned_large_frac + aligned_small_frac; // Add or subtract

//     // Normalize result
//     wire [26:0] f4, f3, f2, f1, f0; // Intermediate normalization steps
//     wire [4:0]  zeros;              // Leading zero count
//     assign zeros[4] = ~|cal_frac[26:11]; // Check for 16-bit zero
//     assign f4 = zeros[4] ? {cal_frac[10:0], 16'b0} : cal_frac[26:0];
//     assign zeros[3] = ~|f4[26:19];       // Check for 8-bit zero
//     assign f3 = zeros[3] ? {f4[18:0], 8'b0} : f4;
//     assign zeros[2] = ~|f3[26:23];       // Check for 4-bit zero
//     assign f2 = zeros[2] ? {f3[22:0], 4'b0} : f3;
//     assign zeros[1] = ~|f2[26:25];       // Check for 2-bit zero
//     assign f1 = zeros[1] ? {f2[24:0], 2'b0} : f2;
//     assign zeros[0] = ~f1[26];           // Check for 1-bit zero
//     assign f0 = zeros[0] ? {f1[25:0], 1'b0} : f1;

//     // Adjust exponent and fraction
//     reg [7:0]   exp0;   // Normalized exponent
//     reg [26:0]  frac0;  // Normalized fraction
//     always @(*) begin
//         if (cal_frac[27]) begin // Result is 1x.xxxx... (overflow)
//             frac0 = cal_frac[27:1]; // Remove overflow bit
//             exp0 = temp_exp + 8'h1; // Increment exponent
            
//         end else begin
//             if ((temp_exp > zeros) && (f0[26])) begin // Normalized number
//                 exp0 = temp_exp - zeros; // Adjust exponent
//                 frac0 = f0;              // Use normalized fraction
//             end else begin // Denormalized number or zero
//                 exp0 = 8'b0;             // Set exponent to zero
//                 if (temp_exp != 0)       // Denormalized case
//                     frac0 = cal_frac[26:0] << (temp_exp - 8'h1);
//                 else
//                     frac0 = cal_frac[26:0]; // No shift if already denormalized
//             end
//         end
//     end

//     // Rounding logic
//     wire frac_plus_1 = // Rounding decision based on mode
//         (~rm[1] & ~rm[0] & frac0[2] & (frac0[1] | frac0[0])) |
//         (~rm[1] & ~rm[0] & frac0[2] & ~frac0[1] & ~frac0[0] & frac0[3]) |
//         (~rm[1] & rm[0] & (frac0[2] | frac0[1] | frac0[0]) & sign) |
//         (rm[1] & ~rm[0] & (frac0[2] | frac0[1] | frac0[0]) & ~sign);
//     wire [24:0] frac_round = {1'b0, frac0[26:3]} + frac_plus_1; // Round fraction
//     wire [7:0]  exponent = frac_round[24] ? exp0 + 8'h1 : exp0; // Adjust exponent for rounding
//     wire        overflow = &exp0 | &exponent;                   // Check for overflow

//     // Final result assembly
//     wire [31:0] s_wire;
//     assign s_wire = final_result(overflow, rm, sign, s_is_nan, s_is_inf, exponent, frac_round[22:0], inf_nan_frac);

//     always @(posedge clk or negedge rstn) begin
//         if (!rstn) begin
//             s <= 32'h0000_0000;
//             valid <= 1'b0;
//         end else begin
//             if (ena) begin
//                 s <= s_wire;
//                 valid <= 1'b1;
//             end else begin
//                 valid <= 1'b0;
//             end
//         end
//     end
    
//     // Function to handle special cases and compute final result
//     function [31:0] final_result;
//         input        overflow;
//         input [1:0]  rm;
//         input        sign;
//         input        is_nan;
//         input        is_inf;
//         input [7:0]  exponent;
//         input [22:0] fraction;
//         input [22:0] inf_nan_frac;
//         casex ({overflow, rm, sign, s_is_nan, s_is_inf})
//             6'b1_00_x_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
//             6'b1_01_0_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max (round to zero)
//             6'b1_01_1_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
//             6'b1_10_0_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
//             6'b1_10_1_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max
//             6'b1_11_x_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max
//             6'b0_xx_x_0_0 : final_result = {sign, exponent, fraction}; // Normal result
//             6'bx_xx_x_1_x : final_result = {1'b1, 8'hff, inf_nan_frac}; // NaN
//             6'bx_xx_x_0_1 : final_result = {sign, 8'hff, inf_nan_frac}; // Infinity
//             default       : final_result = {sign, 8'h00, 23'h000000}; // Zero
//         endcase
//     endfunction
// endmodule

// fp32_adder - 3-stage pipelined version (preserves original logic, no simplification)
// module fp32_adder (
//     input  wire clk,           // Clock
//     input  wire rstn,          // Reset (active low)
//     input  wire ena,           // Input data valid
//     input  wire [31:0] a,      // First operand (IEEE 754 single-precision)
//     input  wire [31:0] b,      // Second operand (IEEE 754 single-precision)
//     input  wire [1:0]  rm,     // Rounding mode (00: nearest, 01: zero, 10: +inf, 11: -inf)
//     input  wire        sel,    // Operation: 1 = subtract, 0 = add
//     output reg [31:0] s,       // Result (IEEE 754 single-precision)
//     output reg valid
// );

//     // ---------------------------
//     // Stage 1 - register inputs and compute selection / small helpers
//     // ---------------------------
//     reg [31:0] a_s1, b_s1;
//     reg [1:0]  rm_s1;
//     reg        sel_s1;
//     reg        valid_s1;

//     // Signals derived in S1 (registered to S2)
//     reg        exchange_s1;
//     reg [31:0] fp_large_s1;
//     reg [31:0] fp_small_s1;
//     reg        fp_large_hidden_bit_s1;
//     reg        fp_small_hidden_bit_s1;
//     reg [23:0] large_frac24_s1;
//     reg [23:0] small_frac24_s1;
//     reg [7:0]  temp_exp_s1;
//     reg        sign_s1;
//     reg        op_sub_s1;
//     reg        fp_large_expo_is_ff_s1;
//     reg        fp_small_expo_is_ff_s1;
//     reg        fp_large_frac_is_00_s1;
//     reg        fp_small_frac_is_00_s1;
//     reg        fp_large_is_inf_s1;
//     reg        fp_small_is_inf_s1;
//     reg        s_is_inf_s1;
//     reg        s_is_nan_s1;
//     reg [22:0] nan_frac_s1;
//     reg [22:0] inf_nan_frac_s1;

//     // Align helpers (computed in S1)
//     reg [7:0]  exp_diff_s1;
//     reg        small_den_only_s1;
//     reg [7:0]  shift_amount_s1;
//     reg [49:0] small_frac50_s1;
//     reg [26:0] small_frac27_s1;

//     // compute S1 signals combinationally then register on clock edge
//     always @(*) begin
//         // exchange and selection based on magnitude (exponent+frac)
//         exchange_s1 = ({1'b0, b[30:0]} > {1'b0, a[30:0]});
//         fp_large_s1 = exchange_s1 ? b : a;
//         fp_small_s1 = exchange_s1 ? a : b;

//         // hidden bits (1 if exponent != 0)
//         fp_large_hidden_bit_s1 = |fp_large_s1[30:23];
//         fp_small_hidden_bit_s1 = |fp_small_s1[30:23];
//         large_frac24_s1 = {fp_large_hidden_bit_s1, fp_large_s1[22:0]};
//         small_frac24_s1 = {fp_small_hidden_bit_s1, fp_small_s1[22:0]};

//         temp_exp_s1 = fp_large_s1[30:23];
//         sign_s1 = exchange_s1 ? (sel ^ b[31]) : a[31];
//         op_sub_s1 = sel ^ fp_large_s1[31] ^ fp_small_s1[31];

//         fp_large_expo_is_ff_s1 = &fp_large_s1[30:23];
//         fp_small_expo_is_ff_s1 = &fp_small_s1[30:23];
//         fp_large_frac_is_00_s1 = ~|fp_large_s1[22:0];
//         fp_small_frac_is_00_s1 = ~|fp_small_s1[22:0];
//         fp_large_is_inf_s1 = fp_large_expo_is_ff_s1 & fp_large_frac_is_00_s1;
//         fp_small_is_inf_s1 = fp_small_expo_is_ff_s1 & fp_small_frac_is_00_s1;
//         s_is_inf_s1 = fp_large_is_inf_s1 | fp_small_is_inf_s1;
//         s_is_nan_s1 = fp_large_expo_is_ff_s1 & ~fp_large_frac_is_00_s1
//                       | fp_small_expo_is_ff_s1 & ~fp_small_frac_is_00_s1
//                       | ((sel ^ fp_small_s1[31] ^ fp_large_s1[31]) & fp_large_is_inf_s1 & fp_small_is_inf_s1);

//         // nan fraction selection (preserve original rule)
//         nan_frac_s1 = ({1'b0, a[22:0]} > {1'b0, b[22:0]}) ? {1'b1, a[21:0]} : {1'b1, b[21:0]};
//         inf_nan_frac_s1 = s_is_nan_s1 ? nan_frac_s1 : 23'h0;

//         // align smaller operand fraction
//         exp_diff_s1 = fp_large_s1[30:23] - fp_small_s1[30:23];
//         small_den_only_s1 = (fp_large_s1[30:23] != 0) & (fp_small_s1[30:23] == 0);
//         shift_amount_s1 = small_den_only_s1 ? (exp_diff_s1 - 8'h1) : exp_diff_s1;
//         if (shift_amount_s1 >= 8'd26)
//             small_frac50_s1 = {26'h0, small_frac24_s1};
//         else
//             small_frac50_s1 = ({small_frac24_s1, 26'h0} >> shift_amount_s1);
//         small_frac27_s1 = {small_frac50_s1[49:24], |small_frac50_s1[23:0]};
//     end

//     // register S1 outputs into S1 flops (to freeze values for next stage)
//     always @(posedge clk or negedge rstn) begin
//         if (!rstn) begin
//             a_s1 <= 32'b0;
//             b_s1 <= 32'b0;
//             rm_s1 <= 2'b00;
//             sel_s1 <= 1'b0;
//             valid_s1 <= 1'b0;

//             // zero other regs to safe default
//             fp_large_s1 <= 32'b0;
//             fp_small_s1 <= 32'b0;
//             large_frac24_s1 <= 24'b0;
//             small_frac24_s1 <= 24'b0;
//             temp_exp_s1 <= 8'b0;
//             sign_s1 <= 1'b0;
//             op_sub_s1 <= 1'b0;
//             fp_large_expo_is_ff_s1 <= 1'b0;
//             fp_small_expo_is_ff_s1 <= 1'b0;
//             fp_large_frac_is_00_s1 <= 1'b0;
//             fp_small_frac_is_00_s1 <= 1'b0;
//             fp_large_is_inf_s1 <= 1'b0;
//             fp_small_is_inf_s1 <= 1'b0;
//             s_is_inf_s1 <= 1'b0;
//             s_is_nan_s1 <= 1'b0;
//             nan_frac_s1 <= 23'b0;
//             inf_nan_frac_s1 <= 23'b0;
//             exp_diff_s1 <= 8'b0;
//             small_den_only_s1 <= 1'b0;
//             shift_amount_s1 <= 8'b0;
//             small_frac50_s1 <= 50'b0;
//             small_frac27_s1 <= 27'b0;
//         end else begin
//             if (ena) begin
//                 a_s1 <= a;
//                 b_s1 <= b;
//                 rm_s1 <= rm;
//                 sel_s1 <= sel;
//             end
//             valid_s1 <= ena;

//             // snapshot the combinational results computed above
//             fp_large_s1 <= fp_large_s1;
//             fp_small_s1 <= fp_small_s1;
//             large_frac24_s1 <= large_frac24_s1;
//             small_frac24_s1 <= small_frac24_s1;
//             temp_exp_s1 <= temp_exp_s1;
//             sign_s1 <= sign_s1;
//             op_sub_s1 <= op_sub_s1;
//             fp_large_expo_is_ff_s1 <= fp_large_expo_is_ff_s1;
//             fp_small_expo_is_ff_s1 <= fp_small_expo_is_ff_s1;
//             fp_large_frac_is_00_s1 <= fp_large_frac_is_00_s1;
//             fp_small_frac_is_00_s1 <= fp_small_frac_is_00_s1;
//             fp_large_is_inf_s1 <= fp_large_is_inf_s1;
//             fp_small_is_inf_s1 <= fp_small_is_inf_s1;
//             s_is_inf_s1 <= s_is_inf_s1;
//             s_is_nan_s1 <= s_is_nan_s1;
//             nan_frac_s1 <= nan_frac_s1;
//             inf_nan_frac_s1 <= inf_nan_frac_s1;
//             exp_diff_s1 <= exp_diff_s1;
//             small_den_only_s1 <= small_den_only_s1;
//             shift_amount_s1 <= shift_amount_s1;
//             small_frac50_s1 <= small_frac50_s1;
//             small_frac27_s1 <= small_frac27_s1;
//         end
//     end

//     // ---------------------------
//     // Stage 2 - alignment and add/sub (register S1 -> S2 values first)
//     // ---------------------------
//     // S2 registers (inputs to stage 2)
//     reg [7:0]  temp_exp_s2;
//     reg        sign_s2;
//     reg        op_sub_s2;
//     reg [1:0]  rm_s2;
//     reg        s_is_inf_s2;
//     reg        s_is_nan_s2;
//     reg [22:0] inf_nan_frac_s2;
//     reg [23:0] large_frac24_s2;
//     reg [26:0] small_frac27_s2;
//     reg        valid_s2;

//     // aligned fractions and cal_frac computed in S2 (then forwarded to S3)
//     reg [27:0] aligned_large_frac_s2;
//     reg [27:0] aligned_small_frac_s2;
//     reg [27:0] cal_frac_s2; // result of add/sub

//     // register S1 -> S2
//     always @(posedge clk or negedge rstn) begin
//         if (!rstn) begin
//             temp_exp_s2 <= 8'b0;
//             sign_s2 <= 1'b0;
//             op_sub_s2 <= 1'b0;
//             rm_s2 <= 2'b00;
//             s_is_inf_s2 <= 1'b0;
//             s_is_nan_s2 <= 1'b0;
//             inf_nan_frac_s2 <= 23'b0;
//             large_frac24_s2 <= 24'b0;
//             small_frac27_s2 <= 27'b0;
//             valid_s2 <= 1'b0;

//             aligned_large_frac_s2 <= 28'b0;
//             aligned_small_frac_s2 <= 28'b0;
//             cal_frac_s2 <= 28'b0;
//         end else begin
//             // forward control and metadata
//             temp_exp_s2 <= temp_exp_s1;
//             sign_s2 <= sign_s1;
//             op_sub_s2 <= op_sub_s1;
//             rm_s2 <= rm_s1;
//             s_is_inf_s2 <= s_is_inf_s1;
//             s_is_nan_s2 <= s_is_nan_s1;
//             inf_nan_frac_s2 <= inf_nan_frac_s1;
//             large_frac24_s2 <= large_frac24_s1;
//             small_frac27_s2 <= small_frac27_s1;
//             valid_s2 <= valid_s1;

//             // alignment (do per original logic)
//             aligned_large_frac_s2 <= {1'b0, large_frac24_s1, 3'b000};
//             aligned_small_frac_s2 <= {1'b0, small_frac27_s1};

//             // perform add/sub exactly as original
//             if (op_sub_s1)
//                 cal_frac_s2 <= ({1'b0, large_frac24_s1, 3'b000} - {1'b0, small_frac27_s1});
//             else
//                 cal_frac_s2 <= ({1'b0, large_frac24_s1, 3'b000} + {1'b0, small_frac27_s1});
//         end
//     end

//     // ---------------------------
//     // Stage 3 - normalization + rounding + final assembly
//     // ---------------------------
//     // S3 inputs (registered from S2)
//     reg [27:0] cal_frac_s3;
//     reg [7:0]  temp_exp_s3;
//     reg        sign_s3;
//     reg [1:0]  rm_s3;
//     reg        s_is_inf_s3;
//     reg        s_is_nan_s3;
//     reg [22:0] inf_nan_frac_s3;
//     reg        valid_s3;

//     // snapshot S2 -> S3
//     always @(posedge clk or negedge rstn) begin
//         if (!rstn) begin
//             cal_frac_s3 <= 28'b0;
//             temp_exp_s3 <= 8'b0;
//             sign_s3 <= 1'b0;
//             rm_s3 <= 2'b00;
//             s_is_inf_s3 <= 1'b0;
//             s_is_nan_s3 <= 1'b0;
//             inf_nan_frac_s3 <= 23'b0;
//             valid_s3 <= 1'b0;
//         end else begin
//             cal_frac_s3 <= cal_frac_s2;
//             temp_exp_s3 <= temp_exp_s2;
//             sign_s3 <= sign_s2;
//             rm_s3 <= rm_s2;
//             s_is_inf_s3 <= s_is_inf_s2;
//             s_is_nan_s3 <= s_is_nan_s2;
//             inf_nan_frac_s3 <= inf_nan_frac_s2;
//             valid_s3 <= valid_s2;
//         end
//     end

//     // Normalization tree (same structure & semantics as original)
//     wire [26:0] f4_s3, f3_s3, f2_s3, f1_s3, f0_s3;
//     wire [4:0]  zeros_s3;
//     assign zeros_s3[4] = ~|cal_frac_s3[26:11];
//     assign f4_s3 = zeros_s3[4] ? {cal_frac_s3[10:0], 16'b0} : cal_frac_s3[26:0];
//     assign zeros_s3[3] = ~|f4_s3[26:19];
//     assign f3_s3 = zeros_s3[3] ? {f4_s3[18:0], 8'b0} : f4_s3;
//     assign zeros_s3[2] = ~|f3_s3[26:23];
//     assign f2_s3 = zeros_s3[2] ? {f3_s3[22:0], 4'b0} : f3_s3;
//     assign zeros_s3[1] = ~|f2_s3[26:25];
//     assign f1_s3 = zeros_s3[1] ? {f2_s3[24:0], 2'b0} : f2_s3;
//     assign zeros_s3[0] = ~f1_s3[26];
//     assign f0_s3 = zeros_s3[0] ? {f1_s3[25:0], 1'b0} : f1_s3;

//     // regs for normalized results computed in combinational block
//     reg [7:0]  exp0_s3;
//     reg [26:0] frac0_s3;

//     always @(*) begin
//         // follow original always @(*) semantics for normalization
//         if (cal_frac_s3[27]) begin // overflow: 1x.x
//             frac0_s3 = cal_frac_s3[27:1];
//             exp0_s3 = temp_exp_s3 + 8'h1;
//         end else begin
//             if ((temp_exp_s3 > zeros_s3) && (f0_s3[26])) begin
//                 exp0_s3 = temp_exp_s3 - zeros_s3;
//                 frac0_s3 = f0_s3;
//             end else begin
//                 exp0_s3 = 8'b0;
//                 if (temp_exp_s3 != 0)
//                     frac0_s3 = cal_frac_s3[26:0] << (temp_exp_s3 - 8'h1);
//                 else
//                     frac0_s3 = cal_frac_s3[26:0];
//             end
//         end
//     end

//     // Rounding (exactly original)
//     wire frac_plus_3_s3 = (~rm_s3[1] & ~rm_s3[0] & frac0_s3[2] & (frac0_s3[1] | frac0_s3[0])) |
//                          (~rm_s3[1] & ~rm_s3[0] & frac0_s3[2] & ~frac0_s3[1] & ~frac0_s3[0] & frac0_s3[3]) |
//                          (~rm_s3[1] &  rm_s3[0] & (frac0_s3[2] | frac0_s3[1] | frac0_s3[0]) & sign_s3) |
//                          ( rm_s3[1] & ~rm_s3[0] & (frac0_s3[2] | frac0_s3[1] | frac0_s3[0]) & ~sign_s3);
//     wire [24:0] frac_round_s3 = {1'b0, frac0_s3[26:3]} + {24'h0, frac_plus_3_s3};
//     wire [7:0]  exponent_s3 = frac_round_s3[24] ? exp0_s3 + 8'h1 : exp0_s3;
//     wire        overflow_s3 = &exp0_s3 | &exponent_s3;

//     // final assembly (same final_result function semantics)
//     wire [31:0] s_wire_s3;
//     assign s_wire_s3 = final_result(overflow_s3, rm_s3, sign_s3, s_is_nan_s3, s_is_inf_s3, exponent_s3, frac_round_s3[22:0], inf_nan_frac_s3);

//     // Output register: align with valid_s3
//     always @(posedge clk or negedge rstn) begin
//         if (!rstn) begin
//             s <= 32'h0000_0000;
//             valid <= 1'b0;
//         end else begin
//             if (valid_s3) begin
//                 s <= s_wire_s3;
//             end
//             valid <= valid_s3;
//         end
//     end

//     // final_result function preserved from original (no change)
//     function [31:0] final_result;
//         input        overflow;
//         input [1:0]  rm;
//         input        sign;
//         input        is_nan;
//         input        is_inf;
//         input [7:0]  exponent;
//         input [22:0] fraction;
//         input [22:0] inf_nan_frac;
//         casex ({overflow, rm, sign, is_nan, is_inf})
//             6'b1_00_x_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
//             6'b1_01_0_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max (round to zero)
//             6'b1_01_1_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
//             6'b1_10_0_0_x : final_result = {sign, 8'hff, 23'h000000}; // Overflow to infinity
//             6'b1_10_1_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max
//             6'b1_11_x_0_x : final_result = {sign, 8'hfe, 23'h7fffff}; // Overflow to max
//             6'b0_xx_x_0_0 : final_result = {sign, exponent, fraction}; // Normal result
//             6'bx_xx_x_1_x : final_result = {1'b1, 8'hff, inf_nan_frac}; // NaN
//             6'bx_xx_x_0_1 : final_result = {sign, 8'hff, inf_nan_frac}; // Infinity
//             default       : final_result = {sign, 8'h00, 23'h000000}; // Zero
//         endcase
//     endfunction

// endmodule


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
