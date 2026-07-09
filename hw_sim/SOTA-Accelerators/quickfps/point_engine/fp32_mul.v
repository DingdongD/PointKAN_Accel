module fp32_mul (
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
