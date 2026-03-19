

module fp32_add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] result
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

    wire        a_exp_ge    = (a[30:0] >= b[30:0]);
    wire [7:0]  exp_diff    = a_exp_ge ? (a_exp - b_exp) : (b_exp - a_exp);
    wire [4:0]  shift_amt   = (exp_diff > 8'd24) ? 5'd24 : exp_diff[4:0];

    wire [24:0] large_mant  = a_exp_ge ? {1'b0, a_mant_ext} : {1'b0, b_mant_ext};
    wire [24:0] small_shifted = a_exp_ge
                    ? ({1'b0, b_mant_ext} >> shift_amt)
                    : ({1'b0, a_mant_ext} >> shift_amt);

    wire        large_sign  = a_exp_ge ? a_sign : b_sign;
    wire [7:0]  aligned_exp = a_exp_ge ? a_exp  : b_exp;

    wire        eff_sub  = (large_sign ^ (a_exp_ge ? b_sign : a_sign));
    wire [25:0] mant_sum = eff_sub
                    ? ({1'b0, large_mant} - {1'b0, small_shifted})
                    : ({1'b0, large_mant} + {1'b0, small_shifted});

    wire        res_sign = large_sign;

    reg  [4:0]  lzc;
    reg  [7:0]  norm_exp;
    reg  [22:0] norm_mant;

    always @(*) begin
        lzc       = 5'd0;
        norm_exp  = 8'd0;
        norm_mant = 23'd0;
        result    = 32'd0;

        if (a_is_zero && b_is_zero)
            result = 32'd0;
        else if (a_is_zero)
            result = b;
        else if (b_is_zero)
            result = a;
        else if (a_is_inf)
            result = a;
        else if (b_is_inf)
            result = b;
        else if (mant_sum == 26'd0)
            result = 32'd0;
        else if (mant_sum[24]) begin
            norm_exp  = aligned_exp + 8'd1;
            norm_mant = mant_sum[23:1];
            if (norm_exp == 8'hFF)
                result = {res_sign, 8'hFF, 23'd0};
            else
                result = {res_sign, norm_exp, norm_mant};
        end
        else begin
            if      (mant_sum[23]) lzc = 5'd0;
            else if (mant_sum[22]) lzc = 5'd1;
            else if (mant_sum[21]) lzc = 5'd2;
            else if (mant_sum[20]) lzc = 5'd3;
            else if (mant_sum[19]) lzc = 5'd4;
            else if (mant_sum[18]) lzc = 5'd5;
            else if (mant_sum[17]) lzc = 5'd6;
            else if (mant_sum[16]) lzc = 5'd7;
            else if (mant_sum[15]) lzc = 5'd8;
            else if (mant_sum[14]) lzc = 5'd9;
            else if (mant_sum[13]) lzc = 5'd10;
            else if (mant_sum[12]) lzc = 5'd11;
            else if (mant_sum[11]) lzc = 5'd12;
            else if (mant_sum[10]) lzc = 5'd13;
            else if (mant_sum[9])  lzc = 5'd14;
            else if (mant_sum[8])  lzc = 5'd15;
            else if (mant_sum[7])  lzc = 5'd16;
            else if (mant_sum[6])  lzc = 5'd17;
            else if (mant_sum[5])  lzc = 5'd18;
            else if (mant_sum[4])  lzc = 5'd19;
            else if (mant_sum[3])  lzc = 5'd20;
            else if (mant_sum[2])  lzc = 5'd21;
            else if (mant_sum[1])  lzc = 5'd22;
            else                   lzc = 5'd23;

            if (aligned_exp <= {3'd0, lzc})
                result = 32'd0;
            else begin
                norm_exp = aligned_exp - {3'd0, lzc};
                begin : shift_block
                    reg [23:0] shifted;
                    shifted   = mant_sum[23:0] << lzc;
                    norm_mant = shifted[22:0];
                end
                result = {res_sign, norm_exp, norm_mant};
            end
        end
    end

endmodule