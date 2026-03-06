module fp32_add (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire        valid_in,
    output reg  [31:0] result,
    output reg         valid_out
);
    wire        a_sign = a[31];
    wire [7:0]  a_exp  = a[30:23];
    wire [22:0] a_mant = a[22:0];
    wire        b_sign = b[31];
    wire [7:0]  b_exp  = b[30:23];
    wire [22:0] b_mant = b[22:0];
    wire a_is_zero = (a_exp == 8'd0) && (a_mant == 23'd0);
    wire b_is_zero = (b_exp == 8'd0) && (b_mant == 23'd0);
    wire a_is_inf = (a_exp == 8'hFF) && (a_mant == 23'd0);
    wire b_is_inf = (b_exp == 8'hFF) && (b_mant == 23'd0);
    wire [23:0] a_mant_ext = (a_exp == 8'd0) ? {1'b0, a_mant} : {1'b1, a_mant};
    wire [23:0] b_mant_ext = (b_exp == 8'd0) ? {1'b0, b_mant} : {1'b1, b_mant};
    wire a_exp_ge = (a[30:0] >= b[30:0]);
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
    reg  [4:0]  leading_zero_count;
    reg  [7:0]  norm_exp;
    reg  [22:0] norm_mant;
    reg  [31:0] result_comb;
    always @(*) begin
        leading_zero_count = 5'd0;
        norm_exp = 8'd0;
        norm_mant = 23'd0;
        result_comb = 32'd0;
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
            norm_exp = aligned_exp + 8'd1;
            norm_mant = mant_abs[23:1]; 
            if (norm_exp == 8'hFF) begin
                result_comb = {res_sign, 8'hFF, 23'd0}; 
            end else begin
                result_comb = {res_sign, norm_exp, norm_mant};
            end
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
                norm_exp = aligned_exp - {3'd0, leading_zero_count};
                begin : norm_shift_block
                    reg [23:0] shifted_mant;
                    shifted_mant = mant_abs[23:0] << leading_zero_count;
                    norm_mant = shifted_mant[22:0];
                end
                result_comb = {res_sign, norm_exp, norm_mant};
            end
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result    <= 32'd0;
            valid_out <= 1'b0;
        end else begin
            result    <= result_comb;
            valid_out <= valid_in;
        end
    end
endmodule
