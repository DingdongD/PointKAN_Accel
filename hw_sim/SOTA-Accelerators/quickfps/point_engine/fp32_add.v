module fp32_add (
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
