module fp32_sub (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);
    wire [31:0] b_neg = {~b[31], b[30:0]};
    fp32_add u_add (.a(a), .b(b_neg), .result(result));
endmodule
