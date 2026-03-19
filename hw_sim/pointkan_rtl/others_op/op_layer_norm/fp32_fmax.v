

module fp32_fmax (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] out
);

    wire a_neg = a[31];
    wire b_neg = b[31];

    reg a_greater;

    always @(*) begin
        if (a_neg && !b_neg)
            a_greater = 1'b0;
        else if (!a_neg && b_neg)
            a_greater = 1'b1;
        else if (!a_neg)
            a_greater = (a[30:0] > b[30:0]);
        else
            a_greater = (a[30:0] < b[30:0]);
    end

    assign out = a_greater ? a : b;

endmodule