module fp32_max2 (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] y
);
    wire a_sign = a[31];
    wire b_sign = b[31];
    wire a_mag_ge_b = (a[30:0] >= b[30:0]);
    wire a_ge_b = (~a_sign & b_sign)                    
               | (~a_sign & ~b_sign &  a_mag_ge_b)      
               | ( a_sign &  b_sign & ~a_mag_ge_b);     
    assign y = a_ge_b ? a : b;
endmodule
