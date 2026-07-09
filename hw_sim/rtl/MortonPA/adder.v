module full_adder (
    input  a,
    input  b,
    input  cin,
    output sum,
    output cout
);
    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | (a & cin) | (b & cin);
endmodule

module ripple_carry_adder_nbit #(
    parameter WIDTH = 4
)(
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    input              cin,
    output [WIDTH-1:0] sum,
    output             cout
);
    wire [WIDTH:0] carry;
    assign carry[0] = cin;
    
    genvar i;
    generate
        for (i = 0; i < WIDTH; i = i + 1) begin : fa_gen
            full_adder fa (
                .a(a[i]),
                .b(b[i]),
                .cin(carry[i]),
                .sum(sum[i]),
                .cout(carry[i+1])
            );
        end
    endgenerate
    assign cout = carry[WIDTH];
endmodule

module csa_19b (
    input  [18:0] a,
    input  [18:0] b,
    input         cin,
    output [18:0] sum,
    output        cout
);
    wire [2:0] carry;
    
    //4-bit (bits 3:0)
    ripple_carry_adder_nbit #(.WIDTH(4)) block0 (
        .a(a[3:0]),
        .b(b[3:0]),
        .cin(cin),
        .sum(sum[3:0]),
        .cout(carry[0])
    );
    
    //4-bit (bits 7:4)
    wire [3:0] sum0_b1, sum1_b1;
    wire cout0_b1, cout1_b1;
    
    ripple_carry_adder_nbit #(.WIDTH(4)) block1_0 (
        .a(a[7:4]), .b(b[7:4]), .cin(1'b0),
        .sum(sum0_b1), .cout(cout0_b1)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(4)) block1_1 (
        .a(a[7:4]), .b(b[7:4]), .cin(1'b1),
        .sum(sum1_b1), .cout(cout1_b1)
    );
    
    assign sum[7:4] = carry[0] ? sum1_b1 : sum0_b1;
    assign carry[1] = carry[0] ? cout1_b1 : cout0_b1;
    
    //5-bit (bits 12:8)
    wire [4:0] sum0_b2, sum1_b2;
    wire cout0_b2, cout1_b2;
    
    ripple_carry_adder_nbit #(.WIDTH(5)) block2_0 (
        .a(a[12:8]), .b(b[12:8]), .cin(1'b0),
        .sum(sum0_b2), .cout(cout0_b2)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(5)) block2_1 (
        .a(a[12:8]), .b(b[12:8]), .cin(1'b1),
        .sum(sum1_b2), .cout(cout1_b2)
    );
    
    assign sum[12:8] = carry[1] ? sum1_b2 : sum0_b2;
    assign carry[2] = carry[1] ? cout1_b2 : cout0_b2;
    
    //6-bit (bits 18:13)
    wire [5:0] sum0_b3, sum1_b3;
    wire cout0_b3, cout1_b3;
    
    ripple_carry_adder_nbit #(.WIDTH(6)) block3_0 (
        .a(a[18:13]), .b(b[18:13]), .cin(1'b0),
        .sum(sum0_b3), .cout(cout0_b3)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(6)) block3_1 (
        .a(a[18:13]), .b(b[18:13]), .cin(1'b1),
        .sum(sum1_b3), .cout(cout1_b3)
    );
    
    assign sum[18:13] = carry[2] ? sum1_b3 : sum0_b3;
	assign cout = carry[2] ? cout1_b3 : cout0_b3;
/*    
    //6-bit (bits 24:19)
    wire [5:0] sum0_b4, sum1_b4;
    wire cout0_b4, cout1_b4;
    
    ripple_carry_adder_nbit #(.WIDTH(6)) block4_0 (
        .a(a[24:19]), .b(b[24:19]), .cin(1'b0),
        .sum(sum0_b4), .cout(cout0_b4)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(6)) block4_1 (
        .a(a[24:19]), .b(b[24:19]), .cin(1'b1),
        .sum(sum1_b4), .cout(cout1_b4)
    );
    
    assign sum[24:19] = carry[3] ? sum1_b4 : sum0_b4;
    assign carry[4] = carry[3] ? cout1_b4 : cout0_b4;
    
    //7-bit (bits 31:25)
    wire [6:0] sum0_b5, sum1_b5;
    wire cout0_b5, cout1_b5;
    
    ripple_carry_adder_nbit #(.WIDTH(7)) block5_0 (
        .a(a[31:25]), .b(b[31:25]), .cin(1'b0),
        .sum(sum0_b5), .cout(cout0_b5)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(7)) block5_1 (
        .a(a[31:25]), .b(b[31:25]), .cin(1'b1),
        .sum(sum1_b5), .cout(cout1_b5)
    );
    
    assign sum[31:25] = carry[4] ? sum1_b5 : sum0_b5;
    assign cout = carry[4] ? cout1_b5 : cout0_b5;
*/
endmodule

module csa_10b (
    input  [9:0] a,
    input  [9:0] b,
    input         cin,
    output [9:0] sum,
    output        cout
);
    wire [1:0] carry;
    
    //3-bit (bits 2:0)
    ripple_carry_adder_nbit #(.WIDTH(3)) block0 (
        .a(a[2:0]),
        .b(b[2:0]),
        .cin(cin),
        .sum(sum[2:0]),
        .cout(carry[0])
    );
    
    //3-bit (bits 5:3)
    wire [2:0] sum0_b1, sum1_b1;
    wire cout0_b1, cout1_b1;
    
    ripple_carry_adder_nbit #(.WIDTH(3)) block1_0 (
        .a(a[5:3]), .b(b[5:3]), .cin(1'b0),
        .sum(sum0_b1), .cout(cout0_b1)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(3)) block1_1 (
        .a(a[5:3]), .b(b[5:3]), .cin(1'b1),
        .sum(sum1_b1), .cout(cout1_b1)
    );
    
    assign sum[5:3] = carry[0] ? sum1_b1 : sum0_b1;
    assign carry[1] = carry[0] ? cout1_b1 : cout0_b1;
    
    //4-bit (bits 9:6)
    wire [3:0] sum0_b2, sum1_b2;
    wire cout0_b2, cout1_b2;
    
    ripple_carry_adder_nbit #(.WIDTH(4)) block2_0 (
        .a(a[9:6]), .b(b[9:6]), .cin(1'b0),
        .sum(sum0_b2), .cout(cout0_b2)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(4)) block2_1 (
        .a(a[9:6]), .b(b[9:6]), .cin(1'b1),
        .sum(sum1_b2), .cout(cout1_b2)
    );
    
    assign sum[9:6] = carry[1] ? sum1_b2 : sum0_b2;
    assign cout = carry[1] ? cout1_b2 : cout0_b2;
/*    
    //6-bit (bits 24:19)
    wire [5:0] sum0_b4, sum1_b4;
    wire cout0_b4, cout1_b4;
    
    ripple_carry_adder_nbit #(.WIDTH(6)) block4_0 (
        .a(a[24:19]), .b(b[24:19]), .cin(1'b0),
        .sum(sum0_b4), .cout(cout0_b4)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(6)) block4_1 (
        .a(a[24:19]), .b(b[24:19]), .cin(1'b1),
        .sum(sum1_b4), .cout(cout1_b4)
    );
    
    assign sum[24:19] = carry[3] ? sum1_b4 : sum0_b4;
    assign carry[4] = carry[3] ? cout1_b4 : cout0_b4;
    
    //7-bit (bits 31:25)
    wire [6:0] sum0_b5, sum1_b5;
    wire cout0_b5, cout1_b5;
    
    ripple_carry_adder_nbit #(.WIDTH(7)) block5_0 (
        .a(a[31:25]), .b(b[31:25]), .cin(1'b0),
        .sum(sum0_b5), .cout(cout0_b5)
    );
    
    ripple_carry_adder_nbit #(.WIDTH(7)) block5_1 (
        .a(a[31:25]), .b(b[31:25]), .cin(1'b1),
        .sum(sum1_b5), .cout(cout1_b5)
    );
    
    assign sum[31:25] = carry[4] ? sum1_b5 : sum0_b5;
    assign cout = carry[4] ? cout1_b5 : cout0_b5;
*/
endmodule


