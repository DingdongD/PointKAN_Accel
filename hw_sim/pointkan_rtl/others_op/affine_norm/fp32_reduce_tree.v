

module fp32_reduce_tree (
    input  wire             clk,
    input  wire             rst_n,
    input  wire [32*32-1:0] x_in,
    input  wire             valid_in,
    output wire [31:0]      sum_out,
    output wire             valid_out
);

    wire [16*32-1:0] lvl1;
    wire [15:0]      lvl1_v;

    genvar i0;
    generate
        for (i0 = 0; i0 < 16; i0 = i0 + 1) begin : gen_l0
            fp32_add u_add (
                .clk(clk), .rst_n(rst_n),
                .a(x_in[(2*i0)*32     +: 32]),
                .b(x_in[(2*i0+1)*32   +: 32]),
                .valid_in(valid_in),
                .result(lvl1[i0*32 +: 32]),
                .valid_out(lvl1_v[i0])
            );
        end
    endgenerate

    wire [8*32-1:0] lvl2;
    wire [7:0]      lvl2_v;

    genvar i1;
    generate
        for (i1 = 0; i1 < 8; i1 = i1 + 1) begin : gen_l1
            fp32_add u_add (
                .clk(clk), .rst_n(rst_n),
                .a(lvl1[(2*i1)*32     +: 32]),
                .b(lvl1[(2*i1+1)*32   +: 32]),
                .valid_in(lvl1_v[2*i1]),
                .result(lvl2[i1*32 +: 32]),
                .valid_out(lvl2_v[i1])
            );
        end
    endgenerate

    wire [4*32-1:0] lvl3;
    wire [3:0]      lvl3_v;

    genvar i2;
    generate
        for (i2 = 0; i2 < 4; i2 = i2 + 1) begin : gen_l2
            fp32_add u_add (
                .clk(clk), .rst_n(rst_n),
                .a(lvl2[(2*i2)*32     +: 32]),
                .b(lvl2[(2*i2+1)*32   +: 32]),
                .valid_in(lvl2_v[2*i2]),
                .result(lvl3[i2*32 +: 32]),
                .valid_out(lvl3_v[i2])
            );
        end
    endgenerate

    wire [2*32-1:0] lvl4;
    wire [1:0]      lvl4_v;

    genvar i3;
    generate
        for (i3 = 0; i3 < 2; i3 = i3 + 1) begin : gen_l3
            fp32_add u_add (
                .clk(clk), .rst_n(rst_n),
                .a(lvl3[(2*i3)*32     +: 32]),
                .b(lvl3[(2*i3+1)*32   +: 32]),
                .valid_in(lvl3_v[2*i3]),
                .result(lvl4[i3*32 +: 32]),
                .valid_out(lvl4_v[i3])
            );
        end
    endgenerate

    fp32_add u_final (
        .clk(clk), .rst_n(rst_n),
        .a(lvl4[0  +: 32]),
        .b(lvl4[32 +: 32]),
        .valid_in(lvl4_v[0]),
        .result(sum_out),
        .valid_out(valid_out)
    );

endmodule