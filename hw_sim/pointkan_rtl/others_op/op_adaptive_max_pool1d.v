

module op_adaptive_max_pool1d #(
    parameter MAX_DIM = 64
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire [MAX_DIM*32-1:0]     x_in,
    input  wire [$clog2(MAX_DIM):0]  dim_count,
    input  wire                       valid_in,
    output reg  [31:0]                y_out,
    output reg                        valid_out
);

    localparam [31:0] FP32_NEG_INF = 32'hFF800000;

    wire [MAX_DIM*32-1:0] masked;

    genvar gi;
    generate
        for (gi = 0; gi < MAX_DIM; gi = gi + 1) begin : gen_mask
            assign masked[gi*32 +: 32] = (gi < dim_count)
                                        ? x_in[gi*32 +: 32]
                                        : FP32_NEG_INF;
        end
    endgenerate

    wire [32*32-1:0] lvl1;
    genvar i1;
    generate
        for (i1 = 0; i1 < 32; i1 = i1 + 1) begin : gen_lvl1
            fp32_max2 u_cmp (
                .a (masked[(2*i1)*32 +: 32]),
                .b (masked[(2*i1+1)*32 +: 32]),
                .y (lvl1[i1*32 +: 32])
            );
        end
    endgenerate

    wire [32*16-1:0] lvl2;
    genvar i2;
    generate
        for (i2 = 0; i2 < 16; i2 = i2 + 1) begin : gen_lvl2
            fp32_max2 u_cmp (
                .a (lvl1[(2*i2)*32 +: 32]),
                .b (lvl1[(2*i2+1)*32 +: 32]),
                .y (lvl2[i2*32 +: 32])
            );
        end
    endgenerate

    wire [32*8-1:0] lvl3;
    genvar i3;
    generate
        for (i3 = 0; i3 < 8; i3 = i3 + 1) begin : gen_lvl3
            fp32_max2 u_cmp (
                .a (lvl2[(2*i3)*32 +: 32]),
                .b (lvl2[(2*i3+1)*32 +: 32]),
                .y (lvl3[i3*32 +: 32])
            );
        end
    endgenerate

    wire [32*4-1:0] lvl4;
    genvar i4;
    generate
        for (i4 = 0; i4 < 4; i4 = i4 + 1) begin : gen_lvl4
            fp32_max2 u_cmp (
                .a (lvl3[(2*i4)*32 +: 32]),
                .b (lvl3[(2*i4+1)*32 +: 32]),
                .y (lvl4[i4*32 +: 32])
            );
        end
    endgenerate

    wire [32*2-1:0] lvl5;
    genvar i5;
    generate
        for (i5 = 0; i5 < 2; i5 = i5 + 1) begin : gen_lvl5
            fp32_max2 u_cmp (
                .a (lvl4[(2*i5)*32 +: 32]),
                .b (lvl4[(2*i5+1)*32 +: 32]),
                .y (lvl5[i5*32 +: 32])
            );
        end
    endgenerate

    wire [31:0] max_result;
    fp32_max2 u_cmp_final (
        .a (lvl5[0 +: 32]),
        .b (lvl5[32 +: 32]),
        .y (max_result)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out     <= 32'b0;
            valid_out <= 1'b0;
        end else begin
            y_out     <= max_result;
            valid_out <= valid_in;
        end
    end

endmodule