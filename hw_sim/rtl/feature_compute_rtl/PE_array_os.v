`timescale 1ns/1ps
`include "PE_os.v"

module PE_array_os #(
    parameter ROW       = 4,
    parameter COL       = 4,
    parameter K         = 4,
    parameter WORD_SIZE = 8,
    parameter ACC_WIDTH = 32,
    parameter CNT_WIDTH = 16
)(
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire en,
    input  wire psum_clr,

    input  wire [ROW*WORD_SIZE-1:0] a_left_bus,
    input  wire [ROW-1:0]           a_valid_bus,

    input  wire [COL*WORD_SIZE-1:0] b_top_bus,
    input  wire [COL-1:0]           b_valid_bus,

    output wire [ROW*COL*ACC_WIDTH-1:0] c_bus,
    output wire [ROW*COL-1:0]           c_valid_bus,

    output reg done
);

    wire [ROW*(COL+1)*WORD_SIZE-1:0] a_mesh;
    wire [ROW*(COL+1)-1:0]           av_mesh;

    wire [COL*(ROW+1)*WORD_SIZE-1:0] b_mesh;
    wire [COL*(ROW+1)-1:0]           bv_mesh;

    reg [CNT_WIDTH-1:0] run_cnt;

    genvar r, c;

    generate
        for (r = 0; r < ROW; r = r + 1) begin : A_BOUNDARY
            assign a_mesh[(r*(COL+1)+0)*WORD_SIZE +: WORD_SIZE] = a_left_bus[r*WORD_SIZE +: WORD_SIZE];
            assign av_mesh[(r*(COL+1)+0)] = a_valid_bus[r];
        end

        for (c = 0; c < COL; c = c + 1) begin : B_BOUNDARY
            assign b_mesh[(c*(ROW+1)+0)*WORD_SIZE +: WORD_SIZE] = b_top_bus[c*WORD_SIZE +: WORD_SIZE];
            assign bv_mesh[(c*(ROW+1)+0)] = b_valid_bus[c];
        end

        for (r = 0; r < ROW; r = r + 1) begin : ROW_GEN
            for (c = 0; c < COL; c = c + 1) begin : COL_GEN
                wire signed [WORD_SIZE-1:0] a_in_w;
                wire                        av_in_w;
                wire signed [WORD_SIZE-1:0] a_out_w;
                wire                        av_out_w;

                wire signed [WORD_SIZE-1:0] b_in_w;
                wire                        bv_in_w;
                wire signed [WORD_SIZE-1:0] b_out_w;
                wire                        bv_out_w;

                wire signed [ACC_WIDTH-1:0] c_out_w;
                wire                        c_valid_w;

                assign a_in_w  = a_mesh[(r*(COL+1)+c)*WORD_SIZE +: WORD_SIZE];
                assign av_in_w = av_mesh[(r*(COL+1)+c)];

                assign b_in_w  = b_mesh[(c*(ROW+1)+r)*WORD_SIZE +: WORD_SIZE];
                assign bv_in_w = bv_mesh[(c*(ROW+1)+r)];

                PE_os #(
                    .WORD_SIZE(WORD_SIZE),
                    .ACC_WIDTH(ACC_WIDTH)
                ) u_pe (
                    .clk    (clk),
                    .rst_n  (rst_n),
                    .en     (en),
                    .psum_clr(psum_clr),

                    .a_in   (a_in_w),
                    .v_in_a (av_in_w),
                    .a_out  (a_out_w),
                    .v_out_a(av_out_w),

                    .b_in   (b_in_w),
                    .v_in_b (bv_in_w),
                    .b_out  (b_out_w),
                    .v_out_b(bv_out_w),

                    .c_out  (c_out_w),
                    .c_valid(c_valid_w)
                );

                assign a_mesh[(r*(COL+1)+c+1)*WORD_SIZE +: WORD_SIZE] = a_out_w;
                assign av_mesh[(r*(COL+1)+c+1)] = av_out_w;

                assign b_mesh[(c*(ROW+1)+r+1)*WORD_SIZE +: WORD_SIZE] = b_out_w;
                assign bv_mesh[(c*(ROW+1)+r+1)] = bv_out_w;

                assign c_bus[(r*COL+c)*ACC_WIDTH +: ACC_WIDTH] = c_out_w;
                assign c_valid_bus[r*COL+c] = c_valid_w;
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_cnt <= 0;
            done    <= 0;
        end else begin
            if (start) begin
                run_cnt <= 0;
                done    <= 0;
            end else if (en && !done) begin
                run_cnt <= run_cnt + 1'b1;
                if (run_cnt >= (K + ROW + COL + 2))
                    done <= 1'b1;
            end
        end
    end

endmodule