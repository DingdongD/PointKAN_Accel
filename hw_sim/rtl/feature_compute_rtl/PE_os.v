`timescale 1ns/1ps

module PE_os #(
    parameter WORD_SIZE = 8,
    parameter ACC_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,

    input  wire en,
    input  wire psum_clr,

    input  wire signed [WORD_SIZE-1:0] a_in,
    input  wire                        v_in_a,
    output reg  signed [WORD_SIZE-1:0] a_out,
    output reg                         v_out_a,

    input  wire signed [WORD_SIZE-1:0] b_in,
    input  wire                        v_in_b,
    output reg  signed [WORD_SIZE-1:0] b_out,
    output reg                         v_out_b,

    output reg  signed [ACC_WIDTH-1:0] c_out,
    output reg                         c_valid
);

    reg signed [ACC_WIDTH-1:0] psum_reg;
    wire signed [2*WORD_SIZE-1:0] mult_val;
    wire fire;

    assign mult_val = a_in * b_in;
    assign fire = en && v_in_a && v_in_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out    <= 0;
            b_out    <= 0;
            v_out_a  <= 0;
            v_out_b  <= 0;
            psum_reg <= 0;
            c_out    <= 0;
            c_valid  <= 0;
        end else begin
            if (en) begin
                a_out   <= a_in;
                b_out   <= b_in;
                v_out_a <= v_in_a;
                v_out_b <= v_in_b;
            end

            if (psum_clr) begin
                psum_reg <= 0;
                c_out    <= 0;
                c_valid  <= 0;
            end else if (fire) begin
                psum_reg <= psum_reg + mult_val;
                c_out    <= psum_reg + mult_val;
                c_valid  <= 1'b1;
            end
        end
    end

endmodule