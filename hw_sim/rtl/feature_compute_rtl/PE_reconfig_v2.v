`timescale 1ns/1ps

module PE_reconfig_v2 #(
    parameter WORD_SIZE = 16,
    parameter ACC_WIDTH = 64
)(
    input  wire clk,
    input  wire rst_n,

    input  wire en,
    input  wire [1:0] mode,
    input  wire psum_clr,

    // horizontal activation path
    input  wire signed [WORD_SIZE-1:0] a_in,
    input  wire                        a_valid_in,
    output reg  signed [WORD_SIZE-1:0] a_out,
    output reg                         a_valid_out,

    // reusable vertical path
    // MODE_OS         : carries weight stream
    // MODE_WS_*       : carries psum
    input  wire signed [ACC_WIDTH-1:0] vert_in,
    input  wire                        vert_valid_in,
    output reg  signed [ACC_WIDTH-1:0] vert_out,
    output reg                         vert_valid_out,

    // unified local weight injection port
    // WS_PRELOAD : all PE may receive local weight in preload phase
    // WS_DIAG    : only diagonal-selected PE receive local weight each cycle
    input  wire signed [WORD_SIZE-1:0] w_local_in,
    input  wire                        w_local_valid,

    // debug / observation
    output reg  signed [ACC_WIDTH-1:0] psum_snapshot
);

    localparam MODE_OS         = 2'd0;
    localparam MODE_WS_PRELOAD = 2'd1;
    localparam MODE_WS_DIAG    = 2'd2;

    reg signed [WORD_SIZE-1:0] weight_reg;
    reg                        weight_loaded;
    reg signed [ACC_WIDTH-1:0] psum_reg;

    wire signed [WORD_SIZE-1:0] os_weight_in;
    wire signed [WORD_SIZE-1:0] ws_weight_eff;
    wire                        ws_weight_eff_valid;
    wire signed [2*WORD_SIZE-1:0] mult_os;
    wire signed [2*WORD_SIZE-1:0] mult_ws;

    assign os_weight_in       = vert_in[WORD_SIZE-1:0];
    assign ws_weight_eff      = w_local_valid ? w_local_in : weight_reg;
    assign ws_weight_eff_valid= w_local_valid | weight_loaded;

    assign mult_os = a_in * os_weight_in;
    assign mult_ws = a_in * ws_weight_eff;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg      <= 0;
            weight_loaded   <= 1'b0;
            psum_reg        <= 0;
            a_out           <= 0;
            a_valid_out     <= 0;
            vert_out        <= 0;
            vert_valid_out  <= 0;
            psum_snapshot   <= 0;
        end else begin
            if (w_local_valid) begin
                weight_reg    <= w_local_in;
                weight_loaded <= 1'b1;
            end

            if (en) begin
                // always propagate activation horizontally
                a_out       <= a_in;
                a_valid_out <= a_valid_in;

                case (mode)
                    MODE_OS: begin
                        // vertical carries weight stream
                        vert_out       <= vert_in;
                        vert_valid_out <= vert_valid_in;

                        if (psum_clr) begin
                            psum_reg      <= 0;
                            psum_snapshot <= 0;
                        end else if (a_valid_in && vert_valid_in) begin
                            psum_reg      <= psum_reg + mult_os;
                            psum_snapshot <= psum_reg + mult_os;
                        end
                    end

                    MODE_WS_PRELOAD,
                    MODE_WS_DIAG: begin
                        // vertical carries psum
                        if (psum_clr) begin
                            vert_out       <= 0;
                            vert_valid_out <= 0;
                            psum_snapshot  <= 0;
                        end else if (a_valid_in && vert_valid_in && ws_weight_eff_valid) begin
                            vert_out       <= vert_in + mult_ws;
                            vert_valid_out <= 1'b1;
                            psum_snapshot  <= vert_in + mult_ws;
                        end else begin
                            vert_out       <= vert_in;
                            vert_valid_out <= vert_valid_in;
                            psum_snapshot  <= vert_out;
                        end
                    end

                    default: begin
                        vert_out       <= vert_in;
                        vert_valid_out <= vert_valid_in;
                    end
                endcase
            end
        end
    end

endmodule