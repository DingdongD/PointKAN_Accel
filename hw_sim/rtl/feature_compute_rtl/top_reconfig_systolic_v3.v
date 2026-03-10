`timescale 1ns/1ps
`include "PE_array_reconfig_v3.v"

module top_reconfig_systolic_v3 #(
    parameter M          = 8,
    parameter K          = 8,
    parameter N          = 8,
    parameter ARR_R      = 4,
    parameter ARR_C      = 4,
    parameter WORD_SIZE  = 16,
    parameter ACC_WIDTH  = 64
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [1:0] mode,

    input  wire [M*K*WORD_SIZE-1:0] A_flat,
    input  wire [K*N*WORD_SIZE-1:0] B_flat,

    output reg  [M*N*ACC_WIDTH-1:0] C_out_flat,
    output reg  done
);

    localparam MODE_OS         = 2'd0;
    localparam MODE_WS_PRELOAD = 2'd1;
    localparam MODE_WS_DIAG    = 2'd2;
    localparam MODE_EW         = 2'd3;

    reg en, psum_clr;
    reg [ARR_R*WORD_SIZE-1:0]       a_left_bus;
    reg [ARR_R-1:0]                 a_valid_bus;
    reg [ARR_C*ACC_WIDTH-1:0]       vert_top_bus;
    reg [ARR_C-1:0]                 vert_top_valid_bus;
    reg [ARR_R*ARR_C*WORD_SIZE-1:0] w_local_bus;
    reg [ARR_R*ARR_C-1:0]           w_local_valid_bus;

    wire [ARR_C*ACC_WIDTH-1:0]       vert_bottom_bus;
    wire [ARR_C-1:0]                 vert_bottom_valid_bus;
    wire [ARR_R*ARR_C*ACC_WIDTH-1:0] psum_snapshot_bus;

    PE_array_reconfig_v3 #(
        .ARR_R(ARR_R),
        .ARR_C(ARR_C),
        .WORD_SIZE(WORD_SIZE),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_array (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .mode(mode),
        .psum_clr(psum_clr),
        .a_left_bus(a_left_bus),
        .a_valid_bus(a_valid_bus),
        .vert_top_bus(vert_top_bus),
        .vert_top_valid_bus(vert_top_valid_bus),
        .w_local_bus(w_local_bus),
        .w_local_valid_bus(w_local_valid_bus),
        .vert_bottom_bus(vert_bottom_bus),
        .vert_bottom_valid_bus(vert_bottom_valid_bus),
        .psum_snapshot_bus(psum_snapshot_bus)
    );

    // This lightweight top only actively supports MODE_EW for now.
    // For other modes, it stays idle.
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            en <= 0;
            psum_clr <= 0;
            a_left_bus <= 0;
            a_valid_bus <= 0;
            vert_top_bus <= 0;
            vert_top_valid_bus <= 0;
            w_local_bus <= 0;
            w_local_valid_bus <= 0;
            C_out_flat <= 0;
            done <= 0;
        end else begin
            // defaults
            a_left_bus <= 0;
            a_valid_bus <= 0;
            vert_top_bus <= 0;
            vert_top_valid_bus <= 0;
            w_local_bus <= 0;
            w_local_valid_bus <= 0;
            done <= 0;

            if (mode == MODE_EW) begin
                if (start) begin
                    en <= 1;
                    psum_clr <= 1'b1;
                end else begin
                    psum_clr <= 1'b0;
                end
            end else begin
                en <= 0;
                psum_clr <= 0;
            end
        end
    end

endmodule