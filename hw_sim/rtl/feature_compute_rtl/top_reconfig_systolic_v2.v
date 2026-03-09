`timescale 1ns/1ps
`include "PE_array_reconfig_v2.v"
`include "controller_reconfig_v2.v"

module top_reconfig_systolic_v2 #(
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

    output wire [M*N*ACC_WIDTH-1:0] C_out_flat,
    output wire done
);

    localparam MODE_OS         = 2'd0;
    localparam MODE_WS_PRELOAD = 2'd1;
    localparam MODE_WS_DIAG    = 2'd2;

    wire en;
    wire psum_clr;

    wire [ARR_R*WORD_SIZE-1:0]       a_left_bus;
    wire [ARR_R-1:0]                 a_valid_bus;
    wire [ARR_C*ACC_WIDTH-1:0]       vert_top_bus;
    wire [ARR_C-1:0]                 vert_top_valid_bus;
    wire [ARR_R*ARR_C*WORD_SIZE-1:0] w_local_bus;
    wire [ARR_R*ARR_C-1:0]           w_local_valid_bus;

    wire [ARR_C*ACC_WIDTH-1:0]       vert_bottom_bus;
    wire [ARR_C-1:0]                 vert_bottom_valid_bus;
    wire [ARR_R*ARR_C*ACC_WIDTH-1:0] psum_snapshot_bus;

    wire [M*N*ACC_WIDTH-1:0] C_ctrl_flat;
    wire done_ctrl;

    controller_reconfig_v2 #(
        .M(M), .K(K), .N(N),
        .ARR_R(ARR_R), .ARR_C(ARR_C),
        .WORD_SIZE(WORD_SIZE), .ACC_WIDTH(ACC_WIDTH)
    ) u_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .mode(mode),
        .A_flat(A_flat),
        .B_flat(B_flat),
        .vert_bottom_bus(vert_bottom_bus),
        .vert_bottom_valid_bus(vert_bottom_valid_bus),
        .en(en),
        .psum_clr(psum_clr),
        .a_left_bus(a_left_bus),
        .a_valid_bus(a_valid_bus),
        .vert_top_bus(vert_top_bus),
        .vert_top_valid_bus(vert_top_valid_bus),
        .w_local_bus(w_local_bus),
        .w_local_valid_bus(w_local_valid_bus),
        .C_out_flat(C_ctrl_flat),
        .done(done_ctrl)
    );

    PE_array_reconfig_v2 #(
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

    // output selection
    genvar r, c;
    generate
        for (r = 0; r < M; r = r + 1) begin : R_GEN
            for (c = 0; c < N; c = c + 1) begin : C_GEN
                if (r < ARR_R && c < ARR_C) begin
                    assign C_out_flat[(r*N+c)*ACC_WIDTH +: ACC_WIDTH] =
                        (mode == MODE_OS) ? psum_snapshot_bus[(r*ARR_C+c)*ACC_WIDTH +: ACC_WIDTH]
                                          : C_ctrl_flat[(r*N+c)*ACC_WIDTH +: ACC_WIDTH];
                end else begin
                    assign C_out_flat[(r*N+c)*ACC_WIDTH +: ACC_WIDTH] =
                        C_ctrl_flat[(r*N+c)*ACC_WIDTH +: ACC_WIDTH];
                end
            end
        end
    endgenerate

    assign done = done_ctrl;

endmodule