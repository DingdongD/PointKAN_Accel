`timescale 1ns/1ps
`include "controller_os_v2.v"
`include "controller_ws_preload_v2.v"
`include "controller_ws_diag_v2.v"

module controller_reconfig_v2 #(
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

    input  wire [ARR_C*ACC_WIDTH-1:0] vert_bottom_bus,
    input  wire [ARR_C-1:0]           vert_bottom_valid_bus,

    output wire en,
    output wire psum_clr,
    output wire [ARR_R*WORD_SIZE-1:0] a_left_bus,
    output wire [ARR_R-1:0]           a_valid_bus,
    output wire [ARR_C*ACC_WIDTH-1:0] vert_top_bus,
    output wire [ARR_C-1:0]           vert_top_valid_bus,
    output wire [ARR_R*ARR_C*WORD_SIZE-1:0] w_local_bus,
    output wire [ARR_R*ARR_C-1:0]           w_local_valid_bus,

    output wire [M*N*ACC_WIDTH-1:0] C_out_flat,
    output wire done
);

    localparam MODE_OS         = 2'd0;
    localparam MODE_WS_PRELOAD = 2'd1;
    localparam MODE_WS_DIAG    = 2'd2;

    // OS wires
    wire en_os, psum_clr_os, done_os;
    wire [ARR_R*WORD_SIZE-1:0]           a_left_bus_os;
    wire [ARR_R-1:0]                     a_valid_bus_os;
    wire [ARR_C*ACC_WIDTH-1:0]           vert_top_bus_os;
    wire [ARR_C-1:0]                     vert_top_valid_bus_os;
    wire [ARR_R*ARR_C*WORD_SIZE-1:0]     w_local_bus_os;
    wire [ARR_R*ARR_C-1:0]               w_local_valid_bus_os;

    // WS preload wires
    wire en_wsp, psum_clr_wsp, done_wsp;
    wire [ARR_R*WORD_SIZE-1:0]           a_left_bus_wsp;
    wire [ARR_R-1:0]                     a_valid_bus_wsp;
    wire [ARR_C*ACC_WIDTH-1:0]           vert_top_bus_wsp;
    wire [ARR_C-1:0]                     vert_top_valid_bus_wsp;
    wire [ARR_R*ARR_C*WORD_SIZE-1:0]     w_local_bus_wsp;
    wire [ARR_R*ARR_C-1:0]               w_local_valid_bus_wsp;
    wire [M*N*ACC_WIDTH-1:0]             C_out_flat_wsp;

    // WS diag wires
    wire en_wsd, psum_clr_wsd, done_wsd;
    wire [ARR_R*WORD_SIZE-1:0]           a_left_bus_wsd;
    wire [ARR_R-1:0]                     a_valid_bus_wsd;
    wire [ARR_C*ACC_WIDTH-1:0]           vert_top_bus_wsd;
    wire [ARR_C-1:0]                     vert_top_valid_bus_wsd;
    wire [ARR_R*ARR_C*WORD_SIZE-1:0]     w_local_bus_wsd;
    wire [ARR_R*ARR_C-1:0]               w_local_valid_bus_wsd;
    wire [M*N*ACC_WIDTH-1:0]             C_out_flat_wsd;

    controller_os_v2 #(
        .M(M), .K(K), .N(N),
        .ARR_R(ARR_R), .ARR_C(ARR_C),
        .WORD_SIZE(WORD_SIZE), .ACC_WIDTH(ACC_WIDTH)
    ) u_ctrl_os (
        .clk(clk),
        .rst_n(rst_n),
        .start(start && (mode == MODE_OS)),
        .A_flat(A_flat),
        .B_flat(B_flat),
        .en(en_os),
        .psum_clr(psum_clr_os),
        .a_left_bus(a_left_bus_os),
        .a_valid_bus(a_valid_bus_os),
        .vert_top_bus(vert_top_bus_os),
        .vert_top_valid_bus(vert_top_valid_bus_os),
        .w_local_bus(w_local_bus_os),
        .w_local_valid_bus(w_local_valid_bus_os),
        .done(done_os)
    );

    controller_ws_preload_v2 #(
        .M(M), .K(K), .N(N),
        .ARR_R(ARR_R), .ARR_C(ARR_C),
        .WORD_SIZE(WORD_SIZE), .ACC_WIDTH(ACC_WIDTH)
    ) u_ctrl_wsp (
        .clk(clk),
        .rst_n(rst_n),
        .start(start && (mode == MODE_WS_PRELOAD)),
        .A_flat(A_flat),
        .B_flat(B_flat),
        .vert_bottom_bus(vert_bottom_bus),
        .vert_bottom_valid_bus(vert_bottom_valid_bus),
        .en(en_wsp),
        .psum_clr(psum_clr_wsp),
        .a_left_bus(a_left_bus_wsp),
        .a_valid_bus(a_valid_bus_wsp),
        .vert_top_bus(vert_top_bus_wsp),
        .vert_top_valid_bus(vert_top_valid_bus_wsp),
        .w_local_bus(w_local_bus_wsp),
        .w_local_valid_bus(w_local_valid_bus_wsp),
        .C_out_flat(C_out_flat_wsp),
        .done(done_wsp)
    );

    controller_ws_diag_v2 #(
        .M(M), .K(K), .N(N),
        .ARR_R(ARR_R), .ARR_C(ARR_C),
        .WORD_SIZE(WORD_SIZE), .ACC_WIDTH(ACC_WIDTH)
    ) u_ctrl_wsd (
        .clk(clk),
        .rst_n(rst_n),
        .start(start && (mode == MODE_WS_DIAG)),
        .A_flat(A_flat),
        .B_flat(B_flat),
        .vert_bottom_bus(vert_bottom_bus),
        .vert_bottom_valid_bus(vert_bottom_valid_bus),
        .en(en_wsd),
        .psum_clr(psum_clr_wsd),
        .a_left_bus(a_left_bus_wsd),
        .a_valid_bus(a_valid_bus_wsd),
        .vert_top_bus(vert_top_bus_wsd),
        .vert_top_valid_bus(vert_top_valid_bus_wsd),
        .w_local_bus(w_local_bus_wsd),
        .w_local_valid_bus(w_local_valid_bus_wsd),
        .C_out_flat(C_out_flat_wsd),
        .done(done_wsd)
    );

    assign en                 = (mode == MODE_OS)         ? en_os               :
                                (mode == MODE_WS_PRELOAD) ? en_wsp              : en_wsd;

    assign psum_clr           = (mode == MODE_OS)         ? psum_clr_os         :
                                (mode == MODE_WS_PRELOAD) ? psum_clr_wsp        : psum_clr_wsd;

    assign a_left_bus         = (mode == MODE_OS)         ? a_left_bus_os       :
                                (mode == MODE_WS_PRELOAD) ? a_left_bus_wsp      : a_left_bus_wsd;

    assign a_valid_bus        = (mode == MODE_OS)         ? a_valid_bus_os      :
                                (mode == MODE_WS_PRELOAD) ? a_valid_bus_wsp     : a_valid_bus_wsd;

    assign vert_top_bus       = (mode == MODE_OS)         ? vert_top_bus_os     :
                                (mode == MODE_WS_PRELOAD) ? vert_top_bus_wsp    : vert_top_bus_wsd;

    assign vert_top_valid_bus = (mode == MODE_OS)         ? vert_top_valid_bus_os :
                                (mode == MODE_WS_PRELOAD) ? vert_top_valid_bus_wsp :
                                                           vert_top_valid_bus_wsd;

    assign w_local_bus        = (mode == MODE_OS)         ? w_local_bus_os      :
                                (mode == MODE_WS_PRELOAD) ? w_local_bus_wsp     : w_local_bus_wsd;

    assign w_local_valid_bus  = (mode == MODE_OS)         ? w_local_valid_bus_os :
                                (mode == MODE_WS_PRELOAD) ? w_local_valid_bus_wsp :
                                                           w_local_valid_bus_wsd;

    assign C_out_flat         = (mode == MODE_WS_PRELOAD) ? C_out_flat_wsp :
                                (mode == MODE_WS_DIAG)    ? C_out_flat_wsd :
                                                             {M*N*ACC_WIDTH{1'b0}};

    assign done               = (mode == MODE_OS)         ? done_os :
                                (mode == MODE_WS_PRELOAD) ? done_wsp : done_wsd;

endmodule