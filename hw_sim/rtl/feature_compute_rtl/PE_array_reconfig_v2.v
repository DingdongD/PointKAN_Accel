`timescale 1ns/1ps
`include "PE_reconfig_v2.v"

module PE_array_reconfig_v2 #(
    parameter ARR_R     = 4,
    parameter ARR_C     = 4,
    parameter WORD_SIZE = 16,
    parameter ACC_WIDTH = 64
)(
    input  wire clk,
    input  wire rst_n,

    input  wire en,
    input  wire [1:0] mode,
    input  wire psum_clr,

    // activation left boundary
    input  wire [ARR_R*WORD_SIZE-1:0] a_left_bus,
    input  wire [ARR_R-1:0]           a_valid_bus,

    // reusable vertical top boundary
    input  wire [ARR_C*ACC_WIDTH-1:0] vert_top_bus,
    input  wire [ARR_C-1:0]           vert_top_valid_bus,

    // unified local weight injection bus (one local port per PE)
    input  wire [ARR_R*ARR_C*WORD_SIZE-1:0] w_local_bus,
    input  wire [ARR_R*ARR_C-1:0]           w_local_valid_bus,

    // bottom vertical outputs
    output wire [ARR_C*ACC_WIDTH-1:0] vert_bottom_bus,
    output wire [ARR_C-1:0]           vert_bottom_valid_bus,

    // PE snapshots
    output wire [ARR_R*ARR_C*ACC_WIDTH-1:0] psum_snapshot_bus
);

    // activation mesh
    wire [ARR_R*(ARR_C+1)*WORD_SIZE-1:0] a_mesh;
    wire [ARR_R*(ARR_C+1)-1:0]           av_mesh;

    // vertical reusable mesh
    wire [ARR_C*(ARR_R+1)*ACC_WIDTH-1:0] v_mesh;
    wire [ARR_C*(ARR_R+1)-1:0]           vv_mesh;

    genvar r, c;

    generate
        // left activation boundary
        for (r = 0; r < ARR_R; r = r + 1) begin : A_BOUNDARY
            assign a_mesh[(r*(ARR_C+1)+0)*WORD_SIZE +: WORD_SIZE] = a_left_bus[r*WORD_SIZE +: WORD_SIZE];
            assign av_mesh[(r*(ARR_C+1)+0)] = a_valid_bus[r];
        end

        // top vertical boundary
        for (c = 0; c < ARR_C; c = c + 1) begin : V_BOUNDARY
            assign v_mesh[(c*(ARR_R+1)+0)*ACC_WIDTH +: ACC_WIDTH] = vert_top_bus[c*ACC_WIDTH +: ACC_WIDTH];
            assign vv_mesh[(c*(ARR_R+1)+0)] = vert_top_valid_bus[c];
        end

        for (r = 0; r < ARR_R; r = r + 1) begin : ROW_GEN
            for (c = 0; c < ARR_C; c = c + 1) begin : COL_GEN
                wire signed [WORD_SIZE-1:0] a_in_w;
                wire                        av_in_w;
                wire signed [WORD_SIZE-1:0] a_out_w;
                wire                        av_out_w;

                wire signed [ACC_WIDTH-1:0] v_in_w;
                wire                        vv_in_w;
                wire signed [ACC_WIDTH-1:0] v_out_w;
                wire                        vv_out_w;

                wire signed [WORD_SIZE-1:0] w_local_w;
                wire                        w_local_valid_w;

                wire signed [ACC_WIDTH-1:0] psum_snap_w;

                assign a_in_w  = a_mesh[(r*(ARR_C+1)+c)*WORD_SIZE +: WORD_SIZE];
                assign av_in_w = av_mesh[(r*(ARR_C+1)+c)];

                assign v_in_w  = v_mesh[(c*(ARR_R+1)+r)*ACC_WIDTH +: ACC_WIDTH];
                assign vv_in_w = vv_mesh[(c*(ARR_R+1)+r)];

                assign w_local_w       = w_local_bus[(r*ARR_C+c)*WORD_SIZE +: WORD_SIZE];
                assign w_local_valid_w = w_local_valid_bus[(r*ARR_C+c)];

                PE_reconfig_v2 #(
                    .WORD_SIZE(WORD_SIZE),
                    .ACC_WIDTH(ACC_WIDTH)
                ) u_pe (
                    .clk           (clk),
                    .rst_n         (rst_n),
                    .en            (en),
                    .mode          (mode),
                    .psum_clr      (psum_clr),

                    .a_in          (a_in_w),
                    .a_valid_in    (av_in_w),
                    .a_out         (a_out_w),
                    .a_valid_out   (av_out_w),

                    .vert_in       (v_in_w),
                    .vert_valid_in (vv_in_w),
                    .vert_out      (v_out_w),
                    .vert_valid_out(vv_out_w),

                    .w_local_in    (w_local_w),
                    .w_local_valid (w_local_valid_w),

                    .psum_snapshot (psum_snap_w)
                );

                assign a_mesh[(r*(ARR_C+1)+c+1)*WORD_SIZE +: WORD_SIZE] = a_out_w;
                assign av_mesh[(r*(ARR_C+1)+c+1)] = av_out_w;

                assign v_mesh[(c*(ARR_R+1)+r+1)*ACC_WIDTH +: ACC_WIDTH] = v_out_w;
                assign vv_mesh[(c*(ARR_R+1)+r+1)] = vv_out_w;

                assign psum_snapshot_bus[(r*ARR_C+c)*ACC_WIDTH +: ACC_WIDTH] = psum_snap_w;
            end
        end

        for (c = 0; c < ARR_C; c = c + 1) begin : OUT_BOUNDARY
            assign vert_bottom_bus[c*ACC_WIDTH +: ACC_WIDTH] =
                v_mesh[(c*(ARR_R+1)+ARR_R)*ACC_WIDTH +: ACC_WIDTH];
            assign vert_bottom_valid_bus[c] =
                vv_mesh[(c*(ARR_R+1)+ARR_R)];
        end
    endgenerate

endmodule