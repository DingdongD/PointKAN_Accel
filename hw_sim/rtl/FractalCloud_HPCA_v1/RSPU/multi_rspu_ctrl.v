// ============================================================
// Multi-RSPU Controller
// Implements Algorithm 2 from FractalCloud:
//   - FPS mode:    Inter-block parallelism (each RSPU handles one block)
//   - NS  mode:    Intra-block parallelism (multiple RSPUs share search space,
//                  each handles different center points within same block)
//
// N_RSPU: number of parallel RSPU instances
// Ports use packed vectors (no array ports per Verilog standard).
// ============================================================

`timescale 1ns/1ps

module multi_rspu_ctrl #(
    parameter N_RSPU    = 4,
    parameter COORD_W   = 16,
    parameter DIST_W    = 32,
    parameter MAX_PTS   = 256,
    parameter MAX_SAMPLE= 64,
    parameter K         = 16,
    parameter RADIUS_W  = 32,
    parameter IDX_W     = 8,
    parameter N_BLOCKS  = 4    // number of parallel blocks (FPS mode)
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Global control
    input  wire [1:0]            mode,         // 00=FPS, 01=BQ, 10=KNN
    input  wire                  start,
    output reg                   all_done,

    // Per-RSPU configuration (packed: [0] = RSPU0, [1] = RSPU1, ...)
    input  wire [N_RSPU*(IDX_W+1)-1:0]     rspu_num_pts,
    input  wire [N_RSPU*(IDX_W+1)-1:0]     rspu_num_out,
    input  wire [RADIUS_W-1:0]             radius_sq,

    // Coordinate memory per RSPU (packed)
    output wire [N_RSPU*IDX_W-1:0]         coord_rd_addr,
    input  wire [N_RSPU*COORD_W*3-1:0]     coord_rd_data,

    output wire [N_RSPU*IDX_W-1:0]         center_rd_addr,
    input  wire [N_RSPU*COORD_W*3-1:0]     center_rd_data,

    // Aggregated FPS outputs (packed)
    output wire [N_RSPU-1:0]                fps_out_valid,
    output wire [N_RSPU*IDX_W-1:0]          fps_out_idx,

    // Aggregated NS outputs (packed)
    output wire [N_RSPU-1:0]                ns_result_valid,
    output wire [N_RSPU*IDX_W-1:0]         ns_center_idx,
    output wire [N_RSPU*IDX_W-1:0]         ns_nbr_idx,
    output wire [N_RSPU*DIST_W-1:0]        ns_dist_sq
);

    // Per-RSPU done signals (internal array is allowed in Verilog-2001)
    wire [N_RSPU-1:0] done_vec;
    reg  start_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            all_done  <= 1'b0;
            start_reg <= 1'b0;
        end else begin
            start_reg <= start;
            all_done  <= &done_vec;
        end
    end

    genvar g;
    generate
        for (g = 0; g < N_RSPU; g = g + 1) begin : rspu_inst
            wire [IDX_W-1:0]      coord_rd_addr_g;
            wire [IDX_W-1:0]      center_rd_addr_g;
            wire                  fps_out_valid_g;
            wire [IDX_W-1:0]      fps_out_idx_g;
            wire                  ns_result_valid_g;
            wire [IDX_W-1:0]      ns_center_idx_g;
            wire [IDX_W-1:0]      ns_nbr_idx_g;
            wire [DIST_W-1:0]     ns_dist_sq_g;

            assign coord_rd_addr  [g*IDX_W +: IDX_W]   = coord_rd_addr_g;
            assign center_rd_addr [g*IDX_W +: IDX_W]   = center_rd_addr_g;
            assign fps_out_valid  [g]                  = fps_out_valid_g;
            assign fps_out_idx    [g*IDX_W +: IDX_W]   = fps_out_idx_g;
            assign ns_result_valid[g]                  = ns_result_valid_g;
            assign ns_center_idx  [g*IDX_W +: IDX_W]   = ns_center_idx_g;
            assign ns_nbr_idx     [g*IDX_W +: IDX_W]   = ns_nbr_idx_g;
            assign ns_dist_sq     [g*DIST_W +: DIST_W] = ns_dist_sq_g;

            rspu_top #(
                .COORD_W   (COORD_W),
                .DIST_W    (DIST_W),
                .MAX_PTS   (MAX_PTS),
                .MAX_SAMPLE(MAX_SAMPLE),
                .K         (K),
                .RADIUS_W  (RADIUS_W),
                .IDX_W     (IDX_W)
            ) u_rspu (
                .clk                (clk),
                .rst_n              (rst_n),
                .mode               (mode),
                .start              (start_reg),
                .done               (done_vec[g]),
                .num_pts            (rspu_num_pts[g*(IDX_W+1) +: (IDX_W+1)]),
                .num_out            (rspu_num_out[g*(IDX_W+1) +: (IDX_W+1)]),
                .radius_sq          (radius_sq),
                .coord_rd_addr      (coord_rd_addr_g),
                .coord_rd_data     (coord_rd_data[g*COORD_W*3 +: COORD_W*3]),
                .center_rd_addr    (center_rd_addr_g),
                .center_rd_data    (center_rd_data[g*COORD_W*3 +: COORD_W*3]),
                .result_valid      (ns_result_valid_g),
                .result_center_idx (ns_center_idx_g),
                .result_neighbor_idx(ns_nbr_idx_g),
                .result_dist_sq    (ns_dist_sq_g),
                .fps_out_valid     (fps_out_valid_g),
                .fps_out_idx       (fps_out_idx_g),
                .dbg_sample_count  ()
            );
        end
    endgenerate

endmodule
