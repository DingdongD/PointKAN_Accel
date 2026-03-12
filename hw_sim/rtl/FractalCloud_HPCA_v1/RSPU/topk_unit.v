// ============================================================
// Top-K Unit
// Based on merge-sort mechanism described in FractalCloud.
// Maintains a sorted list of K smallest distances.
//
// On each cycle, if insert_en is asserted:
//   - Compare new_dist against worst (largest) in top-K list
//   - If better, replace and re-sort (bubble-down)
//
// Outputs: packed vectors topk_dist, topk_idx (index 0 = smallest)
// count: number of valid entries (saturates at K)
// ============================================================

`timescale 1ns/1ps

module topk_unit #(
    parameter K      = 16,
    parameter DIST_W = 32,
    parameter IDX_W  = 8
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Control
    input  wire                  reset_topk,  // clear all entries

    // Insert interface
    input  wire                  insert_en,
    input  wire [DIST_W-1:0]     insert_dist,
    input  wire [IDX_W-1:0]      insert_idx,

    // BQ: radius gate
    input  wire                  use_radius,
    input  wire [DIST_W-1:0]     radius_sq,

    // Output sorted list, packed (index 0 = smallest dist)
    // topk_dist = { elem[K-1], ..., elem[0] }, each elem DIST_W bits
    output reg  [K*DIST_W-1:0]   topk_dist,
    output reg  [K*IDX_W-1:0]    topk_idx,
    output reg  [4:0]            count      // valid entries, max K
);

    integer i;
    integer k;
    reg [DIST_W-1:0] new_dist_reg;
    reg [IDX_W-1:0]  new_idx_reg;
    reg              do_insert;

    // Internal arrays (Verilog standard: no array ports)
    reg [DIST_W-1:0] topk_dist_arr [0:K-1];
    reg [IDX_W-1:0]  topk_idx_arr  [0:K-1];

    // Block-level vars moved to module level (no reg in unnamed block)
    reg [DIST_W-1:0] tmp_dist;
    reg [IDX_W-1:0]  tmp_idx;
    reg              inserted;
    reg [DIST_W-1:0] swap_d;
    reg [IDX_W-1:0]  swap_i;

    // Pack internal arrays to output
    always @(*) begin
        for (i = 0; i < K; i = i + 1) begin
            topk_dist[i*DIST_W +: DIST_W] = topk_dist_arr[i];
            topk_idx[i*IDX_W +: IDX_W]   = topk_idx_arr[i];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 0;
            for (i = 0; i < K; i = i + 1) begin
                topk_dist_arr[i] <= {DIST_W{1'b1}};
                topk_idx_arr[i]  <= 0;
            end
        end else if (reset_topk) begin
            count <= 0;
            for (i = 0; i < K; i = i + 1) begin
                topk_dist_arr[i] <= {DIST_W{1'b1}};
                topk_idx_arr[i]  <= 0;
            end
        end else if (insert_en) begin
            // Gate on radius for Ball Query
            if (!use_radius || insert_dist <= radius_sq) begin
                // Only insert if better than current worst or list not full
                if (count < K || insert_dist < topk_dist_arr[K-1]) begin
                    // Shift-insert: find insertion position
                    tmp_dist  = insert_dist;
                    tmp_idx   = insert_idx;
                    inserted  = 1'b0;

                    for (k = 0; k < K; k = k + 1) begin
                        if (!inserted && tmp_dist < topk_dist_arr[k]) begin
                            swap_d = topk_dist_arr[k];
                            swap_i = topk_idx_arr[k];
                            topk_dist_arr[k] <= tmp_dist;
                            topk_idx_arr[k]  <= tmp_idx;
                            tmp_dist = swap_d;
                            tmp_idx  = swap_i;
                            inserted  = 1'b1;
                        end else if (inserted) begin
                            swap_d = topk_dist_arr[k];
                            swap_i = topk_idx_arr[k];
                            topk_dist_arr[k] <= tmp_dist;
                            topk_idx_arr[k]  <= tmp_idx;
                            tmp_dist = swap_d;
                            tmp_idx  = swap_i;
                        end
                    end

                    if (count < K)
                        count <= count + 1;
                end
            end
        end
    end

endmodule
