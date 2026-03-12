// ============================================================
// Window Check Module with LOD (Lowest-One Detector)
// From FractalCloud Fig.11(c)
//
// Maintains a bitmask of MAX_PTS bits.
// Each bit = 1 means the point is still valid (not yet sampled).
// LOD finds the lowest set bit = next valid point address.
//
// Operations:
//   - clear_bit(idx)  : mark point idx as sampled
//   - query()         : returns index of next valid point
//   - reset()         : set all bits to 1 (all points valid)
// ============================================================

`timescale 1ns/1ps

module window_check #(
    parameter MAX_PTS = 256,
    parameter IDX_W   = 8
)(
    input  wire              clk,
    input  wire              rst_n,

    // Control
    input  wire              reset_mask,   // synchronous reset: all 1s
    input  wire              clear_en,     // clear bit at clear_idx
    input  wire [IDX_W-1:0]  clear_idx,
    input  wire [IDX_W-1:0]  num_pts,      // how many points are valid

    // LOD query: given a start address, find next '1'
    input  wire [IDX_W-1:0]  lod_start,
    output reg  [IDX_W-1:0]  lod_result,   // index of next valid point
    output reg               lod_valid,    // 1 if any valid point found

    // Direct mask read (for pipeline integration)
    output wire [MAX_PTS-1:0] mask_out
);

    reg [MAX_PTS-1:0] mask;
    assign mask_out = mask;

    integer i;

    // ---- Mask update ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mask <= {MAX_PTS{1'b1}};
        end else if (reset_mask) begin
            mask <= {MAX_PTS{1'b1}};
        end else if (clear_en) begin
            mask[clear_idx] <= 1'b0;
        end
    end

    // ---- LOD: combinational priority encoder ----
    // Searches from lod_start forward (wraps not needed for block-wise FPS)
    integer j;
    always @(*) begin
        lod_result = {IDX_W{1'b1}};
        lod_valid  = 1'b0;
        for (j = MAX_PTS-1; j >= 0; j = j - 1) begin
            if (j >= lod_start && j < num_pts && mask[j]) begin
                lod_result = j[IDX_W-1:0];
                lod_valid  = 1'b1;
            end
        end
    end

endmodule
