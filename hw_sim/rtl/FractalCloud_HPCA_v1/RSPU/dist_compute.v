// ============================================================
// Distance Compute Unit
// Computes squared Euclidean distance between two 3D points.
// Coordinate format: Q8.8 fixed-point (COORD_W=16 bits)
// Output: 32-bit squared distance (same fixed-point scale^2)
//
// Pipeline: 2 stages
//   Stage 1: compute dx,dy,dz
//   Stage 2: accumulate dx^2+dy^2+dz^2
// ============================================================

`timescale 1ns/1ps

module dist_compute #(
    parameter COORD_W = 16,
    parameter DIST_W  = 32
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Input valid signal (flows through pipeline)
    input  wire                  valid_in,
    output reg                   valid_out,

    // Point A (center)
    input  wire [COORD_W-1:0]    ax, ay, az,

    // Point B (candidate)
    input  wire [COORD_W-1:0]    bx, by, bz,

    // Result
    output reg  [DIST_W-1:0]     dist_sq_out
);

    // Stage 1: difference
    reg signed [COORD_W:0] dx_s1, dy_s1, dz_s1;
    reg                    valid_s1;

    // Stage 2: square & accumulate
    wire [DIST_W-1:0] dx2 = dx_s1 * dx_s1;
    wire [DIST_W-1:0] dy2 = dy_s1 * dy_s1;
    wire [DIST_W-1:0] dz2 = dz_s1 * dz_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dx_s1       <= 0;
            dy_s1       <= 0;
            dz_s1       <= 0;
            valid_s1    <= 0;
            dist_sq_out <= 0;
            valid_out   <= 0;
        end else begin
            // Stage 1
            dx_s1    <= $signed({1'b0, ax}) - $signed({1'b0, bx});
            dy_s1    <= $signed({1'b0, ay}) - $signed({1'b0, by});
            dz_s1    <= $signed({1'b0, az}) - $signed({1'b0, bz});
            valid_s1 <= valid_in;

            // Stage 2
            dist_sq_out <= dx2 + dy2 + dz2;
            valid_out   <= valid_s1;
        end
    end

endmodule
