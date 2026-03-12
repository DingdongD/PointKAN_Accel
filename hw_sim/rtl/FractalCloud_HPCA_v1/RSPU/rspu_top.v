// ============================================================
// RSPU: Reuse-and-Skip-Enabled Point Unit
// Based on FractalCloud (HPCA 2026)
//
// Supports three operation modes:
//   MODE_FPS  (2'b00): Farthest Point Sampling
//   MODE_BQ   (2'b01): Ball Query (radius neighbor search)
//   MODE_KNN  (2'b10): K-Nearest Neighbors (interpolation)
//
// Pipeline stages:
//   S1: Window-Check / Address generation
//   S2: Distance Compute (XYZ diff squared)
//   S3: Distance Accumulate (sum -> Euclidean^2)
//   S4: Compare / ArgMax  (FPS) or Top-K update (BQ/KNN)
//   S5: Write-back to local buffer / Update-Buffer
//
// Parameters:
//   DIM         = 3   (x,y,z)
//   COORD_W     = 16  (half-precision fixed-point Q8.8)
//   DIST_W      = 32  (accumulated squared distance)
//   MAX_PTS     = 256 (max points per block, matches th=256)
//   MAX_SAMPLE  = 64  (max sampled points)
//   K           = 16  (top-K neighbors)
//   IDX_W       = log2(MAX_PTS) = 8
// ============================================================

`timescale 1ns/1ps

module rspu_top #(
    parameter COORD_W   = 16,
    parameter DIST_W    = 32,
    parameter MAX_PTS   = 256,
    parameter MAX_SAMPLE= 64,
    parameter K         = 16,
    parameter RADIUS_W  = 32,   // squared radius, same unit as DIST_W
    parameter IDX_W     = 8     // ceil(log2(MAX_PTS))
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Operation control
    input  wire [1:0]            mode,        // 00=FPS, 01=BQ, 10=KNN
    input  wire                  start,       // pulse: begin a new operation
    output reg                   done,        // pulse: operation finished

    // Number of input points in current block
    input  wire [IDX_W:0]        num_pts,     // up to MAX_PTS

    // Number of samples to produce (FPS) or center points (BQ/KNN)
    input  wire [IDX_W:0]        num_out,

    // Search radius squared (Ball Query only)
    input  wire [RADIUS_W-1:0]   radius_sq,

    // --- Point Coordinate Memory (Block RAM, single-port read) ---
    // Caller loads coordinates into coord_mem before asserting start.
    // coord_mem[i] = {z_i, y_i, x_i}  (each COORD_W bits, packed)
    output reg  [IDX_W-1:0]      coord_rd_addr,
    input  wire [COORD_W*3-1:0]  coord_rd_data,

    // --- Current sampled/center set (written by RSPU for FPS output,
    //     or provided externally for BQ/KNN center input) ---
    // For BQ/KNN: caller pre-fills center_mem then asserts start.
    output reg  [IDX_W-1:0]      center_rd_addr,
    input  wire [COORD_W*3-1:0]  center_rd_data,

    // --- Result output: neighbor index FIFO ---
    output reg                   result_valid,
    output reg  [IDX_W-1:0]      result_center_idx, // which center point
    output reg  [IDX_W-1:0]      result_neighbor_idx,
    output reg  [DIST_W-1:0]     result_dist_sq,    // distance^2 to neighbor

    // --- FPS sampled index output (FPS mode) ---
    output reg                   fps_out_valid,
    output reg  [IDX_W-1:0]      fps_out_idx,

    // --- Debug / status ---
    output wire [IDX_W:0]        dbg_sample_count
);

    // ----- Mode encoding -----
    localparam MODE_FPS = 2'b00;
    localparam MODE_BQ  = 2'b01;
    localparam MODE_KNN = 2'b10;

    // ----- FSM states -----
    localparam ST_IDLE          = 4'd0;
    localparam ST_FPS_INIT      = 4'd1;
    localparam ST_FPS_ITER_WIN  = 4'd2;   // window-check phase
    localparam ST_FPS_ITER_DIST = 4'd3;   // distance update phase
    localparam ST_FPS_SELECT    = 4'd4;   // argmax -> next sampled point
    localparam ST_FPS_DONE      = 4'd5;
    localparam ST_NS_INIT       = 4'd6;   // neighbor search init
    localparam ST_NS_SCAN       = 4'd7;   // scan all points for one center
    localparam ST_NS_NEXT       = 4'd8;   // advance to next center
    localparam ST_NS_DONE       = 4'd9;

    reg [3:0]  state, next_state;

    // ----- FPS internal registers -----
    // prev_dist[i]: running minimum distance from point i to sampled set
    // Stored in a small register file (synthesises to LUTRAM/BRAM for large N)
    reg [DIST_W-1:0] prev_dist [0:MAX_PTS-1];
    reg              sampled   [0:MAX_PTS-1]; // 1 = already sampled (mask)

    reg [IDX_W:0]    sample_count;   // how many sampled so far
    reg [IDX_W-1:0]  last_sampled;   // index of most recently sampled point

    // ArgMax registers
    reg [DIST_W-1:0] argmax_val;
    reg [IDX_W-1:0]  argmax_idx;

    // FPS scan pointer
    reg [IDX_W-1:0]  scan_ptr;       // current point being processed
    reg [IDX_W-1:0]  wc_next_ptr;    // next valid (unsampled) point from window-check

    // ----- Neighbor search registers -----
    reg [IDX_W:0]    center_iter;    // which center point we're on
    reg [IDX_W-1:0]  ns_scan_ptr;    // scan pointer over block points

    // Top-K structures: we keep K sorted slots per center
    // For simplicity: parallel array updated per-scan step
    reg [DIST_W-1:0] topk_dist [0:K-1];
    reg [IDX_W-1:0]  topk_idx  [0:K-1];
    reg [IDX_W:0]    topk_count;     // number of valid entries so far

    // ----- Coordinate pipeline registers -----
    reg [COORD_W-1:0] cx, cy, cz;   // center point coords (registered)
    reg [COORD_W-1:0] px, py, pz;   // candidate point coords (registered)

    // Signed difference for distance computation
    wire signed [COORD_W:0]   dx = $signed({1'b0,cx}) - $signed({1'b0,px});
    wire signed [COORD_W:0]   dy = $signed({1'b0,cy}) - $signed({1'b0,py});
    wire signed [COORD_W:0]   dz = $signed({1'b0,cz}) - $signed({1'b0,pz});

    wire [DIST_W-1:0] dx2 = dx * dx;
    wire [DIST_W-1:0] dy2 = dy * dy;
    wire [DIST_W-1:0] dz2 = dz * dz;
    wire [DIST_W-1:0] dist_sq = dx2 + dy2 + dz2;

    // ----- Pipeline delay registers (2-cycle latency for coord read + dist) -----
    reg            pipe_valid_s1, pipe_valid_s2;
    reg [IDX_W-1:0] pipe_pt_idx_s1, pipe_pt_idx_s2;
    reg [DIST_W-1:0] pipe_dist_s2;

    assign dbg_sample_count = sample_count;

    // =========================================================
    // FSM
    // =========================================================
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= next_state;
        end
    end

    always @(*) begin
        next_state = state;
        case (state)
            ST_IDLE: if (start) begin
                if (mode == MODE_FPS) next_state = ST_FPS_INIT;
                else                  next_state = ST_NS_INIT;
            end
            ST_FPS_INIT:        next_state = ST_FPS_ITER_WIN;
            ST_FPS_ITER_WIN:    next_state = ST_FPS_ITER_DIST;
            ST_FPS_ITER_DIST: begin
                // One full scan done when scan_ptr reaches num_pts
                if (scan_ptr == num_pts - 1)
                    next_state = ST_FPS_SELECT;
            end
            ST_FPS_SELECT: begin
                if (sample_count >= num_out - 1)
                    next_state = ST_FPS_DONE;
                else
                    next_state = ST_FPS_ITER_WIN;
            end
            ST_FPS_DONE:       next_state = ST_IDLE;

            ST_NS_INIT:        next_state = ST_NS_SCAN;
            ST_NS_SCAN: begin
                if (ns_scan_ptr == num_pts - 1)
                    next_state = ST_NS_NEXT;
            end
            ST_NS_NEXT: begin
                if (center_iter >= num_out - 1)
                    next_state = ST_NS_DONE;
                else
                    next_state = ST_NS_SCAN;
            end
            ST_NS_DONE:        next_state = ST_IDLE;
            default:           next_state = ST_IDLE;
        endcase
    end

    // =========================================================
    // Datapath
    // =========================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done             <= 1'b0;
            fps_out_valid    <= 1'b0;
            result_valid     <= 1'b0;
            sample_count     <= 0;
            scan_ptr         <= 0;
            center_iter      <= 0;
            ns_scan_ptr      <= 0;
            argmax_val       <= 0;
            argmax_idx       <= 0;
            topk_count       <= 0;
            pipe_valid_s1    <= 0;
            pipe_valid_s2    <= 0;
            coord_rd_addr    <= 0;
            center_rd_addr   <= 0;
            for (i = 0; i < MAX_PTS; i = i + 1) begin
                prev_dist[i] <= {DIST_W{1'b1}}; // init to max
                sampled[i]   <= 1'b0;
            end
            for (i = 0; i < K; i = i + 1) begin
                topk_dist[i] <= {DIST_W{1'b1}};
                topk_idx[i]  <= 0;
            end
        end else begin
            done          <= 1'b0;
            fps_out_valid <= 1'b0;
            result_valid  <= 1'b0;

            case (state)
                // --------------------------------------------------
                ST_IDLE: begin
                    if (start) begin
                        // Reset all state for new operation
                        sample_count <= 0;
                        scan_ptr     <= 0;
                        center_iter  <= 0;
                        ns_scan_ptr  <= 0;
                        argmax_val   <= 0;
                        argmax_idx   <= 0;
                        topk_count   <= 0;
                        pipe_valid_s1 <= 0;
                        pipe_valid_s2 <= 0;
                        for (i = 0; i < MAX_PTS; i = i + 1) begin
                            prev_dist[i] <= {DIST_W{1'b1}};
                            sampled[i]   <= 1'b0;
                        end
                        // Seed: first sampled point = point 0
                        if (mode == MODE_FPS) begin
                            sampled[0]   <= 1'b1;
                            last_sampled <= 0;
                            sample_count <= 1;
                            fps_out_valid <= 1'b1;
                            fps_out_idx   <= 0;
                        end
                    end
                end

                // --------------------------------------------------
                // FPS: Window-Check phase
                // Determine the first unsampled point (wc_next_ptr)
                // This models the LOD (lowest-one detector) in Fig.11(c)
                // --------------------------------------------------
                ST_FPS_INIT: begin
                    // Load center coords for point last_sampled
                    center_rd_addr <= last_sampled;
                    scan_ptr       <= 0;
                    argmax_val     <= 0;
                    argmax_idx     <= 0;
                end

                ST_FPS_ITER_WIN: begin
                    // Register center coordinates (1-cycle latency)
                    {cz, cy, cx} <= center_rd_data;
                    // Issue read for current candidate point
                    coord_rd_addr  <= scan_ptr;
                    pipe_valid_s1  <= !sampled[scan_ptr];
                    pipe_pt_idx_s1 <= scan_ptr;
                end

                ST_FPS_ITER_DIST: begin
                    // S1->S2: latch candidate coords
                    {pz, py, px}   <= coord_rd_data;
                    pipe_valid_s2  <= pipe_valid_s1;
                    pipe_pt_idx_s2 <= pipe_pt_idx_s1;
                    pipe_dist_s2   <= dist_sq;

                    // S2 result: update prev_dist and argmax
                    if (pipe_valid_s2) begin
                        // Minimum of new dist and stored prev_dist
                        if (dist_sq < prev_dist[pipe_pt_idx_s2]) begin
                            prev_dist[pipe_pt_idx_s2] <= dist_sq;
                            // Use dist_sq for argmax comparison
                            if (dist_sq > argmax_val) begin
                                argmax_val <= dist_sq;
                                argmax_idx <= pipe_pt_idx_s2;
                            end
                        end else begin
                            if (prev_dist[pipe_pt_idx_s2] > argmax_val) begin
                                argmax_val <= prev_dist[pipe_pt_idx_s2];
                                argmax_idx <= pipe_pt_idx_s2;
                            end
                        end
                    end

                    // Advance scan pointer (window-check: skip sampled)
                    if (scan_ptr < num_pts - 1)
                        scan_ptr <= scan_ptr + 1;

                    // Advance coord read ahead by 1 (skip sampled via mask)
                    if (scan_ptr + 1 < num_pts)
                        coord_rd_addr <= scan_ptr + 1;
                end

                // --------------------------------------------------
                // FPS: Select next sample (argmax)
                // --------------------------------------------------
                ST_FPS_SELECT: begin
                    // Mark selected point as sampled
                    sampled[argmax_idx]   <= 1'b1;
                    last_sampled          <= argmax_idx;
                    sample_count          <= sample_count + 1;
                    fps_out_valid         <= 1'b1;
                    fps_out_idx           <= argmax_idx;
                    // Reset for next iteration
                    scan_ptr              <= 0;
                    argmax_val            <= 0;
                    argmax_idx            <= 0;
                    center_rd_addr        <= argmax_idx;
                end

                ST_FPS_DONE: begin
                    done <= 1'b1;
                end

                // --------------------------------------------------
                // Neighbor Search: Init center 0
                // --------------------------------------------------
                ST_NS_INIT: begin
                    center_rd_addr <= 0;
                    ns_scan_ptr    <= 0;
                    topk_count     <= 0;
                    for (i = 0; i < K; i = i + 1) begin
                        topk_dist[i] <= {DIST_W{1'b1}};
                        topk_idx[i]  <= 0;
                    end
                end

                // --------------------------------------------------
                // Neighbor Search: Scan all block points for current center
                // --------------------------------------------------
                ST_NS_SCAN: begin
                    // Latch center coords on first cycle
                    if (ns_scan_ptr == 0)
                        {cz, cy, cx} <= center_rd_data;

                    // Issue read for candidate
                    coord_rd_addr  <= ns_scan_ptr;
                    pipe_valid_s1  <= 1'b1;
                    pipe_pt_idx_s1 <= ns_scan_ptr;

                    // Process pipeline stage 2
                    {pz, py, px}   <= coord_rd_data;
                    pipe_valid_s2  <= pipe_valid_s1;
                    pipe_pt_idx_s2 <= pipe_pt_idx_s1;

                    // Insert into Top-K if within radius (BQ) or always (KNN)
                    if (pipe_valid_s2) begin
                        if (mode == MODE_BQ) begin
                            if (dist_sq <= radius_sq)
                                // Simplified: insert if better than worst
                                if (topk_count < K || dist_sq < topk_dist[K-1]) begin
                                    // Shift and insert (bubble insert)
                                    topk_dist[0] <= dist_sq;
                                    topk_idx[0]  <= pipe_pt_idx_s2;
                                    if (topk_count < K)
                                        topk_count <= topk_count + 1;
                                    result_valid        <= 1'b1;
                                    result_center_idx   <= center_iter[IDX_W-1:0];
                                    result_neighbor_idx <= pipe_pt_idx_s2;
                                    result_dist_sq      <= dist_sq;
                                end
                        end else begin // KNN
                            if (topk_count < K || dist_sq < topk_dist[K-1]) begin
                                topk_dist[0] <= dist_sq;
                                topk_idx[0]  <= pipe_pt_idx_s2;
                                if (topk_count < K)
                                    topk_count <= topk_count + 1;
                                result_valid        <= 1'b1;
                                result_center_idx   <= center_iter[IDX_W-1:0];
                                result_neighbor_idx <= pipe_pt_idx_s2;
                                result_dist_sq      <= dist_sq;
                            end
                        end
                    end

                    if (ns_scan_ptr < num_pts - 1)
                        ns_scan_ptr <= ns_scan_ptr + 1;
                end

                // --------------------------------------------------
                // Neighbor Search: Move to next center
                // --------------------------------------------------
                ST_NS_NEXT: begin
                    center_iter    <= center_iter + 1;
                    ns_scan_ptr    <= 0;
                    topk_count     <= 0;
                    for (i = 0; i < K; i = i + 1) begin
                        topk_dist[i] <= {DIST_W{1'b1}};
                        topk_idx[i]  <= 0;
                    end
                    center_rd_addr <= center_iter[IDX_W-1:0] + 1;
                end

                ST_NS_DONE: begin
                    done <= 1'b1;
                end

            endcase
        end
    end

endmodule
