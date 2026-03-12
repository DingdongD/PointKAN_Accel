// ============================================================
// Testbench for rspu_top
// Tests:
//   1. FPS mode: 16 points, select 4 samples
//   2. Ball Query mode: 16 points, radius search
//   3. KNN mode: 16 points, K=4 nearest neighbors
//
// Coordinates: Q8.8 fixed point
//   integer value N -> 16'h(N<<8)
// ============================================================

`timescale 1ns/1ps

`include "rspu_top.v"
`include "topk_unit.v"
`include "window_check.v"
`include "dist_compute.v"



module tb_rspu;

    // ---- Parameters ----
    localparam COORD_W    = 16;
    localparam DIST_W     = 32;
    localparam MAX_PTS    = 32;   // smaller for TB
    localparam MAX_SAMPLE = 8;
    localparam K          = 4;
    localparam RADIUS_W   = 32;
    localparam IDX_W      = 5;   // log2(32)

    // ---- DUT signals ----
    reg                   clk;
    reg                   rst_n;
    reg  [1:0]            mode;
    reg                   start;
    wire                  done;
    reg  [IDX_W:0]        num_pts;
    reg  [IDX_W:0]        num_out;
    reg  [RADIUS_W-1:0]   radius_sq;
    wire [IDX_W-1:0]      coord_rd_addr;
    reg  [COORD_W*3-1:0]  coord_rd_data;
    wire [IDX_W-1:0]      center_rd_addr;
    reg  [COORD_W*3-1:0]  center_rd_data;
    wire                  result_valid;
    wire [IDX_W-1:0]      result_center_idx;
    wire [IDX_W-1:0]      result_neighbor_idx;
    wire [DIST_W-1:0]     result_dist_sq;
    wire                  fps_out_valid;
    wire [IDX_W-1:0]      fps_out_idx;
    wire [IDX_W:0]        dbg_sample_count;

    // ---- Memory arrays ----
    reg [COORD_W*3-1:0]  coord_mem  [0:MAX_PTS-1];
    reg [COORD_W*3-1:0]  center_mem [0:MAX_PTS-1];

    // ---- Memory read ----
    always @(*) begin
        coord_rd_data  = coord_mem [coord_rd_addr];
        center_rd_data = center_mem[center_rd_addr];
    end

    // ---- DUT instantiation ----
    rspu_top #(
        .COORD_W   (COORD_W),
        .DIST_W    (DIST_W),
        .MAX_PTS   (MAX_SAMPLE),  // use MAX_SAMPLE as internal array size for TB
        .MAX_SAMPLE(MAX_SAMPLE),
        .K         (K),
        .RADIUS_W  (RADIUS_W),
        .IDX_W     (IDX_W)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .mode               (mode),
        .start              (start),
        .done               (done),
        .num_pts            (num_pts),
        .num_out            (num_out),
        .radius_sq          (radius_sq),
        .coord_rd_addr      (coord_rd_addr),
        .coord_rd_data      (coord_rd_data),
        .center_rd_addr     (center_rd_addr),
        .center_rd_data     (center_rd_data),
        .result_valid       (result_valid),
        .result_center_idx  (result_center_idx),
        .result_neighbor_idx(result_neighbor_idx),
        .result_dist_sq     (result_dist_sq),
        .fps_out_valid      (fps_out_valid),
        .fps_out_idx        (fps_out_idx),
        .dbg_sample_count   (dbg_sample_count)
    );

    // ---- Clock ----
    initial clk = 0;
    always #5 clk = ~clk; // 100 MHz

    // ---- Helper: pack coordinates ----
    function [COORD_W*3-1:0] pack_coord;
        input [7:0] x, y, z; // integer values (will be shifted to Q8.8)
        begin
            pack_coord = {
                {z[7:0], 8'h00},  // z in Q8.8
                {y[7:0], 8'h00},  // y in Q8.8
                {x[7:0], 8'h00}   // x in Q8.8
            };
        end
    endfunction

    // ---- Task: apply reset ----
    task do_reset;
        begin
            rst_n = 0;
            start = 0;
            repeat(4) @(posedge clk);
            @(negedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask

    // ---- Task: run until done (with timeout) ----
    integer timeout;
    task run_until_done;
        begin
            @(posedge clk); #1;
            start = 1;
            @(posedge clk); #1;
            start = 0;
            timeout = 0;
            while (!done && timeout < 10000) begin
                @(posedge clk); #1;
                timeout = timeout + 1;
            end
            if (timeout >= 10000)
                $display("[ERROR] Timeout!");
        end
    endtask

    // ---- Recorded outputs ----
    integer fps_results [0:MAX_SAMPLE-1];
    integer fps_count;
    integer ns_center  [0:MAX_PTS*K-1];
    integer ns_nbr     [0:MAX_PTS*K-1];
    integer ns_dist    [0:MAX_PTS*K-1];
    integer ns_count;

    // ---- Monitor: record outputs ----
    always @(posedge clk) begin
        if (fps_out_valid) begin
            $display("  [FPS] sample[%0d] = point %0d", fps_count, fps_out_idx);
            fps_results[fps_count] = fps_out_idx;
            fps_count = fps_count + 1;
        end
        if (result_valid) begin
            $display("  [NS ] center=%0d neighbor=%0d dist_sq=%0d",
                     result_center_idx, result_neighbor_idx, result_dist_sq);
            ns_center[ns_count] = result_center_idx;
            ns_nbr   [ns_count] = result_neighbor_idx;
            ns_dist  [ns_count] = result_dist_sq;
            ns_count = ns_count + 1;
        end
    end

    // ============================================================
    // Main test sequence
    // ============================================================
    integer i;

    initial begin
        $dumpfile("rspu_sim.vcd");
        $dumpvars(0, tb_rspu);

        // --------------------------------------------------------
        // Load 16 test points:
        // Layout:  four clusters of 4 points
        //   cluster A: (1,1,0),(2,1,0),(1,2,0),(2,2,0)
        //   cluster B: (8,1,0),(9,1,0),(8,2,0),(9,2,0)
        //   cluster C: (1,8,0),(2,8,0),(1,9,0),(2,9,0)
        //   cluster D: (8,8,0),(9,8,0),(8,9,0),(9,9,0)
        // --------------------------------------------------------
        coord_mem[0]  = pack_coord(1,1,0);
        coord_mem[1]  = pack_coord(2,1,0);
        coord_mem[2]  = pack_coord(1,2,0);
        coord_mem[3]  = pack_coord(2,2,0);
        coord_mem[4]  = pack_coord(8,1,0);
        coord_mem[5]  = pack_coord(9,1,0);
        coord_mem[6]  = pack_coord(8,2,0);
        coord_mem[7]  = pack_coord(9,2,0);
        coord_mem[8]  = pack_coord(1,8,0);
        coord_mem[9]  = pack_coord(2,8,0);
        coord_mem[10] = pack_coord(1,9,0);
        coord_mem[11] = pack_coord(2,9,0);
        coord_mem[12] = pack_coord(8,8,0);
        coord_mem[13] = pack_coord(9,8,0);
        coord_mem[14] = pack_coord(8,9,0);
        coord_mem[15] = pack_coord(9,9,0);

        // --------------------------------------------------------
        // TEST 1: FPS - select 4 samples from 16 points
        // Expected: one point from each cluster
        // --------------------------------------------------------
        $display("=== TEST 1: FPS (16 pts -> 4 samples) ===");
        fps_count = 0; ns_count = 0;
        do_reset();
        mode    = 2'b00; // FPS
        num_pts = 16;
        num_out = 4;
        run_until_done();
        $display("  FPS complete. %0d samples collected.", fps_count);

        // --------------------------------------------------------
        // TEST 2: Ball Query from center (5,5,0), radius^2 = 25*(256*256)
        // Q8.8: 5 -> 5<<8 = 1280 ; 25 in Q8.8^2: 25*(256^2)
        // --------------------------------------------------------
        $display("=== TEST 2: Ball Query (center (5,5,0), r^2=25) ===");
        fps_count = 0; ns_count = 0;
        // Set center_mem[0] = (5,5,0)
        center_mem[0] = pack_coord(5,5,0);
        do_reset();
        mode      = 2'b01; // BQ
        num_pts   = 16;
        num_out   = 1;      // 1 center point
        // radius = 5 units; in Q8.8: distance^2 = 5^2 * (256)^2 = 1638400
        // but our dist is Q8.8^2 = 1 unit = 256^2 = 65536
        // So r=5 -> r_sq = 25 * 65536 = 1638400
        radius_sq = 32'd1638400;
        run_until_done();
        $display("  BQ complete. %0d neighbors found.", ns_count);

        // --------------------------------------------------------
        // TEST 3: KNN - 4 nearest neighbors of center (1,1,0)
        // Expected: points 0,1,2,3 (cluster A)
        // --------------------------------------------------------
        $display("=== TEST 3: KNN (center (1,1,0), K=4) ===");
        fps_count = 0; ns_count = 0;
        center_mem[0] = pack_coord(1,1,0);
        do_reset();
        mode      = 2'b10; // KNN
        num_pts   = 16;
        num_out   = 1;
        radius_sq = 32'hFFFFFFFF; // not used in KNN
        run_until_done();
        $display("  KNN complete. %0d neighbors found.", ns_count);

        $display("=== All tests finished ===");
        #100;
        $finish;
    end

endmodule
