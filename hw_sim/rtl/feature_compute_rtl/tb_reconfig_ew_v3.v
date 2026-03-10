`timescale 1ns/1ps
`include "PE_array_reconfig_v3.v"

module tb_reconfig_v4_ew;

    parameter M          = 8;
    parameter N          = 8;
    parameter ARR_R      = 4;
    parameter ARR_C      = 4;
    parameter WORD_SIZE  = 16;
    parameter ACC_WIDTH  = 64;
    parameter CLK_PERIOD = 10;

    localparam MODE_OS         = 2'd0;
    localparam MODE_WS_PRELOAD = 2'd1;
    localparam MODE_WS_DIAG    = 2'd2;
    localparam MODE_EW         = 2'd3;

    reg clk, rst_n;
    reg en, psum_clr;
    reg [1:0] mode;

    reg  [ARR_R*WORD_SIZE-1:0]       a_left_bus;
    reg  [ARR_R-1:0]                 a_valid_bus;
    reg  [ARR_C*ACC_WIDTH-1:0]       vert_top_bus;
    reg  [ARR_C-1:0]                 vert_top_valid_bus;
    reg  [ARR_R*ARR_C*WORD_SIZE-1:0] w_local_bus;
    reg  [ARR_R*ARR_C-1:0]           w_local_valid_bus;
    reg  [ARR_R*ARR_C*WORD_SIZE-1:0] a_local_bus;
    reg  [ARR_R*ARR_C-1:0]           a_local_valid_bus;

    wire [ARR_C*ACC_WIDTH-1:0]       vert_bottom_bus;
    wire [ARR_C-1:0]                 vert_bottom_valid_bus;
    wire [ARR_R*ARR_C*ACC_WIDTH-1:0] psum_snapshot_bus;

    reg signed [WORD_SIZE-1:0] A_mem [0:M*N-1];
    reg signed [WORD_SIZE-1:0] B_mem [0:M*N-1];
    reg signed [ACC_WIDTH-1:0] C_golden [0:M*N-1];
    reg signed [ACC_WIDTH-1:0] C_dut    [0:M*N-1];

    integer m0, n0, r, c, idx, error_count;
    reg signed [WORD_SIZE-1:0] a_val, b_val;
    reg signed [ACC_WIDTH-1:0] dut_val, golden_val;

    PE_array_reconfig_v4 #(
        .ARR_R(ARR_R),
        .ARR_C(ARR_C),
        .WORD_SIZE(WORD_SIZE),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
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
        .a_local_bus(a_local_bus),
        .a_local_valid_bus(a_local_valid_bus),
        .vert_bottom_bus(vert_bottom_bus),
        .vert_bottom_valid_bus(vert_bottom_valid_bus),
        .psum_snapshot_bus(psum_snapshot_bus)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        $readmemh("A_ew.mem", A_mem);
        $readmemh("B_ew.mem", B_mem);
        $readmemh("C_ew.mem", C_golden);
    end

    task clear_inputs;
    begin
        a_left_bus = 0;
        a_valid_bus = 0;
        vert_top_bus = 0;
        vert_top_valid_bus = 0;
        w_local_bus = 0;
        w_local_valid_bus = 0;
        a_local_bus = 0;
        a_local_valid_bus = 0;
    end
    endtask

    task clear_dut_matrix;
    begin
        for (idx = 0; idx < M*N; idx = idx + 1)
            C_dut[idx] = 0;
    end
    endtask

    task run_one_ew_tile;
        input integer row_base;
        input integer col_base;
        integer rr, cc;
    begin
        // clear snapshots
        psum_clr = 1'b1;
        @(posedge clk);
        #1;
        psum_clr = 1'b0;

        clear_inputs();

        // directly map one tile of A and B to local PE ports
        for (rr = 0; rr < ARR_R; rr = rr + 1) begin
            for (cc = 0; cc < ARR_C; cc = cc + 1) begin
                if ((row_base + rr) < M && (col_base + cc) < N) begin
                    a_val = A_mem[(row_base + rr)*N + (col_base + cc)];
                    b_val = B_mem[(row_base + rr)*N + (col_base + cc)];
                end else begin
                    a_val = 0;
                    b_val = 0;
                end

                a_local_bus[(rr*ARR_C + cc)*WORD_SIZE +: WORD_SIZE] = a_val;
                a_local_valid_bus[(rr*ARR_C + cc)] = 1'b1;

                w_local_bus[(rr*ARR_C + cc)*WORD_SIZE +: WORD_SIZE] = b_val;
                w_local_valid_bus[(rr*ARR_C + cc)] = 1'b1;
            end
        end

        // one compute cycle
        @(posedge clk);
        #1;

        // collect results
        for (rr = 0; rr < ARR_R; rr = rr + 1) begin
            for (cc = 0; cc < ARR_C; cc = cc + 1) begin
                if ((row_base + rr) < M && (col_base + cc) < N) begin
                    C_dut[(row_base + rr)*N + (col_base + cc)] =
                        $signed(psum_snapshot_bus[(rr*ARR_C + cc)*ACC_WIDTH +: ACC_WIDTH]);
                end
            end
        end

        clear_inputs();
        @(posedge clk);
        #1;
    end
    endtask

    initial begin
        rst_n = 0;
        en = 0;
        psum_clr = 0;
        mode = MODE_EW;
        clear_inputs();

        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);

        clear_dut_matrix();
        en = 1'b1;

        for (m0 = 0; m0 < M; m0 = m0 + ARR_R) begin
            for (n0 = 0; n0 < N; n0 = n0 + ARR_C) begin
                run_one_ew_tile(m0, n0);
            end
        end

        en <= 0;
        @(posedge clk);
        #1;

        error_count = 0;

        $display("Golden C (elementwise):");
        for (r = 0; r < M; r = r + 1) begin
            for (c = 0; c < N; c = c + 1)
                $write("%12d ", $signed(C_golden[r*N+c]));
            $write("\n");
        end

        $display("DUT C (elementwise):");
        for (r = 0; r < M; r = r + 1) begin
            for (c = 0; c < N; c = c + 1) begin
                dut_val    = C_dut[r*N+c];
                golden_val = C_golden[r*N+c];
                $write("%12d ", dut_val);
                if (dut_val !== golden_val) begin
                    error_count = error_count + 1;
                    $display("ERROR at C[%0d][%0d]: DUT=%0d, GOLDEN=%0d",
                             r, c, dut_val, golden_val);
                end
            end
            $write("\n");
        end

        if (error_count == 0)
            $display("*** RECONFIG V4 EW TEST PASSED ***");
        else
            $display("*** RECONFIG V4 EW TEST FAILED: %0d mismatches ***", error_count);

        #(CLK_PERIOD*10);
        $finish;
    end

    initial begin
        $dumpfile("tb_reconfig_v4_ew.vcd");
        $dumpvars(0, tb_reconfig_v4_ew);
    end

endmodule