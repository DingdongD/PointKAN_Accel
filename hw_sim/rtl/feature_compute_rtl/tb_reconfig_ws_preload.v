`timescale 1ns/1ps
`include "top_reconfig_systolic_v2.v"

module tb_reconfig_v2_ws_preload;

    parameter M          = 8;
    parameter K          = 8;
    parameter N          = 8;
    parameter ARR_R      = 4;
    parameter ARR_C      = 4;
    parameter WORD_SIZE  = 16;
    parameter ACC_WIDTH  = 64;
    parameter CLK_PERIOD = 10;

    localparam MODE_OS         = 2'd0;
    localparam MODE_WS_PRELOAD = 2'd1;
    localparam MODE_WS_DIAG    = 2'd2;

    reg clk, rst_n, start;
    reg [1:0] mode;

    reg signed [WORD_SIZE-1:0] A_mem [0:M*K-1];
    reg signed [WORD_SIZE-1:0] B_mem [0:K*N-1];
    reg signed [ACC_WIDTH-1:0] C_golden [0:M*N-1];

    reg [M*K*WORD_SIZE-1:0] A_flat;
    reg [K*N*WORD_SIZE-1:0] B_flat;

    wire [M*N*ACC_WIDTH-1:0] C_out_flat;
    wire done;

    integer i, j, error_count;
    reg signed [ACC_WIDTH-1:0] dut_val, golden_val;

    top_reconfig_systolic_v2 #(
        .M(M), .K(K), .N(N),
        .ARR_R(ARR_R), .ARR_C(ARR_C),
        .WORD_SIZE(WORD_SIZE), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .mode(mode),
        .A_flat(A_flat),
        .B_flat(B_flat),
        .C_out_flat(C_out_flat),
        .done(done)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        $readmemh("A.mem", A_mem);
        $readmemh("B.mem", B_mem);
        $readmemh("C_full.mem", C_golden);

        A_flat = 0;
        B_flat = 0;

        for (i = 0; i < M; i = i + 1)
            for (j = 0; j < K; j = j + 1)
                A_flat[(i*K + j)*WORD_SIZE +: WORD_SIZE] = A_mem[i*K + j];

        for (i = 0; i < K; i = i + 1)
            for (j = 0; j < N; j = j + 1)
                B_flat[(i*N + j)*WORD_SIZE +: WORD_SIZE] = B_mem[i*N + j];
    end

    initial begin
        rst_n = 0;
        start = 0;
        mode  = MODE_WS_PRELOAD;

        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);

        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        wait(done == 1'b1);
        @(posedge clk);
        #1;

        error_count = 0;

        $display("Golden C:");
        for (i = 0; i < M; i = i + 1) begin
            for (j = 0; j < N; j = j + 1)
                $write("%12d ", $signed(C_golden[i*N+j]));
            $write("\n");
        end

        $display("DUT C:");
        for (i = 0; i < M; i = i + 1) begin
            for (j = 0; j < N; j = j + 1) begin
                dut_val    = $signed(C_out_flat[(i*N+j)*ACC_WIDTH +: ACC_WIDTH]);
                golden_val = C_golden[i*N+j];
                $write("%12d ", dut_val);
                if (dut_val !== golden_val) begin
                    error_count = error_count + 1;
                    $display("ERROR at C[%0d][%0d]: DUT=%0d, GOLDEN=%0d",
                             i, j, dut_val, golden_val);
                end
            end
            $write("\n");
        end

        if (error_count == 0)
            $display("*** TOP V2 WS_PRELOAD TEST PASSED ***");
        else
            $display("*** TOP V2 WS_PRELOAD TEST FAILED: %0d mismatches ***", error_count);

        #(CLK_PERIOD*10);
        $finish;
    end

    initial begin
        $dumpfile("tb_reconfig_v2_ws_preload.vcd");
        $dumpvars(0, tb_reconfig_v2_ws_preload);
    end

endmodule