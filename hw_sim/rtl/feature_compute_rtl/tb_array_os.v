`timescale 1ns/1ps
`include "PE_array_os.v"

module tb_array_os;

    parameter ROW       = 16;
    parameter COL       = 16;
    parameter K         = 16;
    parameter WORD_SIZE = 16;
    parameter ACC_WIDTH = 32;
    parameter CLK_PERIOD = 10;

    reg clk;
    reg rst_n;
    reg start;
    reg en;
    reg psum_clr;

    reg  [ROW*WORD_SIZE-1:0] a_left_bus;
    reg  [ROW-1:0]           a_valid_bus;
    reg  [COL*WORD_SIZE-1:0] b_top_bus;
    reg  [COL-1:0]           b_valid_bus;

    wire [ROW*COL*ACC_WIDTH-1:0] c_bus;
    wire [ROW*COL-1:0]           c_valid_bus;
    wire                         done;

    // 原始矩阵
    reg signed [WORD_SIZE-1:0] A_mem [0:ROW*K-1];
    reg signed [WORD_SIZE-1:0] B_mem [0:K*COL-1];
    reg signed [ACC_WIDTH-1:0] C_golden [0:ROW*COL-1];

    integer cycle;
    integer i, j, kk;
    integer error_count;

    PE_array_os #(
        .ROW       (ROW),
        .COL       (COL),
        .K         (K),
        .WORD_SIZE (WORD_SIZE),
        .ACC_WIDTH (ACC_WIDTH)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .en          (en),
        .psum_clr    (psum_clr),
        .a_left_bus  (a_left_bus),
        .a_valid_bus(a_valid_bus),
        .b_top_bus   (b_top_bus),
        .b_valid_bus (b_valid_bus),
        .c_bus       (c_bus),
        .c_valid_bus (c_valid_bus),
        .done        (done)
    );

    // 时钟
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // 读取mem
    initial begin
        $readmemh("A.mem", A_mem);
        $readmemh("B.mem", B_mem);
        $readmemh("C_full.mem", C_golden);
    end

    // 注入任务
    task drive_inputs_at_cycle;
        input integer t;
        integer r, c, k_idx;
        reg signed [WORD_SIZE-1:0] a_val;
        reg signed [WORD_SIZE-1:0] b_val;
    begin
        a_left_bus  = 0;
        a_valid_bus = 0;
        b_top_bus   = 0;
        b_valid_bus = 0;

        // A[i][k] 在 t = i + k 注入，有效位随数据
        for (r = 0; r < ROW; r = r + 1) begin
            k_idx = t - r;
            if (k_idx >= 0 && k_idx < K) begin
                a_val = A_mem[r*K + k_idx];
                a_valid_bus[r] = 1'b1;
            end else begin
                a_val = 0;
                a_valid_bus[r] = 1'b0;
            end
            a_left_bus[r*WORD_SIZE +: WORD_SIZE] = a_val;
        end

        // B[k][j] 在 t = j + k 注入，有效位随数据
        for (c = 0; c < COL; c = c + 1) begin
            k_idx = t - c;
            if (k_idx >= 0 && k_idx < K) begin
                b_val = B_mem[k_idx*COL + c];
                b_valid_bus[c] = 1'b1;
            end else begin
                b_val = 0;
                b_valid_bus[c] = 1'b0;
            end
            b_top_bus[c*WORD_SIZE +: WORD_SIZE] = b_val;
        end
    end
    endtask

    task display_matrix_inputs;
    begin
        $display("A matrix:");
        for (i = 0; i < ROW; i = i + 1) begin
            for (j = 0; j < K; j = j + 1)
                $write("%4d ", $signed(A_mem[i*K + j]));
            $write("\n");
        end

        $display("B matrix:");
        for (i = 0; i < K; i = i + 1) begin
            for (j = 0; j < COL; j = j + 1)
                $write("%4d ", $signed(B_mem[i*COL + j]));
            $write("\n");
        end

        $display("Golden C matrix:");
        for (i = 0; i < ROW; i = i + 1) begin
            for (j = 0; j < COL; j = j + 1)
                $write("%8d ", $signed(C_golden[i*COL + j]));
            $write("\n");
        end
    end
    endtask

    task check_results;
        reg signed [ACC_WIDTH-1:0] dut_val;
    begin
        error_count = 0;
        $display("DUT C matrix:");
        for (i = 0; i < ROW; i = i + 1) begin
            for (j = 0; j < COL; j = j + 1) begin
                dut_val = c_bus[(i*COL + j)*ACC_WIDTH +: ACC_WIDTH];
                $write("%8d ", dut_val);

                if (dut_val !== C_golden[i*COL + j]) begin
                    error_count = error_count + 1;
                    $display("ERROR at C[%0d][%0d]: DUT=%0d, GOLDEN=%0d",
                             i, j, dut_val, C_golden[i*COL + j]);
                end
            end
            $write("\n");
        end

        if (error_count == 0)
            $display("*** OS ARRAY TEST PASSED ***");
        else
            $display("*** OS ARRAY TEST FAILED: %0d mismatches ***", error_count);
    end
    endtask

    initial begin
        rst_n       = 0;
        start       = 0;
        en          = 0;
        psum_clr    = 0;
        a_left_bus  = 0;
        a_valid_bus = 0;
        b_top_bus   = 0;
        b_valid_bus = 0;

        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);

        display_matrix_inputs();

        // 启动并清空 PE 内部累加器
        @(posedge clk);
        start    <= 1;
        psum_clr <= 1;
        en       <= 0;
        @(posedge clk);
        start    <= 0;
        psum_clr <= 0;
        en       <= 1;

        // 注入阶段：与 PE_array_os 脉动时序一致
        for (cycle = 0; cycle < K + ROW + COL + 4; cycle = cycle + 1) begin
            drive_inputs_at_cycle(cycle);
            @(posedge clk);
        end

        // 停止注入，等待 done 或尾部波前完成
        a_left_bus  <= 0;
        a_valid_bus <= 0;
        b_top_bus   <= 0;
        b_valid_bus <= 0;
        while (!done) begin
            @(posedge clk);
        end
        @(posedge clk);

        en <= 0;
        @(posedge clk);

        check_results();

        #(CLK_PERIOD*5);
        $finish;
    end

    initial begin
        $dumpfile("tb_array_os.vcd");
        $dumpvars(0, tb_array_os);
    end

endmodule