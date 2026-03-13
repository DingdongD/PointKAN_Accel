// // ============================================
// // Testbench
// // ============================================
// `include "merge_topk.v"
// module tb_topk_merge_sort;
//     parameter DATA_WIDTH = 16;
//     parameter BLOCK_SIZE = 8;
//     parameter NUM_BLOCKS = 4;
//     parameter K = 8;

//     reg clk;
//     reg rst_n;
//     reg start;
//     reg [DATA_WIDTH*BLOCK_SIZE*NUM_BLOCKS-1:0] data_in;
//     wire [DATA_WIDTH*K-1:0] topk_out;
//     wire done;

//     topk_merge_sort #(
//         .DATA_WIDTH(DATA_WIDTH),
//         .BLOCK_SIZE(BLOCK_SIZE),
//         .NUM_BLOCKS(NUM_BLOCKS),
//         .K(K)
//     ) dut (
//         .clk(clk),
//         .rst_n(rst_n),
//         .start(start),
//         .data_in(data_in),
//         .topk_out(topk_out),
//         .done(done)
//     );

//     // clock
//     initial begin
//         clk = 0;
//         forever #5 clk = ~clk;
//     end

//     // helper: pack BLOCK_SIZE elements into a block
//     function [DATA_WIDTH*BLOCK_SIZE-1:0] pack_block;
//         input integer idx; // not used, external assignment
//         input [DATA_WIDTH-1:0] a0, a1, a2, a3, a4, a5, a6, a7;
//         begin
//             pack_block = {a0, a1, a2, a3, a4, a5, a6, a7}; // MSB is element 0 here; adjust when slicing in design
//         end
//     endfunction

//     initial begin
//         rst_n = 0;
//         start = 0;
//         data_in = 0;
//         #20 rst_n = 1;

//         // Prepare 4 blocks each with 8 values. We want a variety of numbers.
//         // We'll pack blocks in order: block0 | block1 | block2 | block3 (block0 at LSB side in our top module slicing convention)
//         // Because top module slices as data_in[block*BLOCK_SIZE*DATA_WIDTH +: BLOCK_SIZE*DATA_WIDTH]
//         // let's construct by direct concatenation (block3 most significant).
//         // Block 0:
//         // elements: 100, 50, 75, 25, 90, 40, 60, 30
//         // Block 1:
//         // elements: 15, 200, 5,  80,  120, 45,  70,  10
//         // Block 2:
//         // elements: 300, 2,  250, 20,  60,  55,  5,   1
//         // Block 3:
//         // elements: 33, 66, 99, 12,  88,  44,  77,  11

//         // pack with block0 as least-significant chunk:
//         data_in = {
//             // block3 (most significant chunk)
//             16'd33, 16'd66, 16'd99, 16'd12, 16'd88, 16'd44, 16'd77, 16'd11,
//             // block2
//             16'd300, 16'd2, 16'd250, 16'd20, 16'd60, 16'd55, 16'd5, 16'd9,
//             // block1
//             16'd15, 16'd200, 16'd5, 16'd80, 16'd120, 16'd45, 16'd70, 16'd10,
//             // block0 (least significant chunk)
//             16'd100, 16'd50, 16'd75, 16'd25, 16'd90, 16'd40, 16'd60, 16'd30
//         };

//         #10 start = 1;
//         #10 start = 0;

//         // wait for done
//         wait(done);
//         #50;

//         // display topk_out
//         $display("Result Top-%0d (packed): %h", K, topk_out);
//         // unpack and print as numbers
//         $display("Top-%0d elements (smallest first):", K);
//         $display("%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d",
//                  topk_out[0*DATA_WIDTH +: DATA_WIDTH],
//                  topk_out[1*DATA_WIDTH +: DATA_WIDTH],
//                  topk_out[2*DATA_WIDTH +: DATA_WIDTH],
//                  topk_out[3*DATA_WIDTH +: DATA_WIDTH],
//                  topk_out[4*DATA_WIDTH +: DATA_WIDTH],
//                  topk_out[5*DATA_WIDTH +: DATA_WIDTH],
//                  topk_out[6*DATA_WIDTH +: DATA_WIDTH],
//                  topk_out[7*DATA_WIDTH +: DATA_WIDTH]);

//         #100 $finish;
//     end
// endmodule



// ============================================
// 调试Testbench - 分步测试Bitonic排序器
// ============================================
// `include "merge_topk.v"
// module tb_bitonic_debug;
//     parameter DATA_WIDTH = 16;
//     parameter BLOCK_SIZE = 8;
//     parameter K = 4;
    
//     reg clk, rst_n, start;
//     reg [DATA_WIDTH*BLOCK_SIZE-1:0] data_in;
//     wire [DATA_WIDTH*K-1:0] data_out;
//     wire valid;
    
//     // 实例化bitonic排序器
//     block_sorter #(
//         .DATA_WIDTH(DATA_WIDTH),
//         .BLOCK_SIZE(BLOCK_SIZE),
//         .K(K)
//     ) sorter (
//         .clk(clk),
//         .rst_n(rst_n),
//         .start(start),
//         .data_in(data_in),
//         .data_out(data_out),
//         .valid(valid)
//     );
    
//     // Clock
//     initial begin
//         clk = 0;
//         forever #5 clk = ~clk;
//     end
    
//     integer i;
    
//     initial begin
//         $dumpfile("wave.vcd");
//         $dumpvars(0, tb_bitonic_debug);
        
//         rst_n = 0;
//         start = 0;
//         #20 rst_n = 1;
        
//         // ========================================
//         // 测试1: 验证打包顺序
//         // ========================================
//         $display("\n=== Test 1: Verify packing order ===");
//         // 打包 {7,6,5,4,3,2,1,0}，按照Verilog规范：
//         // MSB部分是7，LSB部分是0
//         data_in = {16'd7, 16'd6, 16'd5, 16'd4, 16'd3, 16'd2, 16'd1, 16'd0};
        
//         $display("Packed data_in (hex): %h", data_in);
//         $display("Expected: data_in[15:0]=0, data_in[31:16]=1, ..., data_in[127:112]=7");
//         $display("Actual LSB part [15:0] = %h (should be 0000)", data_in[15:0]);
//         $display("Actual MSB part [127:112] = %h (should be 0007)", data_in[127:112]);
        
//         #10 start = 1;
//         #10 start = 0;
//         wait(valid);
//         #10;
        
//         $display("\nAfter sorting, input was: 7,6,5,4,3,2,1,0");
//         $display("Output Top-%0d: %0d, %0d, %0d, %0d", K,
//                  data_out[0*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[1*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[2*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[3*DATA_WIDTH +: DATA_WIDTH]);
//         $display("Expected: 0, 1, 2, 3");
        
//         if (data_out[0*DATA_WIDTH +: DATA_WIDTH] == 0 &&
//             data_out[1*DATA_WIDTH +: DATA_WIDTH] == 1 &&
//             data_out[2*DATA_WIDTH +: DATA_WIDTH] == 2 &&
//             data_out[3*DATA_WIDTH +: DATA_WIDTH] == 3)
//             $display("? PASS");
//         else
//             $display("? FAIL");
        
//         // 显示pipeline[0]加载的值
//         $display("\nPipeline[0] loaded values:");
//         for (i = 0; i < 8; i = i + 1) begin
//             $display("  pipeline[0][%0d] = %0d", i, sorter.pipeline[0][i]);
//         end
        
//         // ========================================
//         // 测试2: 随机序列
//         // ========================================
//         #50;
//         $display("\n=== Test 2: Random sequence ===");
//         // 按照 LSB->MSB 的顺序：30, 60, 40, 90, 25, 75, 50, 100
//         data_in = {16'd100, 16'd50, 16'd75, 16'd25, 16'd90, 16'd40, 16'd60, 16'd30};
        
//         #10 start = 1;
//         #10 start = 0;
        
//         // 显示加载后的值
//         #15; // 等待加载
//         $display("Input packing: {100, 50, 75, 25, 90, 40, 60, 30}");
//         $display("Pipeline[0] after load:");
//         for (i = 0; i < 8; i = i + 1) begin
//             $display("  pipeline[0][%0d] = %0d", i, sorter.pipeline[0][i]);
//         end
        
//         wait(valid);
//         #10;
        
//         $display("\nFull sorted result (pipeline[6]):");
//         for (i = 0; i < 8; i = i + 1) begin
//             $display("  pipeline[6][%0d] = %0d", i, sorter.pipeline[6][i]);
//         end
        
//         $display("\nOutput Top-%0d: %0d, %0d, %0d, %0d", K,
//                  data_out[0*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[1*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[2*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[3*DATA_WIDTH +: DATA_WIDTH]);
//         $display("Expected: 25, 30, 40, 50");
        
//         if (data_out[0*DATA_WIDTH +: DATA_WIDTH] == 25 &&
//             data_out[1*DATA_WIDTH +: DATA_WIDTH] == 30 &&
//             data_out[2*DATA_WIDTH +: DATA_WIDTH] == 40 &&
//             data_out[3*DATA_WIDTH +: DATA_WIDTH] == 50)
//             $display("? PASS");
//         else
//             $display("? FAIL");
        
//         #100 $finish;
//     end
    
//     // 监控所有stage
//     always @(posedge clk) begin
//         if (sorter.valid_pipe[1]) begin
//             $display("Stage 1: %0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
//                 sorter.pipeline[1][0], sorter.pipeline[1][1], sorter.pipeline[1][2], sorter.pipeline[1][3],
//                 sorter.pipeline[1][4], sorter.pipeline[1][5], sorter.pipeline[1][6], sorter.pipeline[1][7]);
//         end
//         if (sorter.valid_pipe[6]) begin
//             $display("Stage 6 (final): %0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
//                 sorter.pipeline[6][0], sorter.pipeline[6][1], sorter.pipeline[6][2], sorter.pipeline[6][3],
//                 sorter.pipeline[6][4], sorter.pipeline[6][5], sorter.pipeline[6][6], sorter.pipeline[6][7]);
//         end
//     end
// endmodule



// module tb_bitonic_debug;
//     parameter DATA_WIDTH = 16;
//     parameter BLOCK_SIZE = 8;
//     parameter K          = 6;          // <-- 现在是 6

//     reg  clk, rst_n, start;
//     reg  [DATA_WIDTH*BLOCK_SIZE-1:0] data_in;
//     wire [DATA_WIDTH*K-1:0]          data_out;
//     wire                            valid;

//     // DUT
//     block_sorter #(
//         .DATA_WIDTH (DATA_WIDTH),
//         .BLOCK_SIZE (BLOCK_SIZE),
//         .K          (K)
//     ) sorter (
//         .clk     (clk),
//         .rst_n   (rst_n),
//         .start   (start),
//         .data_in (data_in),
//         .data_out(data_out),
//         .valid   (valid)
//     );

//     // -------------------------------------------------
//     // Clock
//     // -------------------------------------------------
//     initial begin
//         clk = 0;
//         forever #5 clk = ~clk;
//     end

//     integer i;

//     // -------------------------------------------------
//     // Main stimulus
//     // -------------------------------------------------
//     initial begin
//         $dumpfile("wave.vcd");
//         $dumpvars(0, tb_bitonic_debug);

//         rst_n = 0; start = 0;
//         #20 rst_n = 1;

//         // ==============================================================
//         // Test 1 : packing + small K 
//         // ==============================================================
//         $display("\n=== Test 1: Verify packing order ===");
//         data_in = {16'd7, 16'd6, 16'd5, 16'd45, 16'd3, 16'd2, 16'd1, 16'd0};

//         $display("Packed data_in (hex): %h", data_in);
//         $display("LSB [15:0] = %h  (should be 0000)", data_in[15:0]);
//         $display("MSB [127:112] = %h (should be 0007)", data_in[127:112]);

//         #10 start = 1; #10 start = 0;
//         wait(valid);
//         @(posedge clk);               // 保证 data_out 已更新

//         $display("\nInput was: 7,6,5,45,3,2,1,0");
//         $display("Top-%0d (K=4) : %0d %0d %0d %0d",
//                  4,
//                  data_out[0*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[1*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[2*DATA_WIDTH +: DATA_WIDTH],
//                  data_out[3*DATA_WIDTH +: DATA_WIDTH]);
//         $display("Expected      : 0 1 2 3");
//         if (data_out[0*16 +: 16] == 0 && data_out[1*16 +: 16] == 1 &&
//             data_out[2*16 +: 16] == 2 && data_out[3*16 +: 16] == 3)
//             $display("PASS");
//         else
//             $display("FAIL");

//         // -------------------------------------------------
//         // pipeline[0] 内容（验证加载顺序）
//         // -------------------------------------------------
//         $display("\nPipeline[0] loaded values:");
//         for (i = 0; i < BLOCK_SIZE; i = i + 1)
//             $display("  pipeline[0][%0d] = %0d", i, sorter.pipeline[0][i]);

//         // ==============================================================
//         // Test 2 : random sequence, K = 6
//         // ==============================================================
//         #50;
//         $display("\n=== Test 2: Random sequence (K=%0d) ===", K);
//         // 打包顺序：{MSB, ..., LSB}
//         data_in = {16'd100, 16'd50, 16'd75, 16'd25, 16'd90, 16'd40, 16'd10, 16'd37};

//         #10 start = 1; #10 start = 0;

//         #15;
//         $display("Input packing (MSB to LSB): {100,50,75,25,90,40,10,37}");
//         $display("Pipeline[0] after load:");
//         for (i = 0; i < BLOCK_SIZE; i = i + 1)
//             $display("  pipeline[0][%0d] = %0d", i, sorter.pipeline[0][i]);

//         wait(valid);
//         @(posedge clk);               // 等待 data_out 完全更新

//         // 完整排序结果
//         $display("\nFull sorted result (pipeline[6]):");
//         for (i = 0; i < BLOCK_SIZE; i = i + 1)
//             $display("  pipeline[6][%0d] = %0d", i, sorter.pipeline[6][i]);

//         // Top-K 输出
//         $display("\nOutput Top-%0d :", K);
//         for (i = 0; i < K; i = i + 1)
//             $display("  [%0d] = %0d", i, data_out[i*DATA_WIDTH +: DATA_WIDTH]);

//         $display("Expected : 10, 25, 37, 40, 50, 75");
//         if (data_out[0*16 +: 16] == 10 && data_out[1*16 +: 16] == 25 &&
//             data_out[2*16 +: 16] == 37 && data_out[3*16 +: 16] == 40 &&
//             data_out[4*16 +: 16] == 50 && data_out[5*16 +: 16] == 75)
//             $display("PASS");
//         else
//             $display("FAIL");

//         #100 $finish;
//     end

//     // -------------------------------------------------
//     // 监控中间 stage（方便调试）
//     // -------------------------------------------------
//     always @(posedge clk) begin
//         if (sorter.valid_pipe[1])
//             $display("Stage 1: %0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
//                      sorter.pipeline[1][0],sorter.pipeline[1][1],
//                      sorter.pipeline[1][2],sorter.pipeline[1][3],
//                      sorter.pipeline[1][4],sorter.pipeline[1][5],
//                      sorter.pipeline[1][6],sorter.pipeline[1][7]);
//         if (sorter.valid_pipe[6])
//             $display("Stage 6: %0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
//                      sorter.pipeline[6][0],sorter.pipeline[6][1],
//                      sorter.pipeline[6][2],sorter.pipeline[6][3],
//                      sorter.pipeline[6][4],sorter.pipeline[6][5],
//                      sorter.pipeline[6][6],sorter.pipeline[6][7]);
//     end
// endmodule



// `include "merge_topk.v"
// module tb_topk_merge_sort;

//     parameter DATA_WIDTH = 16;
//     parameter BLOCK_SIZE = 8;
//     parameter NUM_BLOCKS = 2;
//     parameter K          = 4;

//     reg clk, rst_n, start;

//     // 输入内存：NUM_BLOCKS × BLOCK_SIZE
//     reg [DATA_WIDTH-1:0] mem_in [0:NUM_BLOCKS*BLOCK_SIZE-1];

//     // Golden Top-K
//     reg [DATA_WIDTH-1:0] mem_gold [0:K-1];

//     // 打包输入
//     reg [DATA_WIDTH*BLOCK_SIZE*NUM_BLOCKS-1:0] data_in;

//     wire [DATA_WIDTH*K-1:0] topk_out;
//     wire done;

//     // DUT
//     topk_merge_sort #(
//         .DATA_WIDTH(DATA_WIDTH),
//         .BLOCK_SIZE(BLOCK_SIZE),
//         .NUM_BLOCKS(NUM_BLOCKS),
//         .K(K)
//     ) dut (
//         .clk(clk),
//         .rst_n(rst_n),
//         .start(start),
//         .data_in(data_in),
//         .topk_out(topk_out),
//         .done(done)
//     );

//     // clock
//     initial begin
//         clk = 0;
//         forever #5 clk = ~clk;
//     end

//     integer i, blk, idx;

//     // -------------------------------------------------------------
//     // pack data: 将 mem_in 写入 data_in 总线
//     // -------------------------------------------------------------
//     task pack_all_blocks;
//     begin
//         for (blk = 0; blk < NUM_BLOCKS; blk = blk + 1) begin
//             for (idx = 0; idx < BLOCK_SIZE; idx = idx + 1) begin
//                 data_in[(blk*BLOCK_SIZE+idx)*DATA_WIDTH +: DATA_WIDTH]
//                     = mem_in[blk*BLOCK_SIZE + idx];  // 这里会对输入数据进行分发处理 
//             end
//         end
//     end
//     endtask
//     integer err = 0;
//     // -------------------------------------------------------------
//     // main
//     // -------------------------------------------------------------
//     initial begin
//         $dumpfile("wave.vcd");
//         $dumpvars(0, tb_topk_merge_sort);

//         rst_n = 0; start = 0;
//         #20 rst_n = 1;

//         // ============================================
//         // 读取输入与 golden
//         // ============================================
//         $display("Loading input_blocks.mem ...");
//         $readmemh("input1.mem", mem_in);

//         $display("Loading golden_topk.mem ...");
//         $readmemh("golden1.mem", mem_gold);

//         // 打印输入内容
//         $display("\nInput Blocks:");
//         for (blk = 0; blk < NUM_BLOCKS; blk = blk + 1) begin
//             $write("Block %0d : ", blk);
//             for (idx = 0; idx < BLOCK_SIZE; idx = idx + 1)
//                 $write("%0d ", mem_in[blk*BLOCK_SIZE + idx]);
//             $write("\n");
//         end

//         // 打印 golden
//         $display("\nGolden Top-%0d:", K);
//         for (i = 0; i < K; i = i + 1)
//             $display("  %0d", mem_gold[i]);

//         // ============================================
//         // 数据打包
//         // ============================================
//         pack_all_blocks();

//         // ============================================
//         // 运行 DUT
//         // ============================================
//         #10 start = 1;
//         #10 start = 0;

//         wait(done);
//         @(posedge clk);

//         // ============================================
//         // 打印 DUT 输出
//         // ============================================
//         $display("\nDUT Output Top-%0d:", K);
//         for (i = 0; i < K; i = i + 1)
//             $display("  out[%0d] = %0d", i,
//                 topk_out[i*DATA_WIDTH +: DATA_WIDTH]);

//         // ============================================
//         // 比较结果
//         // ============================================
        
//         for (i = 0; i < K; i = i + 1) begin
//             if (topk_out[i*DATA_WIDTH +: DATA_WIDTH] != mem_gold[i]) begin
//                 $display("Mismatch at %0d: DUT=%0d, GOLD=%0d",
//                          i,
//                          topk_out[i*DATA_WIDTH +: DATA_WIDTH],
//                          mem_gold[i]);
//                 err = 1;
//             end
//         end

//         if (err == 0)
//             $display("\n===== PASS ? All Top-K match =====");
//         else
//             $display("\n===== FAIL ? Mismatch =====");

//         #50 $finish;
//     end

// endmodule



// 下面这个版本是时分复用版本 上面的例化多个sorter
`include "merge_topk.v"
module tb_topk_merge_sort;

    parameter DATA_WIDTH = 16;
    parameter BLOCK_SIZE = 8;
    parameter TOTAL_DATA = 16;   // 这里总数据数
    parameter K          = 4;

    reg clk, rst_n, start;

    // 输入内存：TOTAL_DATA 个元素
    reg [DATA_WIDTH-1:0] mem_in [0:TOTAL_DATA-1];

    // Golden Top-K
    reg [DATA_WIDTH-1:0] mem_gold [0:K-1];

    // 打包输入
    reg [DATA_WIDTH*TOTAL_DATA-1:0] data_in;

    wire [DATA_WIDTH*K-1:0] topk_out;
    wire done;

    // DUT
    topk_merge_sort #(
        .DATA_WIDTH(DATA_WIDTH),
        .BLOCK_SIZE(BLOCK_SIZE),
        .TOTAL_DATA(TOTAL_DATA),
        .K(K)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .topk_out(topk_out),
        .done(done)
    );

    wire [DATA_WIDTH-1:0] topk0, topk1, topk2, topk3;

    topk_decoder #(
        .DATA_WIDTH(DATA_WIDTH),
        .K(K)
    ) decoder_inst (
        .packed_topk(topk_out),
        .topk0(topk0),
        .topk1(topk1),
        .topk2(topk2),
        .topk3(topk3)
    );

    // clock
    initial begin
        clk = 1;
        forever #5 clk = ~clk;
    end

    integer i;

    // -------------------------------------------------------------
    // pack data: 将 mem_in 写入 data_in 总线
    // -------------------------------------------------------------
    task pack_all_data;
    begin
        for (i = 0; i < TOTAL_DATA; i = i + 1) begin
            data_in[i*DATA_WIDTH +: DATA_WIDTH] = mem_in[i];
        end
    end
    endtask

    integer err = 0;
    reg [DATA_WIDTH-1:0] topk_array [0:K-1];
    // -------------------------------------------------------------
    // main
    // -------------------------------------------------------------
    initial begin
        $dumpfile("wave.vcd");
        $dumpvars(0, tb_topk_merge_sort);

        rst_n = 0; start = 0;
        #20 rst_n = 1;

        // ============================================
        // 读取输入与 golden
        // ============================================
        $display("Loading input_data.mem ...");
        $readmemh("input1.mem", mem_in);

        $display("Loading golden_topk.mem ...");
        $readmemh("golden1.mem", mem_gold);

        // 打印输入内容
        $display("\nInput Data:");
        for (i = 0; i < TOTAL_DATA; i = i + 1)
            $display("  mem_in[%0d] = %0d", i, mem_in[i]);

        // 打印 golden
        $display("\nGolden Top-%0d:", K);
        for (i = 0; i < K; i = i + 1)
            $display("  %0d", mem_gold[i]);

        // ============================================
        // 数据打包
        // ============================================
        pack_all_data();

        // ============================================
        // 运行 DUT
        // ============================================
        #10 start = 1;
        #10 start = 0;

        wait(done);
        @(posedge clk);

        // ============================================
        // 打印 DUT 输出
        // ============================================
        $display("\nDUT Output Top-%0d:", K);
        for (i = 0; i < K; i = i + 1)
            $display("  out[%0d] = %0d", i,
                topk_out[i*DATA_WIDTH +: DATA_WIDTH]);

        // ============================================
        // 比较结果
        // ============================================
        // for (i = 0; i < K; i = i + 1) begin
        //     if (topk_out[i*DATA_WIDTH +: DATA_WIDTH] != mem_gold[i]) begin
        //         $display("Mismatch at %0d: DUT=%0d, GOLD=%0d",
        //                  i,
        //                  topk_out[i*DATA_WIDTH +: DATA_WIDTH],
        //                  mem_gold[i]);
        //         err = 1;
        //     end
        // end

        topk_array[0] = topk0;
        topk_array[1] = topk1;
        topk_array[2] = topk2;
        topk_array[3] = topk3;

        for (i=0; i<K; i=i+1) begin
            if (topk_array[i] != mem_gold[i]) begin
                $display("Mismatch at %0d: DUT=%0d, GOLD=%0d", 
                        i, topk_array[i], mem_gold[i]);
                err = 1;
            end
        end

        if (err == 0)
            $display("\n===== PASS ? All Top-K match =====");
        else
            $display("\n===== FAIL ? Mismatch =====");

        #50 $finish;
    end

endmodule












// 下面这部分模块用于测试bitonic_genertor.py脚本鲁棒性 修改input.mem和golden.mem可以实现不同数据集测试 对应MOC_SIM A100下py文件
// `include "topk_b16_k4.v"
// module tb_bitonic_readmem;

//     parameter DATA_WIDTH = 16;
//     parameter BLOCK_SIZE = 16;
//     parameter K          = 4;

//     reg clk, rst_n, start;

//     reg [DATA_WIDTH-1:0] mem_in [0:BLOCK_SIZE-1];
//     reg [DATA_WIDTH-1:0] mem_gold [0:K-1];

//     reg [DATA_WIDTH*BLOCK_SIZE-1:0] data_in;
//     wire [DATA_WIDTH*K-1:0] data_out;
//     wire valid;

//     // DUT
//     block_sorter #(
//         .DATA_WIDTH(DATA_WIDTH),
//         .BLOCK_SIZE(BLOCK_SIZE),
//         .K(K)
//     ) dut (
//         .clk(clk),
//         .rst_n(rst_n),
//         .start(start),
//         .data_in(data_in),
//         .data_out(data_out),
//         .valid(valid)
//     );

//     // clock
//     initial begin
//         clk = 1;
//         forever #5 clk = ~clk;
//     end

//     integer i;

//     // -----------------------------------------------------
//     // pack array -> data_in bus
//     // -----------------------------------------------------
//     task pack_data;
//     begin
//         for (i = 0; i < BLOCK_SIZE; i = i + 1) begin
//             data_in[i*DATA_WIDTH +: DATA_WIDTH] = mem_in[i];
//         end
//     end
//     endtask
//     integer err = 0;
//     // -----------------------------------------------------
//     // main
//     // -----------------------------------------------------
//     initial begin
//         $dumpfile("wave.vcd");
//         $dumpvars(0, tb_bitonic_readmem);

//         rst_n = 0; start = 0;
//         #20 rst_n = 1;

//         // ---------------------------------------------------------
//         // 从文件加载随机输入与 Golden Top-K
//         // ---------------------------------------------------------
//         $display("\nLoading input.mem...");
//         $readmemh("input1.mem",  mem_in);

//         $display("Loading golden.mem...");
//         $readmemh("golden1.mem", mem_gold);

//         // 打印加载内容
//         $display("\nLoaded input values:");
//         for (i = 0; i < BLOCK_SIZE; i = i + 1)
//             $display("  in[%0d] = %0d", i, mem_in[i]);

//         $display("\nGolden Top-%0d:", K);
//         for (i = 0; i < K; i = i + 1)
//             $display("  gold[%0d] = %0d", i, mem_gold[i]);

//         // ---------------------------------------------------------
//         // 打包输入
//         // ---------------------------------------------------------
//         pack_data();

//         // ---------------------------------------------------------
//         // 触发 sorter
//         // ---------------------------------------------------------
//         #10 start = 1;
//         #10 start = 0;

//         // 等待 valid = 1
//         wait(valid);
//         @(posedge clk); // data_out valid

//         $display("\nDUT Output:");
//         for (i = 0; i < K; i = i + 1)
//             $display("  out[%0d] = %0d", i, data_out[i*DATA_WIDTH +: DATA_WIDTH]);

//         // ---------------------------------------------------------
//         // Compare with golden
//         // ---------------------------------------------------------
        
//         for (i = 0; i < K; i = i + 1) begin
//             if (data_out[i*DATA_WIDTH +: DATA_WIDTH] != mem_gold[i]) begin
//                 $display("Mismatch at %0d: DUT=%0d, GOLD=%0d",
//                          i,
//                          data_out[i*DATA_WIDTH +: DATA_WIDTH],
//                          mem_gold[i]);
//                 err = 1;
//             end
//         end

//         if (!err)
//             $display("\n==== PASS ? All values match golden ====");
//         else
//             $display("\n==== FAIL ? Mismatch found ====");

//         #50 $finish;
//     end

// endmodule
