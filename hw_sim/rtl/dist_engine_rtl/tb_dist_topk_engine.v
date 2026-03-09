// =============================================================================
// File    : tb_dist_topk_engine.v
// Testbench for dist_topk_engine (脉动阵列 + CMP行 + 归约树)
//
// 测试用例：
//   TC1: Top-1, k=1, 64个点，验证最近邻索引和距离
//   TC2: Top-4, k=4, 64个点，验证4个最近邻
//   TC3: Top-8, k=8, 64个点
//   TC4: Top-64 (全排序), k=64
//   TC5: Top-1连续两批次，验证 start 复位逻辑
// =============================================================================

`timescale 1ns/1ps
`include "cmp_unit.v"
`include "reduce_node.v"
`include "dist_topk_engine.v"
`include "topk_reduce_tree.v"
`include "distance_engine.v"
`include "PE.v"

module tb_dist_topk_engine;

    parameter PE_NUM     = 64;
    parameter DATA_WIDTH = 16;
    parameter ACC_WIDTH  = 48;
    parameter IDX_WIDTH  = 10;
    parameter K_MAX      = 64;
    parameter CLK_PERIOD = 10;

    reg                             clk, rst_n, start;
    reg [$clog2(K_MAX):0]           k_sel;
    reg [DATA_WIDTH-1:0]            q_x, q_y, q_z;
    reg [PE_NUM*DATA_WIDTH-1:0]     db_x_packed, db_y_packed, db_z_packed;
    reg [IDX_WIDTH-1:0]             batch_base_idx;

    wire [K_MAX*ACC_WIDTH-1:0]      topk_dist;
    wire [K_MAX*IDX_WIDTH-1:0]      topk_idx;
    wire                            topk_valid;
    wire                            pe_result_valid_dbg;
    wire [$clog2(PE_NUM)-1:0]       pe_result_idx_dbg;

    dist_topk_engine #(
        .PE_NUM    (PE_NUM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .IDX_WIDTH (IDX_WIDTH),
        .K_MAX     (K_MAX)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (start),
        .k_sel             (k_sel),
        .q_x               (q_x),
        .q_y               (q_y),
        .q_z               (q_z),
        .db_x_packed       (db_x_packed),
        .db_y_packed       (db_y_packed),
        .db_z_packed       (db_z_packed),
        .batch_base_idx    (batch_base_idx),
        .topk_dist         (topk_dist),
        .topk_idx          (topk_idx),
        .topk_valid        (topk_valid),
        .pe_result_valid_dbg(pe_result_valid_dbg),
        .pe_result_idx_dbg (pe_result_idx_dbg)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // 黄金模型：暴力计算Top-K
    // =========================================================================
    reg [ACC_WIDTH-1:0] all_dist  [0:PE_NUM-1];
    reg [IDX_WIDTH-1:0] all_idx   [0:PE_NUM-1];
    reg [ACC_WIDTH-1:0] golden_dist[0:K_MAX-1];
    reg [IDX_WIDTH-1:0] golden_idx [0:K_MAX-1];

    task compute_golden;
        input signed [DATA_WIDTH-1:0] qx, qy, qz;
        input integer k;
        input [IDX_WIDTH-1:0] base;
        integer i, j, min_j;
        reg [ACC_WIDTH-1:0] tmp_d;
        reg [IDX_WIDTH-1:0] tmp_i;
        reg signed [DATA_WIDTH-1:0] dx, dy, dz;
        reg used [0:PE_NUM-1];
        begin
            // 计算所有距离
            for (i = 0; i < PE_NUM; i = i + 1) begin
                dx = qx - $signed(db_x_packed[i*DATA_WIDTH +: DATA_WIDTH]);
                dy = qy - $signed(db_y_packed[i*DATA_WIDTH +: DATA_WIDTH]);
                dz = qz - $signed(db_z_packed[i*DATA_WIDTH +: DATA_WIDTH]);
                all_dist[i] = dx*dx + dy*dy + dz*dz;
                all_idx [i] = base + i[IDX_WIDTH-1:0];
                used[i]     = 0;
            end
            // 选择排序取Top-K（k个最小）
            for (i = 0; i < k && i < PE_NUM; i = i + 1) begin
                min_j = -1;
                tmp_d = {ACC_WIDTH{1'b1}};
                for (j = 0; j < PE_NUM; j = j + 1) begin
                    if (!used[j] && all_dist[j] < tmp_d) begin
                        tmp_d = all_dist[j];
                        tmp_i = all_idx[j];
                        min_j = j;
                    end
                end
                golden_dist[i] = tmp_d;
                golden_idx [i] = tmp_i;
                if (min_j >= 0) used[min_j] = 1;
            end
            // 剩余槽填最大值
            for (i = k; i < K_MAX; i = i + 1) begin
                golden_dist[i] = {ACC_WIDTH{1'b1}};
                golden_idx [i] = {IDX_WIDTH{1'b0}};
            end
        end
    endtask

    // =========================================================================
    // 结果检查
    // =========================================================================
    task check_topk;
        input integer tc;
        input integer k;
        integer i, err;
        reg [ACC_WIDTH-1:0] dut_d;
        reg [IDX_WIDTH-1:0] dut_i;
        begin
            err = 0;
            $display("[TC%0d] Top-%0d 结果检查:", tc, k);
            for (i = 0; i < k; i = i + 1) begin
                dut_d = topk_dist[i*ACC_WIDTH +: ACC_WIDTH];
                dut_i = topk_idx [i*IDX_WIDTH +: IDX_WIDTH];
                if (dut_d !== golden_dist[i]) begin
                    $display("  [%0d] dist MISMATCH: DUT=%0d GOLDEN=%0d (idx DUT=%0d GOLDEN=%0d)",
                             i, dut_d, golden_dist[i], dut_i, golden_idx[i]);
                    err = err + 1;
                end else begin
                    if (i < 5)  // 只打印前5个
                        $display("  [%0d] dist=%0d idx=%0d ✓", i, dut_d, dut_i);
                end
            end
            if (err == 0)
                $display("  *** TC%0d PASS: %0d个结果全部匹配 ***", tc, k);
            else
                $display("  *** TC%0d FAIL: %0d个不匹配 ***", tc, err);
        end
    endtask

    // =========================================================================
    // 等待topk_valid
    // =========================================================================
    task wait_topk;
        integer timeout;
        begin
            timeout = 0;
            while (!topk_valid && timeout < 200) begin
                @(posedge clk); timeout = timeout + 1;
            end
            if (timeout >= 200)
                $display("[ERROR] topk_valid 超时！");
            else
                $display("[INFO] topk_valid 在第%0d拍出现（从start计）", timeout);
        end
    endtask

    integer i;
    integer start_cycle;

    initial begin
        $dumpfile("tb_dist_topk.vcd");
        $dumpvars(0, tb_dist_topk_engine);

        // 复位
        rst_n = 0; start = 0; k_sel = 1;
        q_x = 0; q_y = 0; q_z = 0;
        db_x_packed = 0; db_y_packed = 0; db_z_packed = 0;
        batch_base_idx = 0;
        repeat(5) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat(2) @(posedge clk);

        // ================================================================
        // TC1: Top-1, k=1
        // 数据库: DB[i]=(i*2, i*2, i*2), 查询点=(5,5,5)
        // 最近点: i=2 → dist²=3*(5-4)²=3, idx=2
        // ================================================================
        $display("\n====== TC1: Top-1 (k=1) ======");
        k_sel = 7'd1;
        q_x = 16'sd5; q_y = 16'sd5; q_z = 16'sd5;
        batch_base_idx = 10'd0;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = i * 2;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = i * 2;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = i * 2;
        end
        compute_golden(16'sd5, 16'sd5, 16'sd5, 1, 10'd0);

        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait_topk();
        @(posedge clk); #1;
        $display("  Top-1: dist²=%0d idx=%0d",
                 topk_dist[0 +: ACC_WIDTH], topk_idx[0 +: IDX_WIDTH]);
        $display("  Golden: dist²=%0d idx=%0d", golden_dist[0], golden_idx[0]);
        check_topk(1, 1);

        repeat(10) @(posedge clk);

        // ================================================================
        // TC2: Top-4, k=4
        // 数据库: DB[i]=(i,i,i), 查询=(10,10,10)
        // 最近4个: i=10(dist=0),i=9或11(dist=3),i=8或12(dist=12),...
        // ================================================================
        $display("\n====== TC2: Top-4 (k=4) ======");
        k_sel = 7'd4;
        q_x = 16'sd10; q_y = 16'sd10; q_z = 16'sd10;
        batch_base_idx = 10'd0;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = i[DATA_WIDTH-1:0];
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = i[DATA_WIDTH-1:0];
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = i[DATA_WIDTH-1:0];
        end
        compute_golden(16'sd10, 16'sd10, 16'sd10, 4, 10'd0);

        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait_topk();
        @(posedge clk); #1;
        check_topk(2, 4);

        repeat(10) @(posedge clk);

        // ================================================================
        // TC3: Top-8, k=8，负坐标
        // ================================================================
        $display("\n====== TC3: Top-8 (k=8), 负坐标 ======");
        k_sel = 7'd8;
        q_x = -16'sd20; q_y = -16'sd20; q_z = -16'sd20;
        batch_base_idx = 10'd0;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = -i;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = -i;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = -i;
        end
        compute_golden(-16'sd20, -16'sd20, -16'sd20, 8, 10'd0);

        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait_topk();
        @(posedge clk); #1;
        check_topk(3, 8);

        repeat(10) @(posedge clk);

        // ================================================================
        // TC4: Top-64, 全排序
        // ================================================================
        $display("\n====== TC4: Top-64 (全排序) ======");
        k_sel = 7'd64;
        q_x = 16'sd0; q_y = 16'sd0; q_z = 16'sd0;
        batch_base_idx = 10'd0;
        // 打乱顺序：DB[i] = (63-i, 63-i, 63-i)，期望输出升序
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = 63 - i;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = 63 - i;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = 63 - i;
        end
        compute_golden(16'sd0, 16'sd0, 16'sd0, 64, 10'd0);

        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait_topk();
        @(posedge clk); #1;
        check_topk(4, 64);

        repeat(10) @(posedge clk);

        // ================================================================
        // TC5: 连续两批次，验证 start 清零复位逻辑
        // ================================================================
        $display("\n====== TC5: 连续两批次, k=1 ======");
        k_sel = 7'd1;

        // 批次0: DB[i]=(100+i,0,0)，查询=(100,0,0)，最近是i=0 dist=0
        $display("  批次0: 期望 Top-1 idx=0, dist=0");
        q_x = 16'sd100; q_y = 16'sd0; q_z = 16'sd0;
        batch_base_idx = 10'd0;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = 100 + i;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'd0;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'd0;
        end
        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait_topk();
        @(posedge clk); #1;
        $display("  批次0结果: dist=%0d idx=%0d",
                 topk_dist[0 +: ACC_WIDTH], topk_idx[0 +: IDX_WIDTH]);

        repeat(5) @(posedge clk);

        // 批次1: DB[i]=(200+i,0,0)，查询=(200,0,0)，最近是i=0(全局idx=64) dist=0
        $display("  批次1: 期望 Top-1 idx=64, dist=0");
        q_x = 16'sd200; q_y = 16'sd0; q_z = 16'sd0;
        batch_base_idx = 10'd64;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = 200 + i;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'd0;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'd0;
        end
        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait_topk();
        @(posedge clk); #1;
        $display("  批次1结果: dist=%0d idx=%0d",
                 topk_dist[0 +: ACC_WIDTH], topk_idx[0 +: IDX_WIDTH]);
        if (topk_dist[0 +: ACC_WIDTH] === 0 && topk_idx[0 +: IDX_WIDTH] === 64)
            $display("  *** TC5 PASS ***");
        else
            $display("  *** TC5 FAIL ***");

        repeat(10) @(posedge clk);
        $display("\n====== 全部测试完成 ======\n");
        $finish;
    end

    initial begin
        #500000;
        $display("[WATCHDOG] 仿真超时！");
        $finish;
    end

    // 监视topk_valid
    always @(posedge clk) begin
        if (topk_valid)
            $display("[%0t ns] topk_valid! dist[0]=%0d idx[0]=%0d",
                     $time,
                     topk_dist[0 +: ACC_WIDTH],
                     topk_idx [0 +: IDX_WIDTH]);
    end

endmodule