
`timescale 1ns/1ps
`include "distance_engine.v"
`include "PE.v"
// =============================================================================
// File    : tb_distance_engine.v
// Testbench for 1×64 Systolic Array Distance Engine
//
// 验证脉动时序：
//   Cycle 1:  PE0 ← qx & k0x
//   Cycle 2:  PE0 ← qy & k0y,  PE1 ← qx & k1x
//   Cycle 3:  PE0 ← qz & k0z,  PE1 ← qy & k1y,  PE2 ← qx & k2x
//   Cycle 3后: PE0输出 dist²[0]
//   ...
//   Cycle 66: PE63输出 dist²[63]
// =============================================================================

`timescale 1ns/1ps

module tb_distance_engine;

    parameter PE_NUM     = 64;
    parameter DATA_WIDTH = 16;
    parameter ACC_WIDTH  = 48;
    parameter CLK_PERIOD = 10;

    // DUT ports
    reg                          clk, rst_n, start;
    reg  [DATA_WIDTH-1:0]        q_x, q_y, q_z;
    reg  [PE_NUM*DATA_WIDTH-1:0] db_x_packed, db_y_packed, db_z_packed;

    wire [ACC_WIDTH-1:0]         dist_sq_out;
    wire                         result_valid;
    wire [$clog2(PE_NUM)-1:0]    result_idx;
    wire                         all_done;

    // DUT
    distance_engine #(
        .PE_NUM    (PE_NUM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .q_x         (q_x),
        .q_y         (q_y),
        .q_z         (q_z),
        .db_x_packed (db_x_packed),
        .db_y_packed (db_y_packed),
        .db_z_packed (db_z_packed),
        .dist_sq_out (dist_sq_out),
        .result_valid(result_valid),
        .result_idx  (result_idx),
        .all_done    (all_done)
    );

    // Clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // 存放DUT输出结果和黄金参考
    reg [ACC_WIDTH-1:0] dut_results [0:PE_NUM-1];
    reg [ACC_WIDTH-1:0] golden      [0:PE_NUM-1];

    // 收集DUT结果
    always @(posedge clk) begin
        if (result_valid) begin
            dut_results[result_idx] <= dist_sq_out;
            $display("[Cycle %0t ns] PE[%0d] → dist²=%0d",
                     $time, result_idx, dist_sq_out);
        end
    end

    // 黄金模型计算任务
    task compute_golden_model;
        input signed [DATA_WIDTH-1:0] qx, qy, qz;
        integer idx;
        reg signed [DATA_WIDTH-1:0] dx, dy, dz;
        reg signed [DATA_WIDTH-1:0] kx, ky, kz;
        begin
            for (idx = 0; idx < PE_NUM; idx = idx + 1) begin
                kx = $signed(db_x_packed[idx*DATA_WIDTH +: DATA_WIDTH]);
                ky = $signed(db_y_packed[idx*DATA_WIDTH +: DATA_WIDTH]);
                kz = $signed(db_z_packed[idx*DATA_WIDTH +: DATA_WIDTH]);
                dx = qx - kx;
                dy = qy - ky;
                dz = qz - kz;
                golden[idx] = dx*dx + dy*dy + dz*dz;
            end
        end
    endtask

    // 检查结果任务
    task check_all;
        input integer tc;
        integer idx, err;
        begin
            err = 0;
            for (idx = 0; idx < PE_NUM; idx = idx + 1) begin
                if (dut_results[idx] !== golden[idx]) begin
                    $display("[TC%0d] MISMATCH PE[%0d]: DUT=%0d GOLDEN=%0d",
                             tc, idx, dut_results[idx], golden[idx]);
                    err = err + 1;
                end
            end
            if (err == 0)
                $display("[TC%0d] *** PASS: 全部 %0d 个结果与黄金模型一致 ***", tc, PE_NUM);
            else
                $display("[TC%0d] *** FAIL: %0d 个不匹配 ***", tc, err);
        end
    endtask

    // 等待全部完成任务
    task wait_done;
        integer timeout;
        begin
            timeout = 0;
            while (!all_done && timeout < 200) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 200)
                $display("[ERROR] all_done 等待超时！");
            else begin
                @(posedge clk); // 再等一拍确保输出稳定
                $display("[INFO] 计算完成，共耗时约 %0d 个周期", timeout+3);
            end
        end
    endtask

    integer i;

    initial begin
        $dumpfile("tb_distance_engine.vcd");
        $dumpvars(0, tb_distance_engine);

        // ---- 复位 ----
        rst_n = 0; start = 0;
        q_x = 0; q_y = 0; q_z = 0;
        db_x_packed = 0; db_y_packed = 0; db_z_packed = 0;
        repeat(4) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat(2) @(posedge clk);

        // ================================================================
        // TC1: 查询点(0,0,0)，数据库点均为(1,1,1)
        //      期望所有距离 = 1²+1²+1² = 3
        // ================================================================
        $display("\n========== TC1: query=(0,0,0), all DB=(1,1,1), expect dist²=3 ==========");
        q_x = 16'sd0; q_y = 16'sd0; q_z = 16'sd0;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd1;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd1;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd1;
        end
        compute_golden_model(16'sd0, 16'sd0, 16'sd0);

        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        wait_done();
        check_all(1);

        repeat(5) @(posedge clk);

        // ================================================================
        // TC2: 查询点(10,20,30)，数据库点 DB[i]=(i, i*2, i*3)
        //       期望 dist²[i] = (10-i)² + (20-2i)² + (30-3i)²
        // ================================================================
        $display("\n========== TC2: query=(10,20,30), DB[i]=(i,2i,3i) ==========");
        q_x = 16'sd10; q_y = 16'sd20; q_z = 16'sd30;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = i;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = i * 2;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = i * 3;
        end
        compute_golden_model(16'sd10, 16'sd20, 16'sd30);

        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        wait_done();
        // 打印前3个验证
        $display("  dist²[0]  DUT=%0d  GOLDEN=%0d", dut_results[0],  golden[0]);
        $display("  dist²[10] DUT=%0d  GOLDEN=%0d", dut_results[10], golden[10]);
        $display("  dist²[63] DUT=%0d  GOLDEN=%0d", dut_results[63], golden[63]);
        check_all(2);

        repeat(5) @(posedge clk);

        // ================================================================
        // TC3: 查询点 == 所有DB点，期望全部 dist²=0
        // ================================================================
        $display("\n========== TC3: query==all DB, expect dist²=0 ==========");
        q_x = 16'sd7; q_y = 16'sd8; q_z = 16'sd9;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd7;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd8;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd9;
        end
        compute_golden_model(16'sd7, 16'sd8, 16'sd9);

        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        wait_done();
        check_all(3);

        repeat(5) @(posedge clk);

        // ================================================================
        // TC4: 负坐标测试 — 验证有符号差值计算
        //       query=(-5,-5,-5), DB[i]=(-i,-i,-i)
        //       dist²[i] = 3*(i-5)²
        // ================================================================
        $display("\n========== TC4: 负坐标, query=(-5,-5,-5), DB[i]=(-i,-i,-i) ==========");
        q_x = -16'sd5; q_y = -16'sd5; q_z = -16'sd5;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = -i;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = -i;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = -i;
        end
        compute_golden_model(-16'sd5, -16'sd5, -16'sd5);

        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        wait_done();
        $display("  dist²[0] DUT=%0d GOLDEN=%0d (expect=%0d)",
                 dut_results[0], golden[0], 3*25);
        $display("  dist²[5] DUT=%0d GOLDEN=%0d (expect=0)",
                 dut_results[5], golden[5]);
        check_all(4);

        repeat(5) @(posedge clk);

        // ================================================================
        // TC5: 脉动时序验证 — 监视PE0,PE1,PE2的输出时间
        //       PE0 应在 Cycle 3 后第一个输出
        //       PE1 应在 Cycle 4 后输出（比PE0晚1拍）
        // ================================================================
        $display("\n========== TC5: 脉动时序验证 — 观察结果出现顺序 ==========");
        $display("  期望：PE0先出，PE1比PE0晚1周期，以此类推");
        q_x = 16'sd100; q_y = 16'sd0; q_z = 16'sd0;
        for (i = 0; i < PE_NUM; i = i + 1) begin
            db_x_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd0;
            db_y_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd0;
            db_z_packed[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd0;
        end
        // 期望: dist²[i] = 100² + 0 + 0 = 10000
        compute_golden_model(16'sd100, 16'sd0, 16'sd0);

        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        wait_done();
        check_all(5);

        $display("\n========== 全部测试完成 ==========\n");
        $finish;
    end

    // 超时保护
    initial begin
        #100000;
        $display("[WATCHDOG] 仿真超时！");
        $finish;
    end

endmodule