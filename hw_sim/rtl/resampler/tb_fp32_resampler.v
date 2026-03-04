`timescale 1ns/1ps
// ============================================================================
// tb_fp32_resampler.v -- 参数化验证：N_IN / N_OUT 决定 DUT 深度与 hex 行数一致
// ============================================================================
// 单次验证（推荐）：
//   先生成 hex（行数 = N_IN / N_OUT）:
//     python ref_resampler.py <N_IN> <N_OUT> -o gold.hex --t-lut-out t_lut.hex
//     python ref_resampler.py <N_IN> <N_OUT> --in-float "..." 若需自定义输入，并准备 in.hex（N_IN 行）
//   编译时传入参数，DUT 深度 = N_IN / N_OUT，无 $readmemh 告警:
//     iverilog -D N_IN=5 -D N_OUT=3 -o sim.vvp -s tb_fp32_resampler tb_fp32_resampler.v
//     vvp sim.vvp
//   多组验证可多次运行，例如:
//     iverilog -D N_IN=5 -D N_OUT=3 ... && vvp sim.vvp
//     iverilog -D N_IN=3 -D N_OUT=5 ... && vvp sim.vvp
//     iverilog -D N_IN=2 -D N_OUT=4 ... && vvp sim.vvp
//
// 一次跑三组用例: iverilog -D RUN_ALL -o sim.vvp ... && vvp sim.vvp
//   DUT 使用 N_IN_MAX=5, N_OUT_MAX=5；hex 需为 5 行以匹配深度（不足则 --t-lut-pad 5 --gold-pad 5 生成）
//   需存在 in_5.hex(5行), in_2.hex(5行), t_lut_5_3/3_5/2_4.hex(各5行), gold_5_3/gold_3_5/gold_2_4.hex(各5行)
//
//   51 点下采样到 32、64 点上采样到 461 (512-51)
//   Python 生成 hex（默认输入 1.0..L；--in-out 写出 in.hex 供 TB）:
//     python ref_resampler.py 51 32 -o gold.hex --t-lut-out t_lut.hex --in-out in.hex --quiet
//     iverilog -D N_IN=51 -D N_OUT=32 -D GOLDEN_ULP_TOL -o sim.vvp -s tb_fp32_resampler tb_fp32_resampler.v && vvp sim.vvp
//     python ref_resampler.py 64 461 -o gold.hex --t-lut-out t_lut.hex --in-out in.hex --quiet
//     iverilog -D N_IN=64 -D N_OUT=461 -D GOLDEN_ULP_TOL -o sim.vvp -s tb_fp32_resampler tb_fp32_resampler.v && vvp sim.vvp
//   一次跑两组可用 run_51_32_64_461.bat（先 51→32 再 64→461，每次会覆盖 in/t_lut/gold.hex）
//
// 64→461 等长链时，若出现个别点差约 1.0（如 out[387] expect 54.002 got 53.002）：
//   原因是 RTL 用整数 step=floor((L-1)*2^16/(N-1)) 累加，与 Python 的 src_pos=j*(L-1)/(N-1)
//   在边界上可能差 1 个索引（left 差 1），导致插值结果差约 1.0（不是 1 ULP）。
//   -D GOLDEN_ULP_TOL 只允许 1 ULP，无法覆盖此类差 1.0 的情况；可查 RTL 该拍 idx/pos 排查。
// ============================================================================
`include "fp32_resampler.v"
module tb_fp32_resampler;

    localparam ADDR_W   = 10;
    localparam FRAC_W   = 16;
    localparam STEP_W   = ADDR_W + FRAC_W;

    // 由编译 +define 或默认值决定本次验证的输入/输出点数
    `ifdef N_IN
    localparam N_IN_P   = `N_IN;
    `else
    localparam N_IN_P   = 5;
    `endif
    `ifdef N_OUT
    localparam N_OUT_P  = `N_OUT;
    `else
    localparam N_OUT_P  = 3;
    `endif

    // RUN_ALL 时用固定深度 5/5 跑三组用例
    `ifdef RUN_ALL
    localparam N_IN_MAX_TB  = 5;
    localparam N_OUT_MAX_TB = 5;
    `else
    localparam N_IN_MAX_TB  = N_IN_P;
    localparam N_OUT_MAX_TB = N_OUT_P;
    `endif

    reg                        clk;
    reg                        rstn;
    reg  [ADDR_W-1:0]          cfg_n_in;
    reg  [ADDR_W-1:0]          cfg_n_out;
    reg  [STEP_W-1:0]          cfg_step;
    reg                        start;
    // 本 TB 用 $readmemh 载入 mem / t_lut，不通过写端口；DUT 仍保留 wr_* / wr_t_* 端口供上板/集成使用
    wire [31:0]                out_data;
    wire                       out_valid;
    wire                       busy;
    wire                       done;

    fp32_resampler #(
        .N_IN_MAX  (N_IN_MAX_TB),
        .N_OUT_MAX (N_OUT_MAX_TB),
        .ADDR_W    (ADDR_W),
        .FRAC_W    (FRAC_W)
    ) u_dut (
        .clk       (clk),
        .rstn      (rstn),
        .cfg_n_in  (cfg_n_in),
        .cfg_n_out (cfg_n_out),
        .cfg_step  (cfg_step),
        .wr_en     (1'b0),
        .wr_addr   ({ADDR_W{1'b0}}),
        .wr_data   (32'b0),
        .wr_t_en   (1'b0),
        .wr_t_addr ({ADDR_W{1'b0}}),
        .wr_t_data (32'b0),
        .start     (start),
        .out_data  (out_data),
        .out_valid (out_valid),
        .busy      (busy),
        .done      (done)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("fp32_resampler.vcd");
        $dumpvars(0, tb_fp32_resampler);
    end

    function [STEP_W-1:0] calc_step;
        input [ADDR_W-1:0] n_in;
        input [ADDR_W-1:0] n_out;
        reg [31:0] num, den, q;
        begin
            num = (n_in - 1) * (1 << FRAC_W);
            den = (n_out - 1);
            if (den == 0)
                q = 0;
            else
                q = num / den;
            calc_step = q[STEP_W-1:0];
        end
    endfunction

    integer i, err;
    reg [31:0] gold [0:511];
    reg [31:0] gold_ref [0:N_OUT_MAX_TB-1];

    `ifdef RUN_ALL
    // -------------------------------------------------------------------------
    // RUN_ALL: 一次跑三组用例 (5→3, 3→5, 2→4)，DUT 深度 5/5
    // -------------------------------------------------------------------------
    integer out_count;
    initial begin
        rstn = 0; cfg_n_in = 0; cfg_n_out = 0; cfg_step = 0; start = 0; err = 0;
        #100; rstn = 1; #20;

        // Test 1: 5 -> 3
        $readmemh("t_lut_5_3.hex", u_dut.t_lut);
        $readmemh("gold_5_3.hex", gold_ref);
        $display("[TB] === Test 1: N_IN=5 N_OUT=3 ===");
        cfg_n_in = 5; cfg_n_out = 3; cfg_step = calc_step(5, 3);
        $readmemh("in_5.hex", u_dut.mem);
        repeat(2) @(posedge clk); start = 1; repeat(2) @(posedge clk); start = 0;
        out_count = 0;
        while (!done) begin wait (out_valid); gold[out_count] = out_data; $display("[TB]   out[%0d] = %h", out_count, out_data); out_count = out_count + 1; wait (!out_valid); end
        @(posedge clk);
        if (out_count != 3) begin $display("[TB] FAIL: expected 3, got %0d", out_count); err = err + 1; end
        else for (i = 0; i < 3; i = i + 1) if (gold[i] !== gold_ref[i]) begin $display("[TB] FAIL: out[%0d] expect %h got %h", i, gold_ref[i], gold[i]); err = err + 1; end
        if (err == 0) $display("[TB] Test 1 PASS."); #50;

        // Test 2: 3 -> 5
        $readmemh("t_lut_3_5.hex", u_dut.t_lut);
        $readmemh("gold_3_5.hex", gold_ref);
        $display("[TB] === Test 2: N_IN=3 N_OUT=5 ===");
        cfg_n_in = 3; cfg_n_out = 5; cfg_step = calc_step(3, 5);
        $readmemh("in_3.hex", u_dut.mem);
        repeat(2) @(posedge clk); start = 1; repeat(2) @(posedge clk); start = 0;
        out_count = 0;
        while (!done) begin wait (out_valid); gold[out_count] = out_data; $display("[TB]   out[%0d] = %h", out_count, out_data); out_count = out_count + 1; wait (!out_valid); end
        @(posedge clk);
        if (out_count != 5) begin $display("[TB] FAIL: expected 5, got %0d", out_count); err = err + 1; end
        else for (i = 0; i < 5; i = i + 1) if (gold[i] !== gold_ref[i]) begin $display("[TB] FAIL: out[%0d] expect %h got %h", i, gold_ref[i], gold[i]); err = err + 1; end
        if (err == 0) $display("[TB] Test 2 PASS."); #50;

        // Test 3: 2 -> 4
        $readmemh("t_lut_2_4.hex", u_dut.t_lut);
        $readmemh("gold_2_4.hex", gold_ref);
        $readmemh("in_2.hex", u_dut.mem);
        $display("[TB] === Test 3: N_IN=2 N_OUT=4 ===");
        cfg_n_in = 2; cfg_n_out = 4; cfg_step = calc_step(2, 4);
        repeat(2) @(posedge clk); start = 1; repeat(2) @(posedge clk); start = 0;
        out_count = 0;
        while (!done) begin wait (out_valid); gold[out_count] = out_data; $display("[TB]   out[%0d] = %h", out_count, out_data); out_count = out_count + 1; wait (!out_valid); end
        if (out_count != 4) begin $display("[TB] FAIL: expected 4, got %0d", out_count); err = err + 1; end
        else for (i = 0; i < 4; i = i + 1) if (gold[i] !== gold_ref[i]) begin $display("[TB] FAIL: out[%0d] expect %h got %h", i, gold_ref[i], gold[i]); err = err + 1; end
        if (err == 0) $display("[TB] Test 3 PASS.");

        if (err == 0) $display("[TB] ========== All tests PASSED ==========");
        else $display("[TB] ========== %0d test(s) FAILED ==========", err);
        $finish;
    end

    `else
    // -------------------------------------------------------------------------
    // 单次参数化验证: N_IN_P, N_OUT_P；hex 固定名 in.hex, t_lut.hex, gold.hex
    // -------------------------------------------------------------------------
    integer out_count;
    initial begin
        rstn = 0; cfg_n_in = 0; cfg_n_out = 0; cfg_step = 0; start = 0; err = 0;
        #100; rstn = 1; #20;

        $display("[TB] === Resampler N_IN=%0d N_OUT=%0d ===", N_IN_P, N_OUT_P);
        $readmemh("t_lut.hex", u_dut.t_lut);
        $readmemh("gold.hex", gold_ref);
        cfg_n_in  = N_IN_P;
        cfg_n_out = N_OUT_P;
        cfg_step  = calc_step(N_IN_P, N_OUT_P);
        $readmemh("in.hex", u_dut.mem);
        repeat(2) @(posedge clk);
        start = 1;
        repeat(2) @(posedge clk);
        start = 0;

        out_count = 0;
        while (!done) begin
            wait (out_valid);
            gold[out_count] = out_data;
            $display("[TB]   out[%0d] = %h", out_count, out_data);
            if (out_data[31] === 1'bx || out_data[30] === 1'bx) begin $display("[TB] WARN: out[%0d] has X", out_count); err = err + 1; end
            out_count = out_count + 1;
            wait (!out_valid);
        end
        @(posedge clk);

        if (out_count != N_OUT_P) begin
            $display("[TB] FAIL: expected %0d outputs, got %0d", N_OUT_P, out_count);
            err = err + 1;
        end else begin
            for (i = 0; i < N_OUT_P; i = i + 1)
                if (gold[i] !== gold_ref[i]) begin
                    `ifdef GOLDEN_ULP_TOL
                    if ((gold[i] === gold_ref[i] + 32'd1) || (gold[i] === gold_ref[i] - 32'd1)) ; // 允许 1 ULP
                    else begin
                    `endif
                        $display("[TB] FAIL: out[%0d] expect %h (ref) got %h", i, gold_ref[i], gold[i]);
                        err = err + 1;
                    `ifdef GOLDEN_ULP_TOL
                    end
                    `endif
                end
            if (err == 0) $display("[TB] PASS (vs gold.hex).");
        end

        if (err == 0) $display("[TB] ========== PASSED ==========");
        else $display("[TB] ========== FAILED ==========");
        $finish;
    end
    `endif

endmodule
