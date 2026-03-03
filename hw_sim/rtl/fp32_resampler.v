`timescale 1ns/1ps
`include "fp32_adder.v"
`include "fp32_mul.v"
// ============================================================================
// fp32_resampler.v  --  FP32 线性插值重采样器
// ============================================================================
//
// 功能: 将 N_IN 个 FP32 采样点通过线性插值重采样为 N_OUT 个点
//       同时支持下采样 (N_IN > N_OUT) 和上采样 (N_IN < N_OUT)
//
// 本文件包含:
//   fp32_lerp       -- 线性插值: y = y0 + (y1 - y0) * t
//   fp32_resampler  -- 顶层: 数据寄存器堆 + t 权重 LUT + 定点位置累加器 + FSM
//
// 插值权重 t 由 Python 预计算 (与 lin_resample_1d 一致):
//   src_pos = j * (L-1) / (out_len-1),  left = floor(src_pos),  w = src_pos - left
//   将 w 转为 FP32 十六进制, 写入 t_lut[j]. 与向量数据同时传入后 start 即可.
//
// step 仍用于计算 left 索引: step = floor((N_IN-1)*2^FRAC_W/(N_OUT-1)), pos += step, idx = pos 整数部分.
//
// 使用流程:
//   1) wr_en/wr_addr/wr_data 写入输入数据 mem[0..N_IN-1]
//   2) wr_t_en/wr_t_addr/wr_t_data 写入 t LUT t_lut[0..N_OUT-1] (Python 预生成 hex)
//   3) 配置 cfg_n_in, cfg_n_out, cfg_step, 拉高 start
//   4) 等待 out_valid 逐个读 out_data, done 表示结束
//
// 周期数 (start 拉高到 done 拉高):
//   - 进入: 1 周期 (IDLE -> F_START, 锁存配置)
//   - 每个输出点 j: 若 t_lut[j]==0 则 2 周期 (F_START->F_NEXT), 否则 17 周期 (F_START->F_WAIT 15 拍->F_NEXT)
//   - 总周期 = 1 + sum_{j=0}^{N_OUT-1} (t_j==0 ? 2 : 17) = 1 + 17*N_OUT - 15*(t==0 的点数)
//   - 全需插值时: 1 + 17*N_OUT; 全整数位置时: 1 + 2*N_OUT
// 当前为单 LERP、顺序 FSM，无输出间流水 (见 README 说明)。
// ============================================================================


// ============================================================================
// fp32_lerp
// 浮点线性插值: y = y0 + (y1 - y0) * t
// 内部实例化 fp32_adder (减法→加法) 和 fp32_mul
// 顺序 FSM 控制, 三步运算:
//   Step 1: diff = y1 - y0   (fp32_adder, sel=1)
//   Step 2: prod = diff * t  (fp32_mul)
//   Step 3: result = y0 + prod  (fp32_adder, sel=0)
// 每步约 5 周期 (4周期流水延迟 + 1周期 NB传播), 总计 ~15 周期
// ============================================================================
module fp32_lerp (
    input  wire        clk,
    input  wire        rstn,
    input  wire        start,       // 单周期脉冲启动
    input  wire [31:0] y0,          // 左端点值 (FP32)
    input  wire [31:0] y1,          // 右端点值 (FP32)
    input  wire [31:0] t,           // 插值权重 [0, 1) (FP32)
    output reg  [31:0] result,      // 插值结果
    output reg         done         // 单周期脉冲, 结果就绪
);

    localparam [1:0] S_IDLE = 2'd0,
                     S_SUB  = 2'd1,
                     S_MUL  = 2'd2,
                     S_ADD  = 2'd3;

    reg [1:0]  state;
    reg [31:0] y0_r, t_r;

    // fp32_adder 接口
    reg         add_ena;
    reg  [31:0] add_a, add_b;
    reg         add_sel;
    wire [31:0] add_s;
    wire        add_vld;

    // fp32_mul 接口
    reg         mul_ena;
    reg  [31:0] mul_a, mul_b;
    wire [31:0] mul_s;
    wire        mul_vld;

    fp32_adder u_add (
        .clk(clk), .rstn(rstn), .ena(add_ena),
        .a(add_a), .b(add_b), .rm(2'b00), .sel(add_sel),
        .s(add_s), .valid(add_vld)
    );

    fp32_mul u_mul (
        .clk(clk), .rstn(rstn), .ena(mul_ena),
        .a(mul_a), .b(mul_b), .rm(2'b00),
        .s(mul_s), .valid(mul_vld)
    );

    localparam LERP_LAT = 5;
    reg [2:0] lerp_cnt;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state   <= S_IDLE;
            lerp_cnt<= 3'b0;
            done    <= 1'b0;
            add_ena <= 1'b0;
            mul_ena <= 1'b0;
            result  <= 32'b0;
            y0_r    <= 32'b0;
            t_r     <= 32'b0;
            add_a   <= 32'b0;
            add_b   <= 32'b0;
            add_sel <= 1'b0;
            mul_a   <= 32'b0;
            mul_b   <= 32'b0;
        end else begin
            done    <= 1'b0;
            add_ena <= 1'b0;
            mul_ena <= 1'b0;

            case (state)
                S_IDLE: begin
                    lerp_cnt <= 3'b0;
                    if (start) begin
                        y0_r    <= y0;
                        t_r     <= t;
                        add_a   <= y1;
                        add_b   <= y0;
                        add_sel <= 1'b1;
                        add_ena <= 1'b1;
                        state   <= S_SUB;
                    end
                end

                S_SUB: begin
                    if (lerp_cnt == LERP_LAT - 1) begin
                        mul_a   <= add_s;
                        mul_b   <= t_r;
                        mul_ena <= 1'b1;
                        lerp_cnt<= 3'b0;
                        state   <= S_MUL;
                    end else
                        lerp_cnt <= lerp_cnt + 1'b1;
                end

                S_MUL: begin
                    if (lerp_cnt == LERP_LAT - 1) begin
                        add_a   <= y0_r;
                        add_b   <= mul_s;
                        add_sel <= 1'b0;
                        add_ena <= 1'b1;
                        lerp_cnt<= 3'b0;
                        state   <= S_ADD;
                    end else
                        lerp_cnt <= lerp_cnt + 1'b1;
                end

                S_ADD: begin
                    if (lerp_cnt == LERP_LAT - 1) begin
                        result <= add_s;
                        done   <= 1'b1;
                        state  <= S_IDLE;
                    end else
                        lerp_cnt <= lerp_cnt + 1'b1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule


// ============================================================================
// fp32_resampler
// 顶层重采样器: 写入 N_IN 个 FP32 点 → 输出 N_OUT 个线性插值点
//
// 地址生成: 定点累加器 (ADDR_W.FRAC_W 格式)
//   pos 从 0 开始, 每输出一个点递增 step
//   整数部分 idx 作为寄存器堆读地址
//   小数部分 frac 经 frac16_to_fp32 转为 FP32 权重 t
//
// 数据存储: 寄存器堆 (组合读, 适合 N_IN_MAX ≤ 128)
//   若 N_IN_MAX > 128, 建议改为块 RAM + 同步读 (多 1 周期延迟)
//
// 参数:
//   N_IN_MAX  -- 寄存器堆深度 (≥ 实际 N_IN)
//   N_OUT_MAX -- 仅用于文档, 不影响硬件
//   ADDR_W    -- 地址/计数位宽 (≥ clog2(max(N_IN_MAX, N_OUT_MAX)))
//   FRAC_W    -- 定点步长小数精度
// ============================================================================

module fp32_resampler #(
    parameter N_IN_MAX  = 64,
    parameter N_OUT_MAX = 512,
    parameter ADDR_W    = 10,
    parameter FRAC_W    = 16
)(
    input  wire                        clk,
    input  wire                        rstn,

    // 配置 (在 start 拉高前设好, start 时锁存)
    input  wire [ADDR_W-1:0]           cfg_n_in,       // 实际输入样本数 (≥ 2)
    input  wire [ADDR_W-1:0]           cfg_n_out,      // 期望输出样本数 (≥ 1)
    input  wire [ADDR_W+FRAC_W-1:0]   cfg_step,       // 定点步长 (用于 left 索引)

    // 写入输入数据 mem[0..N_IN-1]
    input  wire                        wr_en,
    input  wire [ADDR_W-1:0]           wr_addr,
    input  wire [31:0]                 wr_data,

    // 写入 t 权重 LUT (Python 预生成 FP32 hex, 与 lin_resample_1d 的 w 一致)
    input  wire                        wr_t_en,
    input  wire [ADDR_W-1:0]           wr_t_addr,
    input  wire [31:0]                 wr_t_data,

    // 控制
    input  wire                        start,

    // 输出流
    output reg  [31:0]                 out_data,
    output reg                         out_valid,
    output reg                         busy,
    output reg                         done
);

    // ----------------------------------------------------------------
    // 输入数据寄存器堆
    // ----------------------------------------------------------------
    reg [31:0] mem [0:N_IN_MAX-1];
    reg [ADDR_W-1:0] out_cnt;
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    // ----------------------------------------------------------------
    // t 权重 LUT: 预计算 w = src_pos - left (FP32 hex), 由 Python 生成后写入
    // ----------------------------------------------------------------
    reg [31:0] t_lut [0:N_OUT_MAX-1];

    always @(posedge clk) begin
        if (wr_t_en)
            t_lut[wr_t_addr] <= wr_t_data;
    end

    // ----------------------------------------------------------------
    // 配置锁存
    // ----------------------------------------------------------------
    reg [ADDR_W-1:0]         n_in_r, n_out_r;
    reg [ADDR_W+FRAC_W-1:0]  step_r;

    // ----------------------------------------------------------------
    // 定点位置累加器: 整数部分为 left 索引, t 来自 LUT 不在此计算
    // ----------------------------------------------------------------
    reg  [ADDR_W+FRAC_W-1:0] pos;
    wire [ADDR_W-1:0]         idx  = pos[ADDR_W+FRAC_W-1 : FRAC_W];

    wire [ADDR_W-1:0] n_in_m1 = n_in_r - 1'b1;
    wire [ADDR_W-1:0] idx1    = (idx >= n_in_m1) ? n_in_m1 : (idx + 1'b1);

    // t 由 Python 预生成, 写入 t_lut, 按输出序号查表
    wire [31:0] t_fp32 = t_lut[out_cnt];

    // 最后一拍：step 累加可能达不到 (L-1)<<FRAC_W，强制用右端点索引与 lin_resample_1d 一致
    wire [ADDR_W-1:0] idx_eff  = (out_cnt == n_out_r - 1'b1) ? n_in_m1 : idx;
    wire [ADDR_W-1:0] idx1_eff = (idx_eff >= n_in_m1) ? n_in_m1 : (idx_eff + 1'b1);

    // ----------------------------------------------------------------
    // 数据读取
    // ----------------------------------------------------------------
    wire [31:0] y0_w = mem[idx_eff];
    wire [31:0] y1_w = mem[idx1_eff];

    // ----------------------------------------------------------------
    // LERP 单元
    // ----------------------------------------------------------------
    reg         lerp_go;
    wire [31:0] lerp_res;
    wire        lerp_done;

    fp32_lerp u_lerp (
        .clk   (clk),
        .rstn  (rstn),
        .start (lerp_go),
        .y0    (y0_w),
        .y1    (y1_w),
        .t     (t_fp32),
        .result(lerp_res),
        .done  (lerp_done)
    );

    // ----------------------------------------------------------------
    // 输出计数器
    // ----------------------------------------------------------------
    

    // ----------------------------------------------------------------
    // 主状态机
    //   IDLE  等待 start
    //   START 判断是否为整数位置 (跳过LERP优化) 或启动LERP
    //   WAIT  等待 LERP 完成
    //   NEXT  输出完毕判断, 推进到下一点或结束
    // ----------------------------------------------------------------
    localparam [2:0] F_IDLE  = 3'd0,
                     F_START = 3'd1,
                     F_WAIT  = 3'd2,
                     F_NEXT  = 3'd3;

    reg [2:0] fsm;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            fsm       <= F_IDLE;
            pos       <= {(ADDR_W+FRAC_W){1'b0}};
            out_cnt   <= {ADDR_W{1'b0}};
            lerp_go   <= 1'b0;
            out_data  <= 32'b0;
            out_valid <= 1'b0;
            busy      <= 1'b0;
            done      <= 1'b0;
            n_in_r    <= {ADDR_W{1'b0}};
            n_out_r   <= {ADDR_W{1'b0}};
            step_r    <= {(ADDR_W+FRAC_W){1'b0}};
        end else begin
            lerp_go   <= 1'b0;
            out_valid <= 1'b0;
            done      <= 1'b0;

            case (fsm)
                F_IDLE: begin
                    if (start) begin
                        n_in_r  <= cfg_n_in;
                        n_out_r <= cfg_n_out;
                        step_r  <= cfg_step;
                        pos     <= {(ADDR_W+FRAC_W){1'b0}};
                        out_cnt <= {ADDR_W{1'b0}};
                        busy    <= 1'b1;
                        fsm     <= F_START;
                    end
                end

                F_START: begin
                    if (t_fp32 == 32'b0) begin
                        // t==0 即整数位置 直接输出 y0, 无需 LERP
                        out_data  <= y0_w;
                        out_valid <= 1'b1;
                        fsm       <= F_NEXT;
                    end else begin
                        lerp_go <= 1'b1;
                        fsm     <= F_WAIT;
                    end
                end

                F_WAIT: begin
                    if (lerp_done) begin
                        out_data  <= lerp_res;
                        out_valid <= 1'b1;
                        fsm       <= F_NEXT;
                    end
                end

                F_NEXT: begin
                    if (out_cnt >= n_out_r - 1'b1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        fsm  <= F_IDLE;
                    end else begin
                        pos     <= pos + step_r;
                        out_cnt <= out_cnt + 1'b1;
                        fsm     <= F_START;
                    end
                end

                default: fsm <= F_IDLE;
            endcase
        end
    end

endmodule
