// =============================================================================
// File    : distance_engine.v
// Module  : distance_engine — 1×64 真正脉动阵列距离计算引擎
//
// 脉动时序示意（PE_NUM=64，计算64个键集距离）：
//
//  Cycle | PE0          | PE1          | PE2          | ... | PE63
//  ------+--------------+--------------+--------------+-----+--------
//    1   | qx vs k1x(X) |              |              |     |
//    2   | qy vs k1y(Y) | qx vs k2x(X)|              |     |
//    3   | qz vs k1z(Z) | qy vs k2y(Y)| qx vs k3x(X)|     |
//          → PE0输出 dist²[0]
//    4   |              | qz vs k2z(Z)| qy vs k3y(Y)|     |
//          → PE1输出 dist²[1]
//    ...
//   66   |              |              |              |     | qz vs k64z(Z)
//          → PE63输出 dist²[63]
//
// 共 64+2 = 66 个周期完成64个距离计算
//
// 控制器职责：
//   - 在 Cycle t，向 PE(t-1) 喂入 q_coord（X/Y/Z 由PE内部dim_cnt决定）
//     并同时向 PE(t-1) 喂入对应的 d_coord[t-1][dim]
//   - q_coord 通过链式脉动自动到达各PE，控制器只需驱动 PE0 入口
//   - d_coord 按时间对角线规律由控制器直接寻址喂给每个PE
//
// 数据库坐标寻址规律：
//   在 Cycle t（1-indexed），PE[i] 处于维度 dim = (t - i - 1) % 3（i从0开始）
//   PE[i] 需要的数据库点索引 = i（固定，每次只算一个键集点对应PE编号）
//   但坐标维度随周期变化：
//     PE[i] 第一次有效周期 = i+1，此时 dim=0(X)
//     PE[i] 第二次有效周期 = i+2，此时 dim=1(Y)
//     PE[i] 第三次有效周期 = i+3，此时 dim=2(Z)
//
// 因此控制器需要在每个周期为每个PE提供正确维度的坐标。
// 由于PE内部有dim_cnt，控制器只需按 [周期, PE编号] 的对角线规律送坐标即可。
// =============================================================================

module distance_engine #(
    parameter PE_NUM     = 64,
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 48
)(
    input  wire                             clk,
    input  wire                             rst_n,

    // 控制
    input  wire                             start,      // 脉冲：开始新一轮计算

    // 查询点坐标（在start时锁存，全程保持）
    input  wire [DATA_WIDTH-1:0]            q_x,
    input  wire [DATA_WIDTH-1:0]            q_y,
    input  wire [DATA_WIDTH-1:0]            q_z,

    // 数据库点坐标（所有64个点的三维坐标，预先加载）
    // 打包格式：db_x[i] = db_x_packed[i*DATA_WIDTH +: DATA_WIDTH]
    input  wire [PE_NUM*DATA_WIDTH-1:0]     db_x_packed,
    input  wire [PE_NUM*DATA_WIDTH-1:0]     db_y_packed,
    input  wire [PE_NUM*DATA_WIDTH-1:0]     db_z_packed,

    // 输出：每个PE完成时依次拉高 result_valid，result_idx 指示哪个PE出结果
    output wire [ACC_WIDTH-1:0]             dist_sq_out,    // 当前有效的距离平方
    output wire                             result_valid,   // 某个PE结果有效
    output wire [$clog2(PE_NUM)-1:0]        result_idx,     // 结果对应的点编号
    output wire                             all_done        // 所有64个距离计算完毕
);

    // =========================================================================
    // 内部信号：PE链
    // =========================================================================
    wire                   q_valid_chain [0:PE_NUM];   // [0]是入口，[PE_NUM]悬空
    wire [DATA_WIDTH-1:0]  q_coord_chain [0:PE_NUM];
    wire [ACC_WIDTH-1:0]   pe_dist_sq    [0:PE_NUM-1];
    wire                   pe_result_valid[0:PE_NUM-1];

    // =========================================================================
    // 全局周期计数器（从 start 后开始）
    // =========================================================================
    reg  [7:0]  cycle_cnt;      // 最大 PE_NUM+2 = 66，8位足够
    reg         running;

    // 锁存查询点
    reg [DATA_WIDTH-1:0] qx_r, qy_r, qz_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_cnt <= 8'd0;
            running   <= 1'b0;
            qx_r      <= {DATA_WIDTH{1'b0}};
            qy_r      <= {DATA_WIDTH{1'b0}};
            qz_r      <= {DATA_WIDTH{1'b0}};
        end else begin
            if (start && !running) begin
                running   <= 1'b1;
                cycle_cnt <= 8'd1;
                qx_r      <= q_x;
                qy_r      <= q_y;
                qz_r      <= q_z;
            end else if (running) begin
                if (cycle_cnt == PE_NUM + 2)  // 66 cycles done
                    running <= 1'b0;
                else
                    cycle_cnt <= cycle_cnt + 8'd1;
            end
        end
    end

    // =========================================================================
    // 控制器：产生 PE0 入口的 q_valid 和 q_coord
    //
    // 脉动入口规律：
    //   Cycle t（1~PE_NUM），向PE链注入一次有效 q_coord
    //   注入的是X坐标（PE0→PE1→... 每次+1拍延迟，各自在对的周期处理X）
    //   Y、Z 坐标由控制器在后续周期继续注入 PE0 入口
    //
    // 注意：每个周期只注入一个坐标到PE0，该坐标沿链脉动。
    //   - Cycle 1: 注入 qx → PE0处理qx(dim=0)
    //   - Cycle 2: 注入 qy → PE0处理qy(dim=1)；同时qx脉动到PE1，PE1处理qx(dim=0)
    //   - Cycle 3: 注入 qz → PE0处理qz(dim=2)；qy→PE1(dim=1)；qx→PE2(dim=0)
    //   - Cycle 4: 注入 qx（下一轮，但此处只有64个点，Cycle 4开始PE0无新任务）
    //              → 可以注入 "无效" 或直接停止
    //   但对于 Cycle 4~66，PE0 入口不再注入新数据（后续周期Y/Z沿链传递）
    //
    // 实际上：PE0 只需要在 Cycle1/2/3 被喂入 qx/qy/qz，
    //         之后 Cycle4 开始 PE0 无新任务，但PE1~PE63 仍在运行
    // =========================================================================

    // PE0 入口的 valid 和 coord
    reg                   q0_valid;
    reg [DATA_WIDTH-1:0]  q0_coord;

    always @(*) begin
        case (cycle_cnt)
            8'd1:    begin q0_valid = running; q0_coord = qx_r; end
            8'd2:    begin q0_valid = running; q0_coord = qy_r; end
            8'd3:    begin q0_valid = running; q0_coord = qz_r; end
            default: begin q0_valid = 1'b0;    q0_coord = {DATA_WIDTH{1'b0}}; end
        endcase
    end

    assign q_valid_chain[0] = q0_valid;
    assign q_coord_chain[0] = q0_coord;

    // =========================================================================
    // 数据库坐标对角线寻址
    //
    // 在 Cycle t，PE[i] 有效当且仅当 t >= i+1 且 t <= i+3
    // PE[i] 在以下周期有效：
    //   t = i+1 → dim=0, 需要 db_x[i]
    //   t = i+2 → dim=1, 需要 db_y[i]
    //   t = i+3 → dim=2, 需要 db_z[i]
    //
    // 控制器在每个 Cycle t，为所有 PE[i] 计算其需要的坐标维度：
    //   dim_for_pe[i] = t - i - 1  （0,1,2 分别对应 X,Y,Z）
    //   当 0 <= t-i-1 <= 2 时 PE[i] 有效
    // =========================================================================

    genvar gi;
    generate
        for (gi = 0; gi < PE_NUM; gi = gi + 1) begin : gen_pe_chain

            // 为PE[gi]选择当前周期应喂入的坐标维度
            // dim = cycle_cnt - gi - 1
            // 有效范围：cycle_cnt in [gi+1, gi+3]
            wire        pe_d_sel_valid;
            wire [1:0]  pe_dim;
            wire [DATA_WIDTH-1:0] pe_d_coord;

            assign pe_d_sel_valid = running &&
                                    (cycle_cnt >= (gi + 1)) &&
                                    (cycle_cnt <= (gi + 3));

            assign pe_dim = cycle_cnt - gi - 1;  // 0,1,2

            assign pe_d_coord = (pe_dim == 2'd0) ? db_x_packed[gi*DATA_WIDTH +: DATA_WIDTH] :
                                (pe_dim == 2'd1) ? db_y_packed[gi*DATA_WIDTH +: DATA_WIDTH] :
                                                   db_z_packed[gi*DATA_WIDTH +: DATA_WIDTH];

            pe #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH (ACC_WIDTH)
            ) u_pe (
                .clk          (clk),
                .rst_n        (rst_n),
                .q_valid_in   (q_valid_chain[gi]),
                .q_coord_in   (q_coord_chain[gi]),
                .q_valid_out  (q_valid_chain[gi+1]),
                .q_coord_out  (q_coord_chain[gi+1]),
                .d_coord      (pe_d_coord),
                .dist_sq_out  (pe_dist_sq[gi]),
                .result_valid (pe_result_valid[gi])
            );
        end
    endgenerate

    // =========================================================================
    // 输出仲裁：result_valid 依次从 PE0→PE63 轮流拉高（不会同时拉高）
    // =========================================================================
    // 用 OR 树汇总（因同一周期只有一个PE会拉高 result_valid）
    reg [ACC_WIDTH-1:0]          dist_sq_out_r;
    reg                          result_valid_r;
    reg [$clog2(PE_NUM)-1:0]     result_idx_r;

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_sq_out_r  <= {ACC_WIDTH{1'b0}};
            result_valid_r <= 1'b0;
            result_idx_r   <= {$clog2(PE_NUM){1'b0}};
        end else begin
            result_valid_r <= 1'b0;
            for (k = 0; k < PE_NUM; k = k + 1) begin
                if (pe_result_valid[k]) begin
                    dist_sq_out_r  <= pe_dist_sq[k];
                    result_valid_r <= 1'b1;
                    result_idx_r   <= k[$clog2(PE_NUM)-1:0];
                end
            end
        end
    end

    assign dist_sq_out  = dist_sq_out_r;
    assign result_valid = result_valid_r;
    assign result_idx   = result_idx_r;

    // all_done：PE63 输出有效后一拍
    reg done_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_r <= 1'b0;
        else
            done_r <= pe_result_valid[PE_NUM-1];
    end
    assign all_done = done_r;

endmodule