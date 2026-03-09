// =============================================================================
// File    : dist_topk_engine.v
// Module  : dist_topk_engine
// Desc    : 完整系统顶层：1×64脉动距离阵列 + 64个CMP单元 + 6层归约树
//
// 整体流水线时序（64个数据库点，单次查询）：
//
//  Phase 1 [Cycle 1~66]:   脉动阵列计算64个距离，PE[i]在Cycle(i+3)输出
//                           CMP单元逐个锁存（无需等待，立即存储）
//  Phase 2 [Cycle 67]:     all_locked拉高，触发 start_reduce
//  Phase 3 [Cycle 67~72]:  6层归约树流水，Cycle 73输出Top-K结果
//
//  总延迟: 66 + 1 + 6 = 73 个周期
//
// 多批次（1024个点 / 64PE = 16批次）：
//   每批次单独触发一次 start，CMP单元在 batch_done 时 clear 并重新锁存
//   控制器在每批次出结果后与前批次结果合并（此版本：每批次独立输出，
//   批间Top-K合并由外部控制器完成）
//
// 模式选择：
//   k_sel = 1  → Top-1（最近邻）
//   k_sel = K  → Top-K（K近邻，1≤K≤64）
// =============================================================================

module dist_topk_engine #(
    parameter PE_NUM     = 64,
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 48,
    parameter IDX_WIDTH  = 10,   // 支持最多1024个点索引
    parameter K_MAX      = 64
)(
    input  wire                              clk,
    input  wire                              rst_n,

    // 控制
    input  wire                              start,       // 开始新一批次
    input  wire [$clog2(K_MAX):0]            k_sel,       // Top-K的K值

    // 查询点
    input  wire [DATA_WIDTH-1:0]             q_x, q_y, q_z,

    // 数据库点（64个，当前批次）
    input  wire [PE_NUM*DATA_WIDTH-1:0]      db_x_packed,
    input  wire [PE_NUM*DATA_WIDTH-1:0]      db_y_packed,
    input  wire [PE_NUM*DATA_WIDTH-1:0]      db_z_packed,

    // 当前批次基地址（用于生成全局点索引）
    input  wire [IDX_WIDTH-1:0]              batch_base_idx,

    // 输出
    output wire [K_MAX*ACC_WIDTH-1:0]        topk_dist,
    output wire [K_MAX*IDX_WIDTH-1:0]        topk_idx,
    output wire                              topk_valid,

    // 调试/监控
    output wire                              pe_result_valid_dbg,
    output wire [$clog2(PE_NUM)-1:0]         pe_result_idx_dbg
);

    // =========================================================================
    // 脉动距离阵列（复用已有设计）
    // =========================================================================
    wire [ACC_WIDTH-1:0]          pe_dist_sq;
    wire                          pe_result_valid;
    wire [$clog2(PE_NUM)-1:0]     pe_result_idx;
    wire                          pe_all_done;

    assign pe_result_valid_dbg = pe_result_valid;
    assign pe_result_idx_dbg   = pe_result_idx;

    distance_engine #(
        .PE_NUM    (PE_NUM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_dist_array (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .q_x         (q_x),
        .q_y         (q_y),
        .q_z         (q_z),
        .db_x_packed (db_x_packed),
        .db_y_packed (db_y_packed),
        .db_z_packed (db_z_packed),
        .dist_sq_out (pe_dist_sq),
        .result_valid(pe_result_valid),
        .result_idx  (pe_result_idx),
        .all_done    (pe_all_done)
    );

    // =========================================================================
    // 64个CMP单元：逐个锁存PE结果
    // =========================================================================

    // 当前PE结果对应的全局点索引
    wire [IDX_WIDTH-1:0] current_global_idx;
    assign current_global_idx = batch_base_idx + {{(IDX_WIDTH-$clog2(PE_NUM)){1'b0}}, pe_result_idx};

    // CMP单元的输出（作为归约树叶节点输入）
    wire [PE_NUM*ACC_WIDTH-1:0] leaf_dist;
    wire [PE_NUM*IDX_WIDTH-1:0] leaf_idx;
    wire [PE_NUM-1:0]            leaf_valid;

    genvar gi;
    generate
        for (gi = 0; gi < PE_NUM; gi = gi + 1) begin : gen_cmp

            // 每个CMP单元只在 pe_result_idx==gi 且 pe_result_valid 时锁存
            wire cmp_pe_valid;
            assign cmp_pe_valid = pe_result_valid && (pe_result_idx == gi[$clog2(PE_NUM)-1:0]);

            cmp_unit #(
                .ACC_WIDTH(ACC_WIDTH),
                .IDX_WIDTH(IDX_WIDTH)
            ) u_cmp (
                .clk             (clk),
                .rst_n           (rst_n),
                .clear           (start),            // 新批次清零
                .pe_dist_sq      (pe_dist_sq),
                .pe_result_valid (cmp_pe_valid),
                .pe_idx          (current_global_idx),
                .cand_dist       (leaf_dist [gi*ACC_WIDTH +: ACC_WIDTH]),
                .cand_idx        (leaf_idx  [gi*IDX_WIDTH +: IDX_WIDTH]),
                .cand_valid      (leaf_valid[gi])
            );
        end
    endgenerate

    // =========================================================================
    // all_locked：所有CMP单元都已锁存有效结果
    // =========================================================================
    wire all_locked;
    assign all_locked = &leaf_valid;   // 64个valid全为1

    // start_reduce：all_locked 上升沿触发一次
    reg all_locked_r;
    wire start_reduce;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) all_locked_r <= 1'b0;
        else        all_locked_r <= all_locked;
    end
    assign start_reduce = all_locked & ~all_locked_r;

    // =========================================================================
    // 6层归约树
    // =========================================================================
    topk_reduce_tree #(
        .PE_NUM    (PE_NUM),
        .ACC_WIDTH (ACC_WIDTH),
        .IDX_WIDTH (IDX_WIDTH),
        .K_MAX     (K_MAX),
        .LAYERS    (6)
    ) u_reduce_tree (
        .clk          (clk),
        .rst_n        (rst_n),
        .leaf_dist    (leaf_dist),
        .leaf_idx     (leaf_idx),
        .leaf_valid   (leaf_valid),
        .k_sel        (k_sel),
        .start_reduce (start_reduce),
        .topk_dist    (topk_dist),
        .topk_idx     (topk_idx),
        .topk_valid   (topk_valid)
    );

endmodule