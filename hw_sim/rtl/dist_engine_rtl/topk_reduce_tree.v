// =============================================================================
// File    : topk_reduce_tree.v
// Module  : topk_reduce_tree
// Desc    : 6层流水线树形归约，64个叶节点 → Top-K结果
//
// 架构：
//   Layer 0 (叶): 64个 CMP单元，各持 1 个候选
//   Layer 1:      32个归并节点，每个合并2个候选 → 输出前 min(K,2) 个
//   Layer 2:      16个归并节点，合并 min(K,2)×2 → 前 min(K,4) 个
//   Layer 3:       8个归并节点 → 前 min(K,8) 个
//   Layer 4:       4个归并节点 → 前 min(K,16) 个
//   Layer 5:       2个归并节点 → 前 min(K,32) 个
//   Layer 6:       1个归并节点 → 最终Top-K（最多64个）
//
// 每层寄存打拍 → 6级流水线，启动延迟6拍
//
// Top-1复用：k_sel=1时，每层只传递最小的1个候选，退化为纯MIN树
//
// 关键参数：
//   PE_NUM   = 64（叶节点数，必须是2的幂）
//   K_MAX    = 64（最大Top-K，等于PE_NUM）
//   LAYERS   = log2(PE_NUM) = 6
// =============================================================================

module topk_reduce_tree #(
    parameter PE_NUM    = 64,
    parameter ACC_WIDTH = 48,
    parameter IDX_WIDTH = 10,
    parameter K_MAX     = 64,
    parameter LAYERS    = 6     // log2(64)
)(
    input  wire                           clk,
    input  wire                           rst_n,

    // 来自64个CMP单元的叶节点输入
    input  wire [PE_NUM*ACC_WIDTH-1:0]    leaf_dist,   // 每个叶1个候选
    input  wire [PE_NUM*IDX_WIDTH-1:0]    leaf_idx,
    input  wire [PE_NUM-1:0]              leaf_valid,  // 每个叶的有效标志

    // 运行时K配置
    input  wire [$clog2(K_MAX):0]         k_sel,       // 1~K_MAX

    // 归约触发：所有叶都有效后，外部拉高 start_reduce 启动流水线
    input  wire                           start_reduce,

    // 输出：Top-K结果（升序，dist最小的K个）
    output wire [K_MAX*ACC_WIDTH-1:0]     topk_dist,
    output wire [K_MAX*IDX_WIDTH-1:0]     topk_idx,
    output wire                           topk_valid   // 结果有效（start_reduce后6拍）
);

    // =========================================================================
    // 层间信号定义
    // 每层节点数: L0=64, L1=32, L2=16, L3=8, L4=4, L5=2, L6=1
    // 每层每节点候选数（上界）: L0=1, L1=2, L2=4, L3=8, L4=16, L5=32, L6=64
    // 但候选数不超过K_MAX
    //
    // 为简化实现，每层统一传 K_MAX 宽度，不足部分填最大值
    // =========================================================================

    // Layer 0: 叶节点（直接来自CMP单元，每节点1个候选，扩展为K_MAX宽）
    // Layer 1~6: 寄存后的层间数据

    // 每个节点的数据: K_MAX个候选 × (ACC_WIDTH + IDX_WIDTH + 1valid)
    // 节点数组: layer_dist[layer][node][k]
    // 用packed形式: layer_dist_l1[31:0] 每个32 × K_MAX × ACC_WIDTH

    // ---- Layer 0: 扩展叶节点到K_MAX候选宽度 ----
    wire [K_MAX*ACC_WIDTH-1:0] l0_dist [0:PE_NUM-1];
    wire [K_MAX*IDX_WIDTH-1:0] l0_idx  [0:PE_NUM-1];
    wire                        l0_valid[0:PE_NUM-1];

    genvar gi;
    generate
        for (gi = 0; gi < PE_NUM; gi = gi + 1) begin : leaf_expand
            // 叶节点只有1个候选，其余K_MAX-1个填最大值
            assign l0_dist[gi][0*ACC_WIDTH +: ACC_WIDTH] =
                       leaf_dist[gi*ACC_WIDTH +: ACC_WIDTH];
            assign l0_idx [gi][0*IDX_WIDTH +: IDX_WIDTH] =
                       leaf_idx [gi*IDX_WIDTH +: IDX_WIDTH];
            assign l0_valid[gi] = leaf_valid[gi];

            if (K_MAX > 1) begin : pad_slots
                genvar gk;
                for (gk = 1; gk < K_MAX; gk = gk + 1) begin : pad_k
                    assign l0_dist[gi][gk*ACC_WIDTH +: ACC_WIDTH] = {ACC_WIDTH{1'b1}};
                    assign l0_idx [gi][gk*IDX_WIDTH +: IDX_WIDTH] = {IDX_WIDTH{1'b0}};
                end
            end
        end
    endgenerate

    // =========================================================================
    // Layer 1: 32节点，每节点合并2个叶候选
    // =========================================================================
    localparam N1 = PE_NUM/2;   // 32
    wire [K_MAX*ACC_WIDTH-1:0] l1_dist_comb [0:N1-1];
    wire [K_MAX*IDX_WIDTH-1:0] l1_idx_comb  [0:N1-1];
    wire                        l1_valid_comb[0:N1-1];

    reg  [K_MAX*ACC_WIDTH-1:0] l1_dist_reg  [0:N1-1];
    reg  [K_MAX*IDX_WIDTH-1:0] l1_idx_reg   [0:N1-1];
    reg                         l1_valid_reg [0:N1-1];

    generate
        for (gi = 0; gi < N1; gi = gi + 1) begin : layer1
            reduce_node #(
                .ACC_WIDTH(ACC_WIDTH),
                .IDX_WIDTH(IDX_WIDTH),
                .K_MAX    (K_MAX)
            ) u_rn (
                .left_dist (l0_dist [gi*2]),
                .left_idx  (l0_idx  [gi*2]),
                .left_valid(l0_valid[gi*2]),
                .right_dist(l0_dist [gi*2+1]),
                .right_idx (l0_idx  [gi*2+1]),
                .right_valid(l0_valid[gi*2+1]),
                .k_sel     (k_sel),
                .out_dist  (l1_dist_comb[gi]),
                .out_idx   (l1_idx_comb [gi]),
                .out_valid (l1_valid_comb[gi])
            );
            // 寄存打拍
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    l1_dist_reg [gi] <= {(K_MAX*ACC_WIDTH){1'b1}};
                    l1_idx_reg  [gi] <= {(K_MAX*IDX_WIDTH){1'b0}};
                    l1_valid_reg[gi] <= 1'b0;
                end else begin
                    l1_dist_reg [gi] <= l1_dist_comb[gi];
                    l1_idx_reg  [gi] <= l1_idx_comb [gi];
                    l1_valid_reg[gi] <= l1_valid_comb[gi];
                end
            end
        end
    endgenerate

    // =========================================================================
    // Layer 2: 16节点
    // =========================================================================
    localparam N2 = N1/2;  // 16
    wire [K_MAX*ACC_WIDTH-1:0] l2_dist_comb [0:N2-1];
    wire [K_MAX*IDX_WIDTH-1:0] l2_idx_comb  [0:N2-1];
    wire                        l2_valid_comb[0:N2-1];

    reg  [K_MAX*ACC_WIDTH-1:0] l2_dist_reg  [0:N2-1];
    reg  [K_MAX*IDX_WIDTH-1:0] l2_idx_reg   [0:N2-1];
    reg                         l2_valid_reg [0:N2-1];

    generate
        for (gi = 0; gi < N2; gi = gi + 1) begin : layer2
            reduce_node #(.ACC_WIDTH(ACC_WIDTH),.IDX_WIDTH(IDX_WIDTH),.K_MAX(K_MAX))
            u_rn (
                .left_dist (l1_dist_reg[gi*2]),   .left_idx (l1_idx_reg[gi*2]),
                .left_valid(l1_valid_reg[gi*2]),
                .right_dist(l1_dist_reg[gi*2+1]), .right_idx(l1_idx_reg[gi*2+1]),
                .right_valid(l1_valid_reg[gi*2+1]),
                .k_sel(k_sel),
                .out_dist(l2_dist_comb[gi]), .out_idx(l2_idx_comb[gi]),
                .out_valid(l2_valid_comb[gi])
            );
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    l2_dist_reg[gi] <= {(K_MAX*ACC_WIDTH){1'b1}};
                    l2_idx_reg [gi] <= {(K_MAX*IDX_WIDTH){1'b0}};
                    l2_valid_reg[gi]<= 1'b0;
                end else begin
                    l2_dist_reg[gi] <= l2_dist_comb[gi];
                    l2_idx_reg [gi] <= l2_idx_comb [gi];
                    l2_valid_reg[gi]<= l2_valid_comb[gi];
                end
            end
        end
    endgenerate

    // =========================================================================
    // Layer 3: 8节点
    // =========================================================================
    localparam N3 = N2/2;
    wire [K_MAX*ACC_WIDTH-1:0] l3_dist_comb [0:N3-1];
    wire [K_MAX*IDX_WIDTH-1:0] l3_idx_comb  [0:N3-1];
    wire                        l3_valid_comb[0:N3-1];
    reg  [K_MAX*ACC_WIDTH-1:0] l3_dist_reg  [0:N3-1];
    reg  [K_MAX*IDX_WIDTH-1:0] l3_idx_reg   [0:N3-1];
    reg                         l3_valid_reg [0:N3-1];

    generate
        for (gi = 0; gi < N3; gi = gi + 1) begin : layer3
            reduce_node #(.ACC_WIDTH(ACC_WIDTH),.IDX_WIDTH(IDX_WIDTH),.K_MAX(K_MAX))
            u_rn (
                .left_dist (l2_dist_reg[gi*2]),   .left_idx (l2_idx_reg[gi*2]),
                .left_valid(l2_valid_reg[gi*2]),
                .right_dist(l2_dist_reg[gi*2+1]), .right_idx(l2_idx_reg[gi*2+1]),
                .right_valid(l2_valid_reg[gi*2+1]),
                .k_sel(k_sel),
                .out_dist(l3_dist_comb[gi]), .out_idx(l3_idx_comb[gi]),
                .out_valid(l3_valid_comb[gi])
            );
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    l3_dist_reg[gi] <= {(K_MAX*ACC_WIDTH){1'b1}};
                    l3_idx_reg [gi] <= {(K_MAX*IDX_WIDTH){1'b0}};
                    l3_valid_reg[gi]<= 1'b0;
                end else begin
                    l3_dist_reg[gi] <= l3_dist_comb[gi];
                    l3_idx_reg [gi] <= l3_idx_comb [gi];
                    l3_valid_reg[gi]<= l3_valid_comb[gi];
                end
            end
        end
    endgenerate

    // =========================================================================
    // Layer 4: 4节点
    // =========================================================================
    localparam N4 = N3/2;
    wire [K_MAX*ACC_WIDTH-1:0] l4_dist_comb [0:N4-1];
    wire [K_MAX*IDX_WIDTH-1:0] l4_idx_comb  [0:N4-1];
    wire                        l4_valid_comb[0:N4-1];
    reg  [K_MAX*ACC_WIDTH-1:0] l4_dist_reg  [0:N4-1];
    reg  [K_MAX*IDX_WIDTH-1:0] l4_idx_reg   [0:N4-1];
    reg                         l4_valid_reg [0:N4-1];

    generate
        for (gi = 0; gi < N4; gi = gi + 1) begin : layer4
            reduce_node #(.ACC_WIDTH(ACC_WIDTH),.IDX_WIDTH(IDX_WIDTH),.K_MAX(K_MAX))
            u_rn (
                .left_dist (l3_dist_reg[gi*2]),   .left_idx (l3_idx_reg[gi*2]),
                .left_valid(l3_valid_reg[gi*2]),
                .right_dist(l3_dist_reg[gi*2+1]), .right_idx(l3_idx_reg[gi*2+1]),
                .right_valid(l3_valid_reg[gi*2+1]),
                .k_sel(k_sel),
                .out_dist(l4_dist_comb[gi]), .out_idx(l4_idx_comb[gi]),
                .out_valid(l4_valid_comb[gi])
            );
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    l4_dist_reg[gi] <= {(K_MAX*ACC_WIDTH){1'b1}};
                    l4_idx_reg [gi] <= {(K_MAX*IDX_WIDTH){1'b0}};
                    l4_valid_reg[gi]<= 1'b0;
                end else begin
                    l4_dist_reg[gi] <= l4_dist_comb[gi];
                    l4_idx_reg [gi] <= l4_idx_comb [gi];
                    l4_valid_reg[gi]<= l4_valid_comb[gi];
                end
            end
        end
    endgenerate

    // =========================================================================
    // Layer 5: 2节点
    // =========================================================================
    localparam N5 = N4/2;
    wire [K_MAX*ACC_WIDTH-1:0] l5_dist_comb [0:N5-1];
    wire [K_MAX*IDX_WIDTH-1:0] l5_idx_comb  [0:N5-1];
    wire                        l5_valid_comb[0:N5-1];
    reg  [K_MAX*ACC_WIDTH-1:0] l5_dist_reg  [0:N5-1];
    reg  [K_MAX*IDX_WIDTH-1:0] l5_idx_reg   [0:N5-1];
    reg                         l5_valid_reg [0:N5-1];

    generate
        for (gi = 0; gi < N5; gi = gi + 1) begin : layer5
            reduce_node #(.ACC_WIDTH(ACC_WIDTH),.IDX_WIDTH(IDX_WIDTH),.K_MAX(K_MAX))
            u_rn (
                .left_dist (l4_dist_reg[gi*2]),   .left_idx (l4_idx_reg[gi*2]),
                .left_valid(l4_valid_reg[gi*2]),
                .right_dist(l4_dist_reg[gi*2+1]), .right_idx(l4_idx_reg[gi*2+1]),
                .right_valid(l4_valid_reg[gi*2+1]),
                .k_sel(k_sel),
                .out_dist(l5_dist_comb[gi]), .out_idx(l5_idx_comb[gi]),
                .out_valid(l5_valid_comb[gi])
            );
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    l5_dist_reg[gi] <= {(K_MAX*ACC_WIDTH){1'b1}};
                    l5_idx_reg [gi] <= {(K_MAX*IDX_WIDTH){1'b0}};
                    l5_valid_reg[gi]<= 1'b0;
                end else begin
                    l5_dist_reg[gi] <= l5_dist_comb[gi];
                    l5_idx_reg [gi] <= l5_idx_comb [gi];
                    l5_valid_reg[gi]<= l5_valid_comb[gi];
                end
            end
        end
    endgenerate

    // =========================================================================
    // Layer 6: 1个根节点，输出最终Top-K
    // =========================================================================
    wire [K_MAX*ACC_WIDTH-1:0] l6_dist_comb;
    wire [K_MAX*IDX_WIDTH-1:0] l6_idx_comb;
    wire                        l6_valid_comb;
    reg  [K_MAX*ACC_WIDTH-1:0] l6_dist_reg;
    reg  [K_MAX*IDX_WIDTH-1:0] l6_idx_reg;
    reg                         l6_valid_reg;

    reduce_node #(.ACC_WIDTH(ACC_WIDTH),.IDX_WIDTH(IDX_WIDTH),.K_MAX(K_MAX))
    u_root (
        .left_dist (l5_dist_reg[0]), .left_idx (l5_idx_reg[0]),
        .left_valid(l5_valid_reg[0]),
        .right_dist(l5_dist_reg[1]), .right_idx(l5_idx_reg[1]),
        .right_valid(l5_valid_reg[1]),
        .k_sel(k_sel),
        .out_dist(l6_dist_comb), .out_idx(l6_idx_comb),
        .out_valid(l6_valid_comb)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l6_dist_reg  <= {(K_MAX*ACC_WIDTH){1'b1}};
            l6_idx_reg   <= {(K_MAX*IDX_WIDTH){1'b0}};
            l6_valid_reg <= 1'b0;
        end else begin
            l6_dist_reg  <= l6_dist_comb;
            l6_idx_reg   <= l6_idx_comb;
            l6_valid_reg <= l6_valid_comb;
        end
    end

    // =========================================================================
    // start_reduce → topk_valid 延迟6拍
    // =========================================================================
    reg [5:0] valid_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipe <= 6'b0;
        else
            valid_pipe <= {valid_pipe[4:0], start_reduce};
    end

    assign topk_valid = valid_pipe[5] & l6_valid_reg;
    assign topk_dist  = l6_dist_reg;
    assign topk_idx   = l6_idx_reg;

endmodule