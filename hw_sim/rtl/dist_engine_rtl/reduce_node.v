// =============================================================================
// File    : reduce_node.v
// Module  : reduce_node
// Desc    : 归约树内部节点，两路有序K候选归并，输出前K个最小候选
// ─────────────────────────────────────────────────────────────────────────────
// 输入：L[0..K-1] 和 R[0..K-1] 均已按 dist 升序排列
// 输出：out[oi] = 合并后第 oi+1 小的元素，oi = 0..K-1
//
// 对每个输出槽 oi，枚举分割点 p（p个来自L，q=oi-p个来自R）：
//   有效分割条件：L[p-1] <= R[q]  AND  R[q-1] <= L[p]
//   （哨兵：L[-1]=R[-1]=0， L[K]=R[K]=INF）
//   有效时唯一确定：out[oi] = min(L[p], R[q])
//
// 全部组合逻辑，外部寄存打拍。
// =============================================================================

module reduce_node #(
    parameter ACC_WIDTH = 48,
    parameter IDX_WIDTH = 10,
    parameter K_MAX     = 64
)(
    input  wire [K_MAX*ACC_WIDTH-1:0]  left_dist,
    input  wire [K_MAX*IDX_WIDTH-1:0]  left_idx,
    input  wire                         left_valid,

    input  wire [K_MAX*ACC_WIDTH-1:0]  right_dist,
    input  wire [K_MAX*IDX_WIDTH-1:0]  right_idx,
    input  wire                         right_valid,

    input  wire [$clog2(K_MAX):0]       k_sel,

    output reg  [K_MAX*ACC_WIDTH-1:0]  out_dist,
    output reg  [K_MAX*IDX_WIDTH-1:0]  out_idx,
    output reg                          out_valid
);

    localparam INF_D = {ACC_WIDTH{1'b1}};
    localparam INF_I = {IDX_WIDTH{1'b0}};
    localparam ZERO  = {ACC_WIDTH{1'b0}};

    // 展开输入数组
    wire [ACC_WIDTH-1:0] L_d [0:K_MAX-1];
    wire [IDX_WIDTH-1:0] L_i [0:K_MAX-1];
    wire [ACC_WIDTH-1:0] R_d [0:K_MAX-1];
    wire [IDX_WIDTH-1:0] R_i [0:K_MAX-1];

    genvar gi;
    generate
        for (gi = 0; gi < K_MAX; gi = gi + 1) begin : unpack
            assign L_d[gi] = left_dist [gi*ACC_WIDTH +: ACC_WIDTH];
            assign L_i[gi] = left_idx  [gi*IDX_WIDTH +: IDX_WIDTH];
            assign R_d[gi] = right_dist[gi*ACC_WIDTH +: ACC_WIDTH];
            assign R_i[gi] = right_idx [gi*IDX_WIDTH +: IDX_WIDTH];
        end
    endgenerate

    // =========================================================================
    // 归并组合逻辑
    // =========================================================================
    integer oi, p;
    integer q;
    reg [ACC_WIDTH-1:0] lp_prev, lp_cur, rq_prev, rq_cur;
    reg                  split_ok;

    always @(*) begin
        out_valid = left_valid & right_valid;

        for (oi = 0; oi < K_MAX; oi = oi + 1) begin
            // 默认填充INF
            out_dist[oi*ACC_WIDTH +: ACC_WIDTH] = INF_D;
            out_idx [oi*IDX_WIDTH +: IDX_WIDTH] = INF_I;

            for (p = 0; p <= K_MAX; p = p + 1) begin
                q = oi - p;
                // 有效范围：p in [0,K_MAX], q in [0,K_MAX], p+q==oi
                if (q >= 0 && q <= K_MAX) begin

                    // 取 L[p-1]，越界用哨兵0
                    lp_prev = (p == 0)     ? ZERO  : L_d[p-1];
                    // 取 L[p]，越界用INF
                    lp_cur  = (p == K_MAX) ? INF_D : L_d[p];
                    // 取 R[q-1]，越界用哨兵0
                    rq_prev = (q == 0)     ? ZERO  : R_d[q-1];
                    // 取 R[q]，越界用INF
                    rq_cur  = (q == K_MAX) ? INF_D : R_d[q];

                    // 有效分割：L[p-1] <= R[q] 且 R[q-1] <= L[p]
                    split_ok = (lp_prev <= rq_cur) && (rq_prev <= lp_cur);

                    if (split_ok) begin
                        // out[oi] = min(L[p], R[q])
                        if (lp_cur <= rq_cur) begin
                            out_dist[oi*ACC_WIDTH +: ACC_WIDTH] = lp_cur;
                            out_idx [oi*IDX_WIDTH +: IDX_WIDTH] =
                                (p < K_MAX) ? L_i[p] : INF_I;
                        end else begin
                            out_dist[oi*ACC_WIDTH +: ACC_WIDTH] = rq_cur;
                            out_idx [oi*IDX_WIDTH +: IDX_WIDTH] =
                                (q < K_MAX) ? R_i[q] : INF_I;
                        end
                    end
                end
            end
        end
    end

endmodule