// =============================================================================
// File    : cmp_unit.v
// Module  : cmp_unit
// Desc    : CMP单元，挂在每个PE下方
//           功能：
//             1. 锁存对应PE的dist_sq结果和点索引
//             2. 作为归约树的叶节点，向上提供候选(dist_sq, idx)对
//
// 每个CMP单元固定持有1个候选（叶节点），
// 树形归约由 reduce_node.v 完成。
// =============================================================================

module cmp_unit #(
    parameter ACC_WIDTH  = 48,
    parameter IDX_WIDTH  = 10    // 支持最多1024个点的索引
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,          // 开始新一轮计算时清零

    // 来自PE的结果
    input  wire [ACC_WIDTH-1:0]  pe_dist_sq,
    input  wire                  pe_result_valid, // PE结果有效脉冲
    input  wire [IDX_WIDTH-1:0]  pe_idx,          // 该PE对应的点索引（由控制器提供）

    // 向归约树输出的候选
    output reg  [ACC_WIDTH-1:0]  cand_dist,
    output reg  [IDX_WIDTH-1:0]  cand_idx,
    output reg                   cand_valid       // 本单元已锁存有效结果
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cand_dist  <= {ACC_WIDTH{1'b1}};  // 初始化为最大值（无效候选）
            cand_idx   <= {IDX_WIDTH{1'b0}};
            cand_valid <= 1'b0;
        end else if (clear) begin
            cand_dist  <= {ACC_WIDTH{1'b1}};
            cand_idx   <= {IDX_WIDTH{1'b0}};
            cand_valid <= 1'b0;
        end else if (pe_result_valid) begin
            cand_dist  <= pe_dist_sq;
            cand_idx   <= pe_idx;
            cand_valid <= 1'b1;
        end
    end

endmodule