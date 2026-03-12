// =============================================================================
// File    : pe.v
// Module  : PE (Processing Element) — Systolic Version
//
// 脉动逻辑：
//   每个PE从左侧接收 q_coord_in（查询坐标），向右侧传递 q_coord_out（脉动）
//   同时接收本PE对应的数据库坐标 d_coord（由控制器按PE编号分时喂入）
//   内部维度计数器 dim_cnt 记录当前处于 X/Y/Z 哪个维度
//
//   dim_cnt == 0 (X): acc = (q-d)²           （清零并写入）
//   dim_cnt == 1 (Y): acc += (q-d)²
//   dim_cnt == 2 (Z): acc += (q-d)²，同时拉高 result_valid 输出结果
//
// 脉动传递：
//   q_coord_in 在 posedge 寄存后变为 q_coord_out，实现一拍延迟脉动
// =============================================================================

module pe #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 48
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // 脉动输入：来自左侧PE（或控制器）的查询坐标
    input  wire                   q_valid_in,              // 本cycle查询坐标有效
    input  wire [DATA_WIDTH-1:0]  q_coord_in,              // 查询坐标

    // 脉动输出：传递给右侧PE（一拍延迟）
    output reg                    q_valid_out,
    output reg  [DATA_WIDTH-1:0]  q_coord_out,

    // 数据库坐标输入（由控制器在正确周期喂入）
    input  wire [DATA_WIDTH-1:0]  d_coord,

    // 结果输出
    output reg  [ACC_WIDTH-1:0]   dist_sq_out,
    output reg                    result_valid
);

    // -------------------------------------------------------------------------
    // 维度计数器：有效输入时 0→1→2→0 循环
    // -------------------------------------------------------------------------
    reg [1:0] dim_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dim_cnt <= 2'd0;
        else if (q_valid_in) begin
            if (dim_cnt == 2'd2)
                dim_cnt <= 2'd0;
            else
                dim_cnt <= dim_cnt + 2'd1;
        end
    end

    // -------------------------------------------------------------------------
    // 脉动传递：q_coord_in 寄存一拍后送右侧PE
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_valid_out <= 1'b0;
            q_coord_out <= {DATA_WIDTH{1'b0}};
        end else begin
            q_valid_out <= q_valid_in;
            q_coord_out <= q_coord_in;
        end
    end

    // -------------------------------------------------------------------------
    // 差值计算与累加
    // -------------------------------------------------------------------------
    wire signed [DATA_WIDTH-1:0]   diff;
    wire signed [2*DATA_WIDTH-1:0] diff_sq;

    assign diff    = $signed(q_coord_in) - $signed(d_coord);
    assign diff_sq = $signed(diff) * $signed(diff);

    reg [ACC_WIDTH-1:0] acc_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg      <= {ACC_WIDTH{1'b0}};
            dist_sq_out  <= {ACC_WIDTH{1'b0}};
            result_valid <= 1'b0;
        end else begin
            result_valid <= 1'b0;   // 默认低
            if (q_valid_in) begin
                case (dim_cnt)
                    2'd0: begin  // X维度：清零并写入
                        acc_reg <= {{(ACC_WIDTH-2*DATA_WIDTH){1'b0}}, diff_sq};
                    end
                    2'd1: begin  // Y维度：累加
                        acc_reg <= acc_reg + {{(ACC_WIDTH-2*DATA_WIDTH){1'b0}}, diff_sq};
                    end
                    2'd2: begin  // Z维度：累加并输出结果
                        dist_sq_out  <= acc_reg + {{(ACC_WIDTH-2*DATA_WIDTH){1'b0}}, diff_sq};
                        result_valid <= 1'b1;
                    end
                    default: acc_reg <= {ACC_WIDTH{1'b0}};
                endcase
            end
        end
    end

endmodule