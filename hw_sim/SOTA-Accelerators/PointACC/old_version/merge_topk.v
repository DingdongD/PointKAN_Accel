// // merge_topk_syn.v
// `timescale 1ns/1ps

// // ============================================
// // Module 1: block_sorter
// //  - 完整多周期冒泡排序（升序），最后输出前 K 个最小元素
// // ============================================
// module block_sorter #(
//     parameter DATA_WIDTH = 16,
//     parameter BLOCK_SIZE = 8,
//     parameter K = 4
// )(
//     input  wire                       clk,
//     input  wire                       rst_n,
//     input  wire                       start,
//     input  wire [DATA_WIDTH*BLOCK_SIZE-1:0] data_in,  // packed input
//     output reg  [DATA_WIDTH*K-1:0]    data_out,      // packed Top-K (smallest K)
//     output reg                        valid
// );
//     // internal buffer
//     reg [DATA_WIDTH-1:0] buffer [0:BLOCK_SIZE-1];
//     integer i;
//     // sorter state
//     reg [3:0] state;
//     localparam S_IDLE = 0, S_LOAD = 1, S_SORT = 2, S_OUT = 3;

//     // bubble counters
//     integer outer_idx;
//     integer inner_idx;
//     reg [DATA_WIDTH-1:0] tmp;

//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             state <= S_IDLE;
//             valid <= 0;
//             outer_idx <= 0;
//             inner_idx <= 0;
//         end else begin
//             case (state)
//                 S_IDLE: begin
//                     valid <= 0;
//                     if (start) begin
//                         state <= S_LOAD;
//                     end
//                 end

//                 S_LOAD: begin
//                     // unpack inputs into buffer (BLOCK_SIZE is parameter constant)
//                     for (i = 0; i < BLOCK_SIZE; i = i + 1) begin
//                         buffer[i] <= data_in[i*DATA_WIDTH +: DATA_WIDTH];
//                     end
//                     outer_idx <= 0;
//                     inner_idx <= 0;
//                     state <= S_SORT;
//                 end

//                 S_SORT: begin
//                     // do one compare-swap per clock (inner loop single step)
//                     if (outer_idx < BLOCK_SIZE-1) begin
//                         if (inner_idx < BLOCK_SIZE-1-outer_idx) begin
//                             if (buffer[inner_idx] > buffer[inner_idx+1]) begin
//                                 tmp <= buffer[inner_idx];
//                                 buffer[inner_idx] <= buffer[inner_idx+1];
//                                 buffer[inner_idx+1] <= tmp;
//                             end
//                             inner_idx <= inner_idx + 1;
//                         end else begin
//                             inner_idx <= 0;
//                             outer_idx <= outer_idx + 1;
//                         end
//                     end else begin
//                         state <= S_OUT;
//                     end
//                 end

//                 S_OUT: begin
//                     // pack smallest K elements (buffer[0..K-1])
//                     for (i = 0; i < K; i = i + 1) begin
//                         data_out[i*DATA_WIDTH +: DATA_WIDTH] <= buffer[i];
//                     end
//                     valid <= 1;
//                     state <= S_IDLE;
//                 end

//                 default: state <= S_IDLE;
//             endcase
//         end
//     end
// endmodule





// // ============================================
// // Module 2: merge_unit
// //  - 每时钟输出一个合并后的元素（最多 K），直到合并结束
// // ============================================
// module merge_unit #(
//     parameter DATA_WIDTH = 16,
//     parameter K = 4
// )(
//     input  wire                     clk,
//     input  wire                     rst_n,
//     input  wire                     start,
//     input  wire [DATA_WIDTH*K-1:0]  list_a,
//     input  wire [DATA_WIDTH*K-1:0]  list_b,
//     input  wire [3:0]               len_a,  // valid lengths
//     input  wire [3:0]               len_b,
//     output reg  [DATA_WIDTH*K-1:0]  merged,
//     output reg  [3:0]               merged_len,
//     output reg                      valid
// );
//     reg [DATA_WIDTH-1:0] a_buf [0:K-1];
//     reg [DATA_WIDTH-1:0] b_buf [0:K-1];
//     reg [DATA_WIDTH-1:0] out_buf [0:K-1];

//     integer m;
//     reg [3:0] i, j, out_idx;
//     reg [1:0] state;
//     localparam M_IDLE = 0, M_LOAD = 1, M_MERGE = 2, M_OUT = 3;

//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             state <= M_IDLE;
//             valid <= 0;
//             i <= 0; j <= 0; out_idx <= 0;
//             merged_len <= 0;
//         end else begin
//             case (state)
//                 M_IDLE: begin
//                     valid <= 0;
//                     merged_len <= 0;
//                     if (start) begin
//                         state <= M_LOAD;
//                     end
//                 end

//                 M_LOAD: begin
//                     // unpack inputs (K is compile-time constant)
//                     for (m = 0; m < K; m = m + 1) begin
//                         a_buf[m] <= list_a[m*DATA_WIDTH +: DATA_WIDTH];
//                         b_buf[m] <= list_b[m*DATA_WIDTH +: DATA_WIDTH];
//                     end
//                     i <= 0; j <= 0; out_idx <= 0;
//                     state <= M_MERGE;
//                 end

//                 M_MERGE: begin
//                     if (out_idx >= K) begin
//                         state <= M_OUT;
//                     end else if (i >= len_a && j >= len_b) begin
//                         state <= M_OUT;
//                     end else if (i >= len_a) begin
//                         out_buf[out_idx] <= b_buf[j];
//                         j <= j + 1;
//                         out_idx <= out_idx + 1;
//                     end else if (j >= len_b) begin
//                         out_buf[out_idx] <= a_buf[i];
//                         i <= i + 1;
//                         out_idx <= out_idx + 1;
//                     end else if (a_buf[i] <= b_buf[j]) begin
//                         out_buf[out_idx] <= a_buf[i];
//                         i <= i + 1;
//                         out_idx <= out_idx + 1;
//                     end else begin
//                         out_buf[out_idx] <= b_buf[j];
//                         j <= j + 1;
//                         out_idx <= out_idx + 1;
//                     end
//                 end

//                 M_OUT: begin
//                     for (m = 0; m < K; m = m + 1) begin
//                         if (m < out_idx) begin
//                             merged[m*DATA_WIDTH +: DATA_WIDTH] <= out_buf[m];
//                         end else begin
//                             merged[m*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
//                         end
//                     end
//                     merged_len <= out_idx;
//                     valid <= 1;
//                     state <= M_IDLE;
//                 end

//                 default: state <= M_IDLE;
//             endcase
//         end
//     end
// endmodule


// // ============================================
// // Module 3: topk_merge_sort (顶层) - 综合友好版本
// //   - 时分复用单个 block_sorter 与 单个 merge_unit
// //   - 替换所有动态上限循环为固定上限 + 条件判断（避免 ELAB-900）
// // ============================================
// module topk_merge_sort #(
//     parameter DATA_WIDTH = 16,
//     parameter BLOCK_SIZE = 8,
//     parameter NUM_BLOCKS = 4,
//     parameter K = 4
// )(
//     input  wire                             clk,
//     input  wire                             rst_n,
//     input  wire                             start,
//     input  wire [DATA_WIDTH*BLOCK_SIZE*NUM_BLOCKS-1:0] data_in,
//     output reg  [DATA_WIDTH*K-1:0]          topk_out,
//     output reg                              done
// );
//     // storage for blocks (each stores packed K elements)
//     reg [DATA_WIDTH*K-1:0] blocks [0:NUM_BLOCKS-1];
//     reg [3:0] block_lens [0:NUM_BLOCKS-1];

//     // instantiate single block_sorter (time-multiplexed)
//     reg bs_start;
//     wire bs_valid;
//     reg  [DATA_WIDTH*BLOCK_SIZE-1:0] bs_data_in;
//     wire [DATA_WIDTH*K-1:0] bs_data_out;

//     block_sorter #(
//         .DATA_WIDTH(DATA_WIDTH),
//         .BLOCK_SIZE(BLOCK_SIZE),
//         .K(K)
//     ) bs (
//         .clk(clk),
//         .rst_n(rst_n),
//         .start(bs_start),
//         .data_in(bs_data_in),
//         .data_out(bs_data_out),
//         .valid(bs_valid)
//     );

//     // instantiate single merge_unit (time-multiplexed)
//     reg mu_start;
//     wire mu_valid;
//     reg [DATA_WIDTH*K-1:0] mu_list_a;
//     reg [DATA_WIDTH*K-1:0] mu_list_b;
//     reg [3:0] mu_len_a;
//     reg [3:0] mu_len_b;
//     wire [DATA_WIDTH*K-1:0] mu_out;
//     wire [3:0] mu_out_len;

//     merge_unit #(
//         .DATA_WIDTH(DATA_WIDTH),
//         .K(K)
//     ) mu (
//         .clk(clk),
//         .rst_n(rst_n),
//         .start(mu_start),
//         .list_a(mu_list_a),
//         .list_b(mu_list_b),
//         .len_a(mu_len_a),
//         .len_b(mu_len_b),
//         .merged(mu_out),
//         .merged_len(mu_out_len),
//         .valid(mu_valid)
//     );

//     // top state machine
//     reg [3:0] state;
//     localparam T_IDLE = 0, T_INIT_BLOCKS = 1, T_WAIT_BS = 2,
//                T_MERGE_ROUND = 3, T_WAIT_MU = 4, T_DONE = 5;

//     integer idx;           // general loop index for static loops
//     integer cur_block;     // which block we're currently sorting (0..NUM_BLOCKS-1)
//     integer pair_idx;      // for merge: current pair base index (0,2,4,...)
//     integer write_idx;     // next write index in next_blocks (0..NUM_BLOCKS)
//     integer num_active;    // number of active blocks in current round

//     // temporary storage for next-level blocks during merging
//     reg [DATA_WIDTH*K-1:0] next_blocks [0:NUM_BLOCKS-1];
//     reg [3:0] next_lens [0:NUM_BLOCKS-1];

//     // initialize / reset
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             state <= T_IDLE;
//             done <= 0;
//             bs_start <= 0;
//             mu_start <= 0;
//             num_active <= NUM_BLOCKS;
//             cur_block <= 0;
//             pair_idx <= 0;
//             write_idx <= 0;
//             // clear blocks
//             for (idx = 0; idx < NUM_BLOCKS; idx = idx + 1) begin
//                 blocks[idx] <= {DATA_WIDTH*K{1'b0}};
//                 block_lens[idx] <= 0;
//             end
//         end else begin
//             case (state)
//                 T_IDLE: begin
//                     done <= 0;
//                     if (start) begin
//                         // start sorting blocks sequentially
//                         cur_block <= 0;
//                         num_active <= NUM_BLOCKS;
//                         state <= T_INIT_BLOCKS;
//                     end
//                 end

//                 // feed each block into block_sorter sequentially
//                 T_INIT_BLOCKS: begin
//                     // slice out block cur_block from data_in (block 0 is LSB chunk)
//                     bs_data_in <= data_in[cur_block*BLOCK_SIZE*DATA_WIDTH +: BLOCK_SIZE*DATA_WIDTH];
//                     bs_start <= 1;
//                     state <= T_WAIT_BS;
//                 end

//                 T_WAIT_BS: begin
//                     bs_start <= 0;
//                     if (bs_valid) begin
//                         // store sorted top-K for this block
//                         blocks[cur_block] <= bs_data_out;
//                         block_lens[cur_block] <= (K <= BLOCK_SIZE) ? K : BLOCK_SIZE;
//                         // next block or move to merging
//                         if (cur_block + 1 < NUM_BLOCKS) begin
//                             cur_block <= cur_block + 1;
//                             state <= T_INIT_BLOCKS;
//                         end else begin
//                             // all blocks processed -> start first merge round
//                             pair_idx <= 0;
//                             write_idx <= 0;
//                             state <= T_MERGE_ROUND;
//                         end
//                     end
//                 end

//                 // merging round: iterate pairs (0&1, 2&3, ...) and store merged results into next_blocks
//                 T_MERGE_ROUND: begin
//                     mu_start <= 0;
//                     if (num_active == 1) begin
//                         // done
//                         state <= T_DONE;
//                     end else begin
//                         // if pair_idx < num_active then process this pair
//                         if (pair_idx < num_active) begin
//                             if (pair_idx + 1 < num_active) begin
//                                 // normal pair: start merge of blocks[pair_idx] and blocks[pair_idx+1]
//                                 mu_list_a <= blocks[pair_idx];
//                                 mu_list_b <= blocks[pair_idx+1];
//                                 mu_len_a  <= block_lens[pair_idx];
//                                 mu_len_b  <= block_lens[pair_idx+1];
//                                 mu_start <= 1;
//                                 state <= T_WAIT_MU;
//                             end else begin
//                                 // odd block out: copy it to next level at write_idx
//                                 next_blocks[write_idx] <= blocks[pair_idx];
//                                 next_lens[write_idx] <= block_lens[pair_idx];
//                                 write_idx <= write_idx + 1;
//                                 pair_idx <= pair_idx + 2;
//                                 // remain in T_MERGE_ROUND to continue pairs
//                             end
//                         end else begin
//                             // finished this round: copy next_blocks back into blocks[]
//                             // IMPORTANT: use fixed-loop limit NUM_BLOCKS and conditional copy for synthesis
//                             for (idx = 0; idx < NUM_BLOCKS; idx = idx + 1) begin
//                                 if (idx < write_idx) begin
//                                     blocks[idx] <= next_blocks[idx];
//                                     block_lens[idx] <= next_lens[idx];
//                                 end else begin
//                                     blocks[idx] <= {DATA_WIDTH*K{1'b0}};
//                                     block_lens[idx] <= 0;
//                                 end
//                             end
//                             // update active count and reset indices for next round
//                             num_active <= write_idx;
//                             pair_idx <= 0;
//                             write_idx <= 0;
//                             // remain in T_MERGE_ROUND to either start another round or finish
//                         end
//                     end
//                 end

//                 T_WAIT_MU: begin
//                     mu_start <= 0;
//                     if (mu_valid) begin
//                         // capture merged result into next_blocks[write_idx]
//                         next_blocks[write_idx] <= mu_out;
//                         next_lens[write_idx] <= mu_out_len;
//                         write_idx <= write_idx + 1;
//                         pair_idx <= pair_idx + 2;
//                         state <= T_MERGE_ROUND;
//                     end
//                 end

//                 T_DONE: begin
//                     topk_out <= blocks[0];
//                     done <= 1;
//                     state <= T_IDLE;
//                 end

//                 default: state <= T_IDLE;
//             endcase
//         end
//     end
// endmodule




























// ============================================
// Module 1a: 比较交换单元（Compare-Swap）
// ============================================
// module compare_swap #(
//     parameter DATA_WIDTH = 16
// )(
//     input wire [DATA_WIDTH-1:0] in_a,
//     input wire [DATA_WIDTH-1:0] in_b,
//     input wire ascending,  // 1=升序, 0=降序
//     output wire [DATA_WIDTH-1:0] out_min,
//     output wire [DATA_WIDTH-1:0] out_max
// );
//     wire swap;
//     assign swap = ascending ? (in_a > in_b) : (in_a < in_b);
//     assign out_min = swap ? in_b : in_a;
//     assign out_max = swap ? in_a : in_b;
// endmodule

// 可配置signed和unsinged
module compare_swap #(
    parameter DATA_WIDTH = 16,
    parameter SIGNED = 1 // 0: unsigned  1: signed
)(
    input wire [DATA_WIDTH-1:0] in_a,
    input wire [DATA_WIDTH-1:0] in_b,
    input wire ascending,  // 1=升序, 0=降序
    output wire [DATA_WIDTH-1:0] out_min,
    output wire [DATA_WIDTH-1:0] out_max
);
    wire signed [DATA_WIDTH - 1 : 0] a_s = in_a;
    wire signed [DATA_WIDTH - 1 : 0] b_s = in_b;
    wire cmp = SIGNED ? (a_s > b_s) : (in_a > in_b);
    wire swap = ascending ? cmp : ~cmp;
    assign out_min = swap ? in_b : in_a;
    assign out_max = swap ? in_a : in_b;
endmodule


// ============================================
// Module 1b: Bitonic排序单元（流水线实现）
// ============================================
// module block_sorter #(
//     parameter DATA_WIDTH = 16,
//     parameter BLOCK_SIZE = 8,  // 必须是2的幂次
//     parameter K = 4
// )(
//     input wire clk,
//     input wire rst_n,
//     input wire start,
//     input wire [DATA_WIDTH*BLOCK_SIZE-1:0] data_in,
//     output reg [DATA_WIDTH*K-1:0] data_out,
//     output reg valid
// );
//     // 计算Bitonic排序的级数
//     // 对于N=8: log2(8)=3, 总级数 = 3*(3+1)/2 = 6
//     localparam LOG_N = $clog2(BLOCK_SIZE);
//     localparam NUM_STAGES = (LOG_N * (LOG_N + 1)) / 2; 
    
//     // 流水线寄存器: [stage][element]
//     reg [DATA_WIDTH-1:0] pipeline [0:NUM_STAGES][0:BLOCK_SIZE-1];  // 这里包含LOGN*(LOGN+1)/2+1级寄存器
    
//     // 中间wire信号用于连接比较器输出
//     wire [DATA_WIDTH-1:0] stage_wire [1:NUM_STAGES][0:BLOCK_SIZE-1];
    
//     // 有效信号流水线
//     reg valid_pipe [0:NUM_STAGES];
    
//     integer i;
    
//     // ========================================
//     // 第0级：输入加载
//     // ========================================
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             valid_pipe[0] <= 0;
//             for (i = 0; i < BLOCK_SIZE; i = i + 1) begin
//                 pipeline[0][i] <= 0;
//             end
//         end else begin
//             valid_pipe[0] <= start;
//             if (start) begin
//                 for (i = 0; i < BLOCK_SIZE; i = i + 1) begin
//                     pipeline[0][i] <= data_in[i*DATA_WIDTH +: DATA_WIDTH];  // 加载数据
//                 end
//             end
//         end
//     end
    
//     // ========================================
//     // 生成Bitonic排序网络
//     // ========================================
//     genvar stage;
//     generate
//         for (stage = 1; stage <= NUM_STAGES; stage = stage + 1) begin : gen_stages
            
//             // 为每个stage添加寄存器
//             always @(posedge clk or negedge rst_n) begin : stage_reg
//                 integer j;
//                 if (!rst_n) begin
//                     valid_pipe[stage] <= 0;
//                     for (j = 0; j < BLOCK_SIZE; j = j + 1) begin
//                         pipeline[stage][j] <= 0;  // 复位初始化0
//                     end
//                 end else begin
//                     valid_pipe[stage] <= valid_pipe[stage-1]; // 
//                     for (j = 0; j < BLOCK_SIZE; j = j + 1) begin
//                         pipeline[stage][j] <= stage_wire[stage][j];
//                     end
//                 end
//             end
            
//             // Stage 1: outer=0, inner=0 (box_size=2, compare_dist=1)
//             if (stage == 1) begin : stage1
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs0 (
//                     .in_a(pipeline[stage-1][0]), .in_b(pipeline[stage-1][1]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][0]), .out_max(stage_wire[stage][1])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs1 (
//                     .in_a(pipeline[stage-1][2]), .in_b(pipeline[stage-1][3]),
//                     .ascending(1'b0), .out_min(stage_wire[stage][2]), .out_max(stage_wire[stage][3])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs2 (
//                     .in_a(pipeline[stage-1][4]), .in_b(pipeline[stage-1][5]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][4]), .out_max(stage_wire[stage][5])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs3 (
//                     .in_a(pipeline[stage-1][6]), .in_b(pipeline[stage-1][7]),
//                     .ascending(1'b0), .out_min(stage_wire[stage][6]), .out_max(stage_wire[stage][7])
//                 );
//             end // 阶段1中需要对每8个元素的0-1 2-3 4-5 6-7进行两两交换 得到四个序列
            
//             // Stage 2: outer=1, inner=0 (box_size=4, compare_dist=2)
//             else if (stage == 2) begin : stage2
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs0 (
//                     .in_a(pipeline[stage-1][0]), .in_b(pipeline[stage-1][2]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][0]), .out_max(stage_wire[stage][2])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs1 (
//                     .in_a(pipeline[stage-1][1]), .in_b(pipeline[stage-1][3]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][1]), .out_max(stage_wire[stage][3])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs2 (
//                     .in_a(pipeline[stage-1][4]), .in_b(pipeline[stage-1][6]),
//                     .ascending(1'b0), .out_min(stage_wire[stage][4]), .out_max(stage_wire[stage][6])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs3 (
//                     .in_a(pipeline[stage-1][5]), .in_b(pipeline[stage-1][7]),
//                     .ascending(1'b0), .out_min(stage_wire[stage][5]), .out_max(stage_wire[stage][7])
//                 );
//             end // 第二级需要对0-2 1-3 4-6 5-7进行两两处理
            
//             // Stage 3: outer=1, inner=1 (box_size=4, compare_dist=1)
//             else if (stage == 3) begin : stage3
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs0 (
//                     .in_a(pipeline[stage-1][0]), .in_b(pipeline[stage-1][1]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][0]), .out_max(stage_wire[stage][1])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs1 (
//                     .in_a(pipeline[stage-1][2]), .in_b(pipeline[stage-1][3]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][2]), .out_max(stage_wire[stage][3])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs2 (
//                     .in_a(pipeline[stage-1][4]), .in_b(pipeline[stage-1][5]),
//                     .ascending(1'b0), .out_min(stage_wire[stage][4]), .out_max(stage_wire[stage][5])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs3 (
//                     .in_a(pipeline[stage-1][6]), .in_b(pipeline[stage-1][7]),
//                     .ascending(1'b0), .out_min(stage_wire[stage][6]), .out_max(stage_wire[stage][7])
//                 );
//             end // 第三极需要对0-1 2-3 4-5 6-7进行两两处理  前4最小 后4最大
            
//             // Stage 4: outer=2, inner=0 (box_size=8, compare_dist=4)
//             else if (stage == 4) begin : stage4
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs0 (
//                     .in_a(pipeline[stage-1][0]), .in_b(pipeline[stage-1][4]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][0]), .out_max(stage_wire[stage][4])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs1 (
//                     .in_a(pipeline[stage-1][1]), .in_b(pipeline[stage-1][5]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][1]), .out_max(stage_wire[stage][5])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs2 (
//                     .in_a(pipeline[stage-1][2]), .in_b(pipeline[stage-1][6]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][2]), .out_max(stage_wire[stage][6])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs3 (
//                     .in_a(pipeline[stage-1][3]), .in_b(pipeline[stage-1][7]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][3]), .out_max(stage_wire[stage][7])
//                 );
//             end // 这一步把“低半区升序”和“高半区降序”合并，整个序列变成一个大的 bitonic 序列（先升后降）
            
//             // Stage 5: outer=2, inner=1 (box_size=8, compare_dist=2)
//             else if (stage == 5) begin : stage5
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs0 (
//                     .in_a(pipeline[stage-1][0]), .in_b(pipeline[stage-1][2]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][0]), .out_max(stage_wire[stage][2])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs1 (
//                     .in_a(pipeline[stage-1][1]), .in_b(pipeline[stage-1][3]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][1]), .out_max(stage_wire[stage][3])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs2 (
//                     .in_a(pipeline[stage-1][4]), .in_b(pipeline[stage-1][6]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][4]), .out_max(stage_wire[stage][6])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs3 (
//                     .in_a(pipeline[stage-1][5]), .in_b(pipeline[stage-1][7]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][5]), .out_max(stage_wire[stage][7])
//                 );
//             end
            
//             // Stage 6: outer=2, inner=2 (box_size=8, compare_dist=1)
//             else if (stage == 6) begin : stage6
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs0 (
//                     .in_a(pipeline[stage-1][0]), .in_b(pipeline[stage-1][1]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][0]), .out_max(stage_wire[stage][1])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs1 (
//                     .in_a(pipeline[stage-1][2]), .in_b(pipeline[stage-1][3]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][2]), .out_max(stage_wire[stage][3])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs2 (
//                     .in_a(pipeline[stage-1][4]), .in_b(pipeline[stage-1][5]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][4]), .out_max(stage_wire[stage][5])
//                 );
//                 compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs3 (
//                     .in_a(pipeline[stage-1][6]), .in_b(pipeline[stage-1][7]),
//                     .ascending(1'b1), .out_min(stage_wire[stage][6]), .out_max(stage_wire[stage][7])
//                 );
//             end
//         end
//     endgenerate
    
//     // ========================================
//     // 输出级：提取Top-K
//     // ========================================
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             valid <= 0;
//             data_out <= 0;
//         end else begin
//             valid <= valid_pipe[NUM_STAGES];
//             if (valid_pipe[NUM_STAGES]) begin
//                 for (i = 0; i < K; i = i + 1) begin
//                     data_out[i*DATA_WIDTH +: DATA_WIDTH] <= pipeline[NUM_STAGES][i]; // 取出最后一级的排序结果
//                 end
//             end
//         end
//     end
//     // 总共7级寄存器 延迟7拍
// endmodule


// Auto-generated by gen_bitonic_vfile.py  2025-11-19T13:05:39.879774Z
// Bitonic sorter: BLOCK_SIZE=8, DATA_WIDTH=16, K=4
module block_sorter #(
    parameter DATA_WIDTH = 16,
    parameter BLOCK_SIZE = 8,
    parameter K = 4
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start,
    input  wire [DATA_WIDTH*BLOCK_SIZE-1:0] data_in,
    output reg  [DATA_WIDTH*K-1:0]    data_out,
    output reg                        valid
);

    localparam LOG_N = 3;
    localparam NUM_STAGES = 6;

    // pipeline registers: pipeline[stage][index]
    // stage 0 registers
    reg [DATA_WIDTH-1:0] pipeline_s0_0;
    reg [DATA_WIDTH-1:0] pipeline_s0_1;
    reg [DATA_WIDTH-1:0] pipeline_s0_2;
    reg [DATA_WIDTH-1:0] pipeline_s0_3;
    reg [DATA_WIDTH-1:0] pipeline_s0_4;
    reg [DATA_WIDTH-1:0] pipeline_s0_5;
    reg [DATA_WIDTH-1:0] pipeline_s0_6;
    reg [DATA_WIDTH-1:0] pipeline_s0_7;
    reg valid_s0;

    // stage 1 registers
    reg [DATA_WIDTH-1:0] pipeline_s1_0;
    reg [DATA_WIDTH-1:0] pipeline_s1_1;
    reg [DATA_WIDTH-1:0] pipeline_s1_2;
    reg [DATA_WIDTH-1:0] pipeline_s1_3;
    reg [DATA_WIDTH-1:0] pipeline_s1_4;
    reg [DATA_WIDTH-1:0] pipeline_s1_5;
    reg [DATA_WIDTH-1:0] pipeline_s1_6;
    reg [DATA_WIDTH-1:0] pipeline_s1_7;
    reg valid_s1;

    // stage 2 registers
    reg [DATA_WIDTH-1:0] pipeline_s2_0;
    reg [DATA_WIDTH-1:0] pipeline_s2_1;
    reg [DATA_WIDTH-1:0] pipeline_s2_2;
    reg [DATA_WIDTH-1:0] pipeline_s2_3;
    reg [DATA_WIDTH-1:0] pipeline_s2_4;
    reg [DATA_WIDTH-1:0] pipeline_s2_5;
    reg [DATA_WIDTH-1:0] pipeline_s2_6;
    reg [DATA_WIDTH-1:0] pipeline_s2_7;
    reg valid_s2;

    // stage 3 registers
    reg [DATA_WIDTH-1:0] pipeline_s3_0;
    reg [DATA_WIDTH-1:0] pipeline_s3_1;
    reg [DATA_WIDTH-1:0] pipeline_s3_2;
    reg [DATA_WIDTH-1:0] pipeline_s3_3;
    reg [DATA_WIDTH-1:0] pipeline_s3_4;
    reg [DATA_WIDTH-1:0] pipeline_s3_5;
    reg [DATA_WIDTH-1:0] pipeline_s3_6;
    reg [DATA_WIDTH-1:0] pipeline_s3_7;
    reg valid_s3;

    // stage 4 registers
    reg [DATA_WIDTH-1:0] pipeline_s4_0;
    reg [DATA_WIDTH-1:0] pipeline_s4_1;
    reg [DATA_WIDTH-1:0] pipeline_s4_2;
    reg [DATA_WIDTH-1:0] pipeline_s4_3;
    reg [DATA_WIDTH-1:0] pipeline_s4_4;
    reg [DATA_WIDTH-1:0] pipeline_s4_5;
    reg [DATA_WIDTH-1:0] pipeline_s4_6;
    reg [DATA_WIDTH-1:0] pipeline_s4_7;
    reg valid_s4;

    // stage 5 registers
    reg [DATA_WIDTH-1:0] pipeline_s5_0;
    reg [DATA_WIDTH-1:0] pipeline_s5_1;
    reg [DATA_WIDTH-1:0] pipeline_s5_2;
    reg [DATA_WIDTH-1:0] pipeline_s5_3;
    reg [DATA_WIDTH-1:0] pipeline_s5_4;
    reg [DATA_WIDTH-1:0] pipeline_s5_5;
    reg [DATA_WIDTH-1:0] pipeline_s5_6;
    reg [DATA_WIDTH-1:0] pipeline_s5_7;
    reg valid_s5;

    // stage 6 registers
    reg [DATA_WIDTH-1:0] pipeline_s6_0;
    reg [DATA_WIDTH-1:0] pipeline_s6_1;
    reg [DATA_WIDTH-1:0] pipeline_s6_2;
    reg [DATA_WIDTH-1:0] pipeline_s6_3;
    reg [DATA_WIDTH-1:0] pipeline_s6_4;
    reg [DATA_WIDTH-1:0] pipeline_s6_5;
    reg [DATA_WIDTH-1:0] pipeline_s6_6;
    reg [DATA_WIDTH-1:0] pipeline_s6_7;
    reg valid_s6;

    // combinational outputs produced by stage 0 -> latched into stage 1
    wire [DATA_WIDTH-1:0] next_s0_0;
    wire [DATA_WIDTH-1:0] next_s0_1;
    wire [DATA_WIDTH-1:0] next_s0_2;
    wire [DATA_WIDTH-1:0] next_s0_3;
    wire [DATA_WIDTH-1:0] next_s0_4;
    wire [DATA_WIDTH-1:0] next_s0_5;
    wire [DATA_WIDTH-1:0] next_s0_6;
    wire [DATA_WIDTH-1:0] next_s0_7;
    wire next_valid_s0;

    // combinational outputs produced by stage 1 -> latched into stage 2
    wire [DATA_WIDTH-1:0] next_s1_0;
    wire [DATA_WIDTH-1:0] next_s1_1;
    wire [DATA_WIDTH-1:0] next_s1_2;
    wire [DATA_WIDTH-1:0] next_s1_3;
    wire [DATA_WIDTH-1:0] next_s1_4;
    wire [DATA_WIDTH-1:0] next_s1_5;
    wire [DATA_WIDTH-1:0] next_s1_6;
    wire [DATA_WIDTH-1:0] next_s1_7;
    wire next_valid_s1;

    // combinational outputs produced by stage 2 -> latched into stage 3
    wire [DATA_WIDTH-1:0] next_s2_0;
    wire [DATA_WIDTH-1:0] next_s2_1;
    wire [DATA_WIDTH-1:0] next_s2_2;
    wire [DATA_WIDTH-1:0] next_s2_3;
    wire [DATA_WIDTH-1:0] next_s2_4;
    wire [DATA_WIDTH-1:0] next_s2_5;
    wire [DATA_WIDTH-1:0] next_s2_6;
    wire [DATA_WIDTH-1:0] next_s2_7;
    wire next_valid_s2;

    // combinational outputs produced by stage 3 -> latched into stage 4
    wire [DATA_WIDTH-1:0] next_s3_0;
    wire [DATA_WIDTH-1:0] next_s3_1;
    wire [DATA_WIDTH-1:0] next_s3_2;
    wire [DATA_WIDTH-1:0] next_s3_3;
    wire [DATA_WIDTH-1:0] next_s3_4;
    wire [DATA_WIDTH-1:0] next_s3_5;
    wire [DATA_WIDTH-1:0] next_s3_6;
    wire [DATA_WIDTH-1:0] next_s3_7;
    wire next_valid_s3;

    // combinational outputs produced by stage 4 -> latched into stage 5
    wire [DATA_WIDTH-1:0] next_s4_0;
    wire [DATA_WIDTH-1:0] next_s4_1;
    wire [DATA_WIDTH-1:0] next_s4_2;
    wire [DATA_WIDTH-1:0] next_s4_3;
    wire [DATA_WIDTH-1:0] next_s4_4;
    wire [DATA_WIDTH-1:0] next_s4_5;
    wire [DATA_WIDTH-1:0] next_s4_6;
    wire [DATA_WIDTH-1:0] next_s4_7;
    wire next_valid_s4;

    // combinational outputs produced by stage 5 -> latched into stage 6
    wire [DATA_WIDTH-1:0] next_s5_0;
    wire [DATA_WIDTH-1:0] next_s5_1;
    wire [DATA_WIDTH-1:0] next_s5_2;
    wire [DATA_WIDTH-1:0] next_s5_3;
    wire [DATA_WIDTH-1:0] next_s5_4;
    wire [DATA_WIDTH-1:0] next_s5_5;
    wire [DATA_WIDTH-1:0] next_s5_6;
    wire [DATA_WIDTH-1:0] next_s5_7;
    wire next_valid_s5;

    // Stage 0: load inputs when start asserted
    integer li;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipeline_s0_0 <= {DATA_WIDTH{1'b0}};
            pipeline_s0_1 <= {DATA_WIDTH{1'b0}};
            pipeline_s0_2 <= {DATA_WIDTH{1'b0}};
            pipeline_s0_3 <= {DATA_WIDTH{1'b0}};
            pipeline_s0_4 <= {DATA_WIDTH{1'b0}};
            pipeline_s0_5 <= {DATA_WIDTH{1'b0}};
            pipeline_s0_6 <= {DATA_WIDTH{1'b0}};
            pipeline_s0_7 <= {DATA_WIDTH{1'b0}};
            valid_s0 <= 1'b0;
        end else begin
            valid_s0 <= start;
            if (start) begin
                pipeline_s0_0 <= data_in[0*DATA_WIDTH +: DATA_WIDTH];
                pipeline_s0_1 <= data_in[1*DATA_WIDTH +: DATA_WIDTH];
                pipeline_s0_2 <= data_in[2*DATA_WIDTH +: DATA_WIDTH];
                pipeline_s0_3 <= data_in[3*DATA_WIDTH +: DATA_WIDTH];
                pipeline_s0_4 <= data_in[4*DATA_WIDTH +: DATA_WIDTH];
                pipeline_s0_5 <= data_in[5*DATA_WIDTH +: DATA_WIDTH];
                pipeline_s0_6 <= data_in[6*DATA_WIDTH +: DATA_WIDTH];
                pipeline_s0_7 <= data_in[7*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end

    // ----------------- stage 0: k=2, j=1 -----------------
    assign next_valid_s0 = valid_s0;
    wire [DATA_WIDTH-1:0] cs_0_0_1_min;
    wire [DATA_WIDTH-1:0] cs_0_0_1_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s0_0_1 (
        .in_a(pipeline_s0_0),
        .in_b(pipeline_s0_1),
        .ascending(1'b1),
        .out_min(cs_0_0_1_min),
        .out_max(cs_0_0_1_max)
    );
    wire [DATA_WIDTH-1:0] cs_0_2_3_min;
    wire [DATA_WIDTH-1:0] cs_0_2_3_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s0_2_3 (
        .in_a(pipeline_s0_2),
        .in_b(pipeline_s0_3),
        .ascending(1'b0),
        .out_min(cs_0_2_3_min),
        .out_max(cs_0_2_3_max)
    );
    wire [DATA_WIDTH-1:0] cs_0_4_5_min;
    wire [DATA_WIDTH-1:0] cs_0_4_5_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s0_4_5 (
        .in_a(pipeline_s0_4),
        .in_b(pipeline_s0_5),
        .ascending(1'b1),
        .out_min(cs_0_4_5_min),
        .out_max(cs_0_4_5_max)
    );
    wire [DATA_WIDTH-1:0] cs_0_6_7_min;
    wire [DATA_WIDTH-1:0] cs_0_6_7_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s0_6_7 (
        .in_a(pipeline_s0_6),
        .in_b(pipeline_s0_7),
        .ascending(1'b0),
        .out_min(cs_0_6_7_min),
        .out_max(cs_0_6_7_max)
    );
    assign next_s0_0 = cs_0_0_1_min;
    assign next_s0_1 = cs_0_0_1_max;
    assign next_s0_2 = cs_0_2_3_min;
    assign next_s0_3 = cs_0_2_3_max;
    assign next_s0_4 = cs_0_4_5_min;
    assign next_s0_5 = cs_0_4_5_max;
    assign next_s0_6 = cs_0_6_7_min;
    assign next_s0_7 = cs_0_6_7_max;

    // latch stage 0 outputs into stage 1 registers
    integer li_s0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1 <= 1'b0;
            pipeline_s1_0 <= {DATA_WIDTH{1'b0}};
            pipeline_s1_1 <= {DATA_WIDTH{1'b0}};
            pipeline_s1_2 <= {DATA_WIDTH{1'b0}};
            pipeline_s1_3 <= {DATA_WIDTH{1'b0}};
            pipeline_s1_4 <= {DATA_WIDTH{1'b0}};
            pipeline_s1_5 <= {DATA_WIDTH{1'b0}};
            pipeline_s1_6 <= {DATA_WIDTH{1'b0}};
            pipeline_s1_7 <= {DATA_WIDTH{1'b0}};
        end else begin
            valid_s1 <= next_valid_s0;
            pipeline_s1_0 <= next_s0_0;
            pipeline_s1_1 <= next_s0_1;
            pipeline_s1_2 <= next_s0_2;
            pipeline_s1_3 <= next_s0_3;
            pipeline_s1_4 <= next_s0_4;
            pipeline_s1_5 <= next_s0_5;
            pipeline_s1_6 <= next_s0_6;
            pipeline_s1_7 <= next_s0_7;
        end
    end

    // ----------------- stage 1: k=4, j=2 -----------------
    assign next_valid_s1 = valid_s1;
    wire [DATA_WIDTH-1:0] cs_1_0_2_min;
    wire [DATA_WIDTH-1:0] cs_1_0_2_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s1_0_2 (
        .in_a(pipeline_s1_0),
        .in_b(pipeline_s1_2),
        .ascending(1'b1),
        .out_min(cs_1_0_2_min),
        .out_max(cs_1_0_2_max)
    );
    wire [DATA_WIDTH-1:0] cs_1_1_3_min;
    wire [DATA_WIDTH-1:0] cs_1_1_3_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s1_1_3 (
        .in_a(pipeline_s1_1),
        .in_b(pipeline_s1_3),
        .ascending(1'b1),
        .out_min(cs_1_1_3_min),
        .out_max(cs_1_1_3_max)
    );
    wire [DATA_WIDTH-1:0] cs_1_4_6_min;
    wire [DATA_WIDTH-1:0] cs_1_4_6_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s1_4_6 (
        .in_a(pipeline_s1_4),
        .in_b(pipeline_s1_6),
        .ascending(1'b0),
        .out_min(cs_1_4_6_min),
        .out_max(cs_1_4_6_max)
    );
    wire [DATA_WIDTH-1:0] cs_1_5_7_min;
    wire [DATA_WIDTH-1:0] cs_1_5_7_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s1_5_7 (
        .in_a(pipeline_s1_5),
        .in_b(pipeline_s1_7),
        .ascending(1'b0),
        .out_min(cs_1_5_7_min),
        .out_max(cs_1_5_7_max)
    );
    assign next_s1_0 = cs_1_0_2_min;
    assign next_s1_1 = cs_1_1_3_min;
    assign next_s1_2 = cs_1_0_2_max;
    assign next_s1_3 = cs_1_1_3_max;
    assign next_s1_4 = cs_1_4_6_min;
    assign next_s1_5 = cs_1_5_7_min;
    assign next_s1_6 = cs_1_4_6_max;
    assign next_s1_7 = cs_1_5_7_max;

    // latch stage 1 outputs into stage 2 registers
    integer li_s1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2 <= 1'b0;
            pipeline_s2_0 <= {DATA_WIDTH{1'b0}};
            pipeline_s2_1 <= {DATA_WIDTH{1'b0}};
            pipeline_s2_2 <= {DATA_WIDTH{1'b0}};
            pipeline_s2_3 <= {DATA_WIDTH{1'b0}};
            pipeline_s2_4 <= {DATA_WIDTH{1'b0}};
            pipeline_s2_5 <= {DATA_WIDTH{1'b0}};
            pipeline_s2_6 <= {DATA_WIDTH{1'b0}};
            pipeline_s2_7 <= {DATA_WIDTH{1'b0}};
        end else begin
            valid_s2 <= next_valid_s1;
            pipeline_s2_0 <= next_s1_0;
            pipeline_s2_1 <= next_s1_1;
            pipeline_s2_2 <= next_s1_2;
            pipeline_s2_3 <= next_s1_3;
            pipeline_s2_4 <= next_s1_4;
            pipeline_s2_5 <= next_s1_5;
            pipeline_s2_6 <= next_s1_6;
            pipeline_s2_7 <= next_s1_7;
        end
    end

    // ----------------- stage 2: k=4, j=1 -----------------
    assign next_valid_s2 = valid_s2;
    wire [DATA_WIDTH-1:0] cs_2_0_1_min;
    wire [DATA_WIDTH-1:0] cs_2_0_1_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s2_0_1 (
        .in_a(pipeline_s2_0),
        .in_b(pipeline_s2_1),
        .ascending(1'b1),
        .out_min(cs_2_0_1_min),
        .out_max(cs_2_0_1_max)
    );
    wire [DATA_WIDTH-1:0] cs_2_2_3_min;
    wire [DATA_WIDTH-1:0] cs_2_2_3_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s2_2_3 (
        .in_a(pipeline_s2_2),
        .in_b(pipeline_s2_3),
        .ascending(1'b1),
        .out_min(cs_2_2_3_min),
        .out_max(cs_2_2_3_max)
    );
    wire [DATA_WIDTH-1:0] cs_2_4_5_min;
    wire [DATA_WIDTH-1:0] cs_2_4_5_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s2_4_5 (
        .in_a(pipeline_s2_4),
        .in_b(pipeline_s2_5),
        .ascending(1'b0),
        .out_min(cs_2_4_5_min),
        .out_max(cs_2_4_5_max)
    );
    wire [DATA_WIDTH-1:0] cs_2_6_7_min;
    wire [DATA_WIDTH-1:0] cs_2_6_7_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s2_6_7 (
        .in_a(pipeline_s2_6),
        .in_b(pipeline_s2_7),
        .ascending(1'b0),
        .out_min(cs_2_6_7_min),
        .out_max(cs_2_6_7_max)
    );
    assign next_s2_0 = cs_2_0_1_min;
    assign next_s2_1 = cs_2_0_1_max;
    assign next_s2_2 = cs_2_2_3_min;
    assign next_s2_3 = cs_2_2_3_max;
    assign next_s2_4 = cs_2_4_5_min;
    assign next_s2_5 = cs_2_4_5_max;
    assign next_s2_6 = cs_2_6_7_min;
    assign next_s2_7 = cs_2_6_7_max;

    // latch stage 2 outputs into stage 3 registers
    integer li_s2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s3 <= 1'b0;
            pipeline_s3_0 <= {DATA_WIDTH{1'b0}};
            pipeline_s3_1 <= {DATA_WIDTH{1'b0}};
            pipeline_s3_2 <= {DATA_WIDTH{1'b0}};
            pipeline_s3_3 <= {DATA_WIDTH{1'b0}};
            pipeline_s3_4 <= {DATA_WIDTH{1'b0}};
            pipeline_s3_5 <= {DATA_WIDTH{1'b0}};
            pipeline_s3_6 <= {DATA_WIDTH{1'b0}};
            pipeline_s3_7 <= {DATA_WIDTH{1'b0}};
        end else begin
            valid_s3 <= next_valid_s2;
            pipeline_s3_0 <= next_s2_0;
            pipeline_s3_1 <= next_s2_1;
            pipeline_s3_2 <= next_s2_2;
            pipeline_s3_3 <= next_s2_3;
            pipeline_s3_4 <= next_s2_4;
            pipeline_s3_5 <= next_s2_5;
            pipeline_s3_6 <= next_s2_6;
            pipeline_s3_7 <= next_s2_7;
        end
    end

    // ----------------- stage 3: k=8, j=4 -----------------
    assign next_valid_s3 = valid_s3;
    wire [DATA_WIDTH-1:0] cs_3_0_4_min;
    wire [DATA_WIDTH-1:0] cs_3_0_4_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s3_0_4 (
        .in_a(pipeline_s3_0),
        .in_b(pipeline_s3_4),
        .ascending(1'b1),
        .out_min(cs_3_0_4_min),
        .out_max(cs_3_0_4_max)
    );
    wire [DATA_WIDTH-1:0] cs_3_1_5_min;
    wire [DATA_WIDTH-1:0] cs_3_1_5_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s3_1_5 (
        .in_a(pipeline_s3_1),
        .in_b(pipeline_s3_5),
        .ascending(1'b1),
        .out_min(cs_3_1_5_min),
        .out_max(cs_3_1_5_max)
    );
    wire [DATA_WIDTH-1:0] cs_3_2_6_min;
    wire [DATA_WIDTH-1:0] cs_3_2_6_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s3_2_6 (
        .in_a(pipeline_s3_2),
        .in_b(pipeline_s3_6),
        .ascending(1'b1),
        .out_min(cs_3_2_6_min),
        .out_max(cs_3_2_6_max)
    );
    wire [DATA_WIDTH-1:0] cs_3_3_7_min;
    wire [DATA_WIDTH-1:0] cs_3_3_7_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s3_3_7 (
        .in_a(pipeline_s3_3),
        .in_b(pipeline_s3_7),
        .ascending(1'b1),
        .out_min(cs_3_3_7_min),
        .out_max(cs_3_3_7_max)
    );
    assign next_s3_0 = cs_3_0_4_min;
    assign next_s3_1 = cs_3_1_5_min;
    assign next_s3_2 = cs_3_2_6_min;
    assign next_s3_3 = cs_3_3_7_min;
    assign next_s3_4 = cs_3_0_4_max;
    assign next_s3_5 = cs_3_1_5_max;
    assign next_s3_6 = cs_3_2_6_max;
    assign next_s3_7 = cs_3_3_7_max;

    // latch stage 3 outputs into stage 4 registers
    integer li_s3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s4 <= 1'b0;
            pipeline_s4_0 <= {DATA_WIDTH{1'b0}};
            pipeline_s4_1 <= {DATA_WIDTH{1'b0}};
            pipeline_s4_2 <= {DATA_WIDTH{1'b0}};
            pipeline_s4_3 <= {DATA_WIDTH{1'b0}};
            pipeline_s4_4 <= {DATA_WIDTH{1'b0}};
            pipeline_s4_5 <= {DATA_WIDTH{1'b0}};
            pipeline_s4_6 <= {DATA_WIDTH{1'b0}};
            pipeline_s4_7 <= {DATA_WIDTH{1'b0}};
        end else begin
            valid_s4 <= next_valid_s3;
            pipeline_s4_0 <= next_s3_0;
            pipeline_s4_1 <= next_s3_1;
            pipeline_s4_2 <= next_s3_2;
            pipeline_s4_3 <= next_s3_3;
            pipeline_s4_4 <= next_s3_4;
            pipeline_s4_5 <= next_s3_5;
            pipeline_s4_6 <= next_s3_6;
            pipeline_s4_7 <= next_s3_7;
        end
    end

    // ----------------- stage 4: k=8, j=2 -----------------
    assign next_valid_s4 = valid_s4;
    wire [DATA_WIDTH-1:0] cs_4_0_2_min;
    wire [DATA_WIDTH-1:0] cs_4_0_2_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s4_0_2 (
        .in_a(pipeline_s4_0),
        .in_b(pipeline_s4_2),
        .ascending(1'b1),
        .out_min(cs_4_0_2_min),
        .out_max(cs_4_0_2_max)
    );
    wire [DATA_WIDTH-1:0] cs_4_1_3_min;
    wire [DATA_WIDTH-1:0] cs_4_1_3_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s4_1_3 (
        .in_a(pipeline_s4_1),
        .in_b(pipeline_s4_3),
        .ascending(1'b1),
        .out_min(cs_4_1_3_min),
        .out_max(cs_4_1_3_max)
    );
    wire [DATA_WIDTH-1:0] cs_4_4_6_min;
    wire [DATA_WIDTH-1:0] cs_4_4_6_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s4_4_6 (
        .in_a(pipeline_s4_4),
        .in_b(pipeline_s4_6),
        .ascending(1'b1),
        .out_min(cs_4_4_6_min),
        .out_max(cs_4_4_6_max)
    );
    wire [DATA_WIDTH-1:0] cs_4_5_7_min;
    wire [DATA_WIDTH-1:0] cs_4_5_7_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s4_5_7 (
        .in_a(pipeline_s4_5),
        .in_b(pipeline_s4_7),
        .ascending(1'b1),
        .out_min(cs_4_5_7_min),
        .out_max(cs_4_5_7_max)
    );
    assign next_s4_0 = cs_4_0_2_min;
    assign next_s4_1 = cs_4_1_3_min;
    assign next_s4_2 = cs_4_0_2_max;
    assign next_s4_3 = cs_4_1_3_max;
    assign next_s4_4 = cs_4_4_6_min;
    assign next_s4_5 = cs_4_5_7_min;
    assign next_s4_6 = cs_4_4_6_max;
    assign next_s4_7 = cs_4_5_7_max;

    // latch stage 4 outputs into stage 5 registers
    integer li_s4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s5 <= 1'b0;
            pipeline_s5_0 <= {DATA_WIDTH{1'b0}};
            pipeline_s5_1 <= {DATA_WIDTH{1'b0}};
            pipeline_s5_2 <= {DATA_WIDTH{1'b0}};
            pipeline_s5_3 <= {DATA_WIDTH{1'b0}};
            pipeline_s5_4 <= {DATA_WIDTH{1'b0}};
            pipeline_s5_5 <= {DATA_WIDTH{1'b0}};
            pipeline_s5_6 <= {DATA_WIDTH{1'b0}};
            pipeline_s5_7 <= {DATA_WIDTH{1'b0}};
        end else begin
            valid_s5 <= next_valid_s4;
            pipeline_s5_0 <= next_s4_0;
            pipeline_s5_1 <= next_s4_1;
            pipeline_s5_2 <= next_s4_2;
            pipeline_s5_3 <= next_s4_3;
            pipeline_s5_4 <= next_s4_4;
            pipeline_s5_5 <= next_s4_5;
            pipeline_s5_6 <= next_s4_6;
            pipeline_s5_7 <= next_s4_7;
        end
    end

    // ----------------- stage 5: k=8, j=1 -----------------
    assign next_valid_s5 = valid_s5;
    wire [DATA_WIDTH-1:0] cs_5_0_1_min;
    wire [DATA_WIDTH-1:0] cs_5_0_1_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s5_0_1 (
        .in_a(pipeline_s5_0),
        .in_b(pipeline_s5_1),
        .ascending(1'b1),
        .out_min(cs_5_0_1_min),
        .out_max(cs_5_0_1_max)
    );
    wire [DATA_WIDTH-1:0] cs_5_2_3_min;
    wire [DATA_WIDTH-1:0] cs_5_2_3_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s5_2_3 (
        .in_a(pipeline_s5_2),
        .in_b(pipeline_s5_3),
        .ascending(1'b1),
        .out_min(cs_5_2_3_min),
        .out_max(cs_5_2_3_max)
    );
    wire [DATA_WIDTH-1:0] cs_5_4_5_min;
    wire [DATA_WIDTH-1:0] cs_5_4_5_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s5_4_5 (
        .in_a(pipeline_s5_4),
        .in_b(pipeline_s5_5),
        .ascending(1'b1),
        .out_min(cs_5_4_5_min),
        .out_max(cs_5_4_5_max)
    );
    wire [DATA_WIDTH-1:0] cs_5_6_7_min;
    wire [DATA_WIDTH-1:0] cs_5_6_7_max;
    compare_swap #(.DATA_WIDTH(DATA_WIDTH)) cs_s5_6_7 (
        .in_a(pipeline_s5_6),
        .in_b(pipeline_s5_7),
        .ascending(1'b1),
        .out_min(cs_5_6_7_min),
        .out_max(cs_5_6_7_max)
    );
    assign next_s5_0 = cs_5_0_1_min;
    assign next_s5_1 = cs_5_0_1_max;
    assign next_s5_2 = cs_5_2_3_min;
    assign next_s5_3 = cs_5_2_3_max;
    assign next_s5_4 = cs_5_4_5_min;
    assign next_s5_5 = cs_5_4_5_max;
    assign next_s5_6 = cs_5_6_7_min;
    assign next_s5_7 = cs_5_6_7_max;

    // latch stage 5 outputs into stage 6 registers
    integer li_s5;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s6 <= 1'b0;
            pipeline_s6_0 <= {DATA_WIDTH{1'b0}};
            pipeline_s6_1 <= {DATA_WIDTH{1'b0}};
            pipeline_s6_2 <= {DATA_WIDTH{1'b0}};
            pipeline_s6_3 <= {DATA_WIDTH{1'b0}};
            pipeline_s6_4 <= {DATA_WIDTH{1'b0}};
            pipeline_s6_5 <= {DATA_WIDTH{1'b0}};
            pipeline_s6_6 <= {DATA_WIDTH{1'b0}};
            pipeline_s6_7 <= {DATA_WIDTH{1'b0}};
        end else begin
            valid_s6 <= next_valid_s5;
            pipeline_s6_0 <= next_s5_0;
            pipeline_s6_1 <= next_s5_1;
            pipeline_s6_2 <= next_s5_2;
            pipeline_s6_3 <= next_s5_3;
            pipeline_s6_4 <= next_s5_4;
            pipeline_s6_5 <= next_s5_5;
            pipeline_s6_6 <= next_s5_6;
            pipeline_s6_7 <= next_s5_7;
        end
    end

    // Output: Top-K (the K smallest elements at the start of final stage)
    integer oi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid <= 1'b0;
            data_out <= {DATA_WIDTH*K{1'b0}};
        end else begin
            valid <= valid_s6;
            if (valid_s6) begin
                data_out[0*DATA_WIDTH +: DATA_WIDTH] <= pipeline_s6_0;
                data_out[1*DATA_WIDTH +: DATA_WIDTH] <= pipeline_s6_1;
                data_out[2*DATA_WIDTH +: DATA_WIDTH] <= pipeline_s6_2;
                data_out[3*DATA_WIDTH +: DATA_WIDTH] <= pipeline_s6_3;
            end
        end
    end

endmodule


// Module 2: 双路归并单元
// ============================================
// module merge_unit #(
//     parameter DATA_WIDTH = 16,
//     parameter K = 4
// )(
//     input wire clk,
//     input wire rst_n,
//     input wire start,
//     input wire [DATA_WIDTH*K-1:0] list_a,
//     input wire [DATA_WIDTH*K-1:0] list_b,
//     input wire [3:0] len_a,
//     input wire [3:0] len_b,
//     output reg [DATA_WIDTH*K-1:0] merged,
//     output reg [3:0] merged_len,
//     output reg valid
// );
//     reg [DATA_WIDTH-1:0] a_buf [0:K-1];
//     reg [DATA_WIDTH-1:0] b_buf [0:K-1];
//     reg [DATA_WIDTH-1:0] out_buf [0:K-1];

//     reg [3:0] i, j, out_idx;
//     reg [2:0] state;
//     reg output_done;

//     localparam IDLE = 0, LOAD = 1, MERGE = 2, OUTPUT = 3;

//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             state <= IDLE;
//             valid <= 0;
//             output_done <= 0;
//             i <= 0; j <= 0; out_idx <= 0;
//             merged_len <= 0;
//         end else begin
//             valid <= 0;  // 默认清零
//             output_done <= 0;

//             case (state)
//                 IDLE: begin
//                     if (start) begin
//                         i <= 0; j <= 0; out_idx <= 0;
//                         state <= LOAD;
//                     end
//                 end

//                 LOAD: begin
//                     a_buf[0] <= list_a[0*DATA_WIDTH +: DATA_WIDTH];
//                     a_buf[1] <= list_a[1*DATA_WIDTH +: DATA_WIDTH];
//                     a_buf[2] <= list_a[2*DATA_WIDTH +: DATA_WIDTH];
//                     a_buf[3] <= list_a[3*DATA_WIDTH +: DATA_WIDTH];
//                     b_buf[0] <= list_b[0*DATA_WIDTH +: DATA_WIDTH];
//                     b_buf[1] <= list_b[1*DATA_WIDTH +: DATA_WIDTH];
//                     b_buf[2] <= list_b[2*DATA_WIDTH +: DATA_WIDTH];
//                     b_buf[3] <= list_b[3*DATA_WIDTH +: DATA_WIDTH];
//                     state <= MERGE;
//                 end

//                 MERGE: begin
//                     if (out_idx >= K || (i >= len_a && j >= len_b)) begin
//                         state <= OUTPUT;
//                     end else if (i >= len_a) begin
//                         out_buf[out_idx] <= b_buf[j];
//                         j <= j + 1;
//                         out_idx <= out_idx + 1;
//                     end else if (j >= len_b) begin
//                         out_buf[out_idx] <= a_buf[i];
//                         i <= i + 1;
//                         out_idx <= out_idx + 1;
//                     end else if (a_buf[i] < b_buf[j]) begin
//                         out_buf[out_idx] <= a_buf[i];
//                         i <= i + 1;
//                         out_idx <= out_idx + 1;
//                     end else begin
//                         out_buf[out_idx] <= b_buf[j];
//                         j <= j + 1;
//                         out_idx <= out_idx + 1;
//                     end
//                 end

//                 OUTPUT: begin
//                     merged[0*DATA_WIDTH +: DATA_WIDTH] <= out_buf[0];
//                     merged[1*DATA_WIDTH +: DATA_WIDTH] <= out_buf[1];
//                     merged[2*DATA_WIDTH +: DATA_WIDTH] <= out_buf[2];
//                     merged[3*DATA_WIDTH +: DATA_WIDTH] <= out_buf[3];
//                     merged_len <= out_idx;
//                     output_done <= 1;
//                     state <= IDLE;
//                 end
//             endcase
//         end
//     end

//     // valid 延迟一个周期
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) valid <= 0;
//         else valid <= output_done;
//     end
// endmodule

module merge_unit #(
    parameter DATA_WIDTH = 16,
    parameter K = 4
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [DATA_WIDTH*K-1:0] list_a,
    input wire [DATA_WIDTH*K-1:0] list_b,
    input wire [3:0] len_a,
    input wire [3:0] len_b,
    output reg [DATA_WIDTH*K-1:0] merged,
    output reg [3:0] merged_len,
    output reg valid
);

    reg [DATA_WIDTH-1:0] a_buf [0:K-1];
    reg [DATA_WIDTH-1:0] b_buf [0:K-1];
    reg [DATA_WIDTH-1:0] out_buf [0:K-1];

    reg [3:0] i, j, out_idx;
    reg [2:0] state;
    reg output_done;

    localparam IDLE=0, LOAD=1, MERGE=2, OUTPUT=3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid <= 0;
            output_done <= 0;
            i <= 0; j <= 0; out_idx <= 0;
            merged_len <= 0;
        end else begin
            valid <= 0;
            output_done <= 0;
            case(state)
                IDLE: begin
                    if(start) begin
                        i <= 0; j <= 0; out_idx <= 0;
                        state <= LOAD;
                    end
                end
                LOAD: begin
                    a_buf[0] <= list_a[0*DATA_WIDTH +: DATA_WIDTH];
                    b_buf[0] <= list_b[0*DATA_WIDTH +: DATA_WIDTH];
                    a_buf[1] <= list_a[1*DATA_WIDTH +: DATA_WIDTH];
                    b_buf[1] <= list_b[1*DATA_WIDTH +: DATA_WIDTH];
                    a_buf[2] <= list_a[2*DATA_WIDTH +: DATA_WIDTH];
                    b_buf[2] <= list_b[2*DATA_WIDTH +: DATA_WIDTH];
                    a_buf[3] <= list_a[3*DATA_WIDTH +: DATA_WIDTH];
                    b_buf[3] <= list_b[3*DATA_WIDTH +: DATA_WIDTH];
                    state <= MERGE;
                end
                MERGE: begin
                    if(out_idx >= K || (i>=len_a && j>=len_b)) begin
                        state <= OUTPUT;
                    end else if(i>=len_a) begin
                        out_buf[out_idx] <= b_buf[j];
                        j <= j + 1;
                        out_idx <= out_idx + 1;
                    end else if(j>=len_b) begin
                        out_buf[out_idx] <= a_buf[i];
                        i <= i + 1;
                        out_idx <= out_idx + 1;
                    end else if(a_buf[i] < b_buf[j]) begin
                        out_buf[out_idx] <= a_buf[i];
                        i <= i + 1;
                        out_idx <= out_idx + 1;
                    end else begin
                        out_buf[out_idx] <= b_buf[j];
                        j <= j + 1;
                        out_idx <= out_idx + 1;
                    end
                end
                OUTPUT: begin
                    merged[0*DATA_WIDTH +: DATA_WIDTH] <= out_buf[0];
                    merged[1*DATA_WIDTH +: DATA_WIDTH] <= out_buf[1];
                    merged[2*DATA_WIDTH +: DATA_WIDTH] <= out_buf[2];
                    merged[3*DATA_WIDTH +: DATA_WIDTH] <= out_buf[3];
                    merged_len <= out_idx;
                    output_done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) valid <= 0;
        else valid <= output_done;
    end

endmodule


// // ============================================
// // Module 3: 顶层控制器（时分复用merge unit）
// // ============================================
// module topk_merge_sort #(
//     parameter DATA_WIDTH = 16,
//     parameter BLOCK_SIZE = 8,
//     parameter NUM_BLOCKS = 4,
//     parameter K = 4
// )(
//     input wire clk,
//     input wire rst_n,
//     input wire start,
//     input wire [DATA_WIDTH*BLOCK_SIZE*NUM_BLOCKS-1:0] data_in,
//     output reg [DATA_WIDTH*K-1:0] topk_out,
//     output reg done
// );
//     // 块缓存
//     reg [DATA_WIDTH*K-1:0] blocks [0:NUM_BLOCKS-1];
//     reg [3:0] block_lens [0:NUM_BLOCKS-1];

//     // block_sorter 实例
//     wire [DATA_WIDTH*K-1:0] sorted_block [0:NUM_BLOCKS-1];
//     wire sorter_valid [0:NUM_BLOCKS-1];
//     reg sorter_start [0:NUM_BLOCKS-1];

//     genvar g;
//     generate
//         for (g = 0; g < NUM_BLOCKS; g = g + 1) begin : gen_sorters
//             block_sorter #(
//                 .DATA_WIDTH(DATA_WIDTH),
//                 .BLOCK_SIZE(BLOCK_SIZE),
//                 .K(K)
//             ) sorter_inst (
//                 .clk(clk),
//                 .rst_n(rst_n),
//                 .start(sorter_start[g]),
//                 .data_in(data_in[g*BLOCK_SIZE*DATA_WIDTH +: BLOCK_SIZE*DATA_WIDTH]),
//                 .data_out(sorted_block[g]),
//                 .valid(sorter_valid[g])
//             );
//         end
//     endgenerate

//     // 归并单元接口
//     reg merge_start;
//     wire merge_valid;
//     wire [DATA_WIDTH*K-1:0] merge_out;
//     wire [3:0] merge_out_len;
//     reg [3:0] merge_idx_a, merge_idx_b;

//     merge_unit #(
//         .DATA_WIDTH(DATA_WIDTH),
//         .K(K)
//     ) merger (
//         .clk(clk),
//         .rst_n(rst_n),
//         .start(merge_start),
//         .list_a(blocks[merge_idx_a]),
//         .list_b(blocks[merge_idx_b]),
//         .len_a(block_lens[merge_idx_a]),
//         .len_b(block_lens[merge_idx_b]),
//         .merged(merge_out),
//         .merged_len(merge_out_len),
//         .valid(merge_valid)
//     );

//     // 控制状态机
//     reg [3:0] state;
//     reg [3:0] num_active_blocks;
//     reg all_sorters_done;
//     integer i;

//     localparam IDLE = 0, START_SORT = 1, WAIT_SORT = 2,
//                MERGE_PHASE = 3, WAIT_MERGE = 4, UPDATE = 5, DONE = 6;

//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             state <= IDLE;
//             done <= 0;
//             merge_start <= 0;
//             all_sorters_done <= 0;
//             num_active_blocks <= 0;
//             for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
//                 sorter_start[i] <= 0;
//             end
//         end else begin
//             case (state)
//                 IDLE: begin
//                     done <= 0;
//                     if (start) begin
//                         num_active_blocks <= NUM_BLOCKS;
//                         for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
//                             sorter_start[i] <= 1;
//                         end
//                         state <= START_SORT;
//                     end
//                 end

//                 START_SORT: begin
//                     for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
//                         sorter_start[i] <= 0;
//                     end
//                     state <= WAIT_SORT;
//                 end

//                 WAIT_SORT: begin
//                     all_sorters_done = 1;
//                     for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
//                         if (!sorter_valid[i]) all_sorters_done = 0;
//                         if (sorter_valid[i]) begin
//                             blocks[i] <= sorted_block[i];
//                             block_lens[i] <= K;
//                         end
//                     end
//                     if (all_sorters_done) begin
//                         merge_idx_a <= 0;
//                         merge_idx_b <= 1;
//                         state <= MERGE_PHASE;
//                     end
//                 end

//                 MERGE_PHASE: begin
//                     if (num_active_blocks == 1) begin
//                         state <= DONE;
//                     end else begin
//                         merge_start <= 1;
//                         state <= WAIT_MERGE;
//                     end
//                 end

//                 WAIT_MERGE: begin
//                     merge_start <= 0;
//                     if (merge_valid) begin
//                         blocks[merge_idx_a] <= merge_out;
//                         block_lens[merge_idx_a] <= merge_out_len;
//                         state <= UPDATE;
//                     end
//                 end

//                 UPDATE: begin
//                     if (merge_idx_b + 1 < num_active_blocks) begin
//                         merge_idx_a <= merge_idx_a + 1;
//                         merge_idx_b <= merge_idx_b + 2;
//                     end else begin
//                         num_active_blocks <= (num_active_blocks + 1) >> 1;
//                         merge_idx_a <= 0;
//                         merge_idx_b <= 1;
//                     end
//                     state <= MERGE_PHASE;
//                 end

//                 DONE: begin
//                     topk_out <= blocks[0];
//                     done <= 1;
//                     state <= IDLE;
//                 end

//                 default: state <= IDLE;
//             endcase
//         end
//     end
// endmodule

`timescale 1ns/1ps
module topk_merge_sort #(
    parameter DATA_WIDTH = 16,
    parameter BLOCK_SIZE = 8,
    parameter TOTAL_DATA = 32,
    parameter K = 4
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [DATA_WIDTH*TOTAL_DATA-1:0] data_in,
    output reg [DATA_WIDTH*K-1:0] topk_out,
    output reg done
);
    // -------------------------------------------------
    // 计算批次数
    // -------------------------------------------------
    localparam NUM_BATCHES = (TOTAL_DATA + BLOCK_SIZE - 1)/BLOCK_SIZE;

    // -------------------------------------------------
    // 缓存
    // -------------------------------------------------
    reg [DATA_WIDTH*BLOCK_SIZE-1:0] batch_data;
    reg [DATA_WIDTH*K-1:0] topk_buffer [0:NUM_BATCHES-1];
    reg [3:0] batch_idx;
    reg [3:0] merge_idx_a, merge_idx_b;
    
    // -------------------------------------------------
    // sorter 接口
    // -------------------------------------------------
    wire [DATA_WIDTH*K-1:0] sorter_out;
    wire sorter_valid;
    reg sorter_start;

    block_sorter #(
        .DATA_WIDTH(DATA_WIDTH),
        .BLOCK_SIZE(BLOCK_SIZE),
        .K(K)
    ) sorter_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(sorter_start),
        .data_in(batch_data),
        .data_out(sorter_out),
        .valid(sorter_valid)
    );

    // -------------------------------------------------
    // FSM 状态机
    // -------------------------------------------------
    reg [3:0] state;
    localparam IDLE          = 0,
               LOAD_BATCH    = 1,
               WAIT_SORT    = 2,
               CHECK_MORE   = 3,
               MERGE_PHASE  = 4,
               WAIT_MERGE   = 5,
               CHECK_MERGE  = 6,
               DONE_STATE   = 7;

    integer i;
    reg [3:0] num_active;

    // -------------------------------------------------
    // FSM
    // -------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            sorter_start <= 0;
            batch_idx <= 0;
            merge_idx_a <= 0;
            merge_idx_b <= 1;
            num_active <= NUM_BATCHES;
            done <= 0;
            topk_out <= 0;
        end else begin
            sorter_start <= 0;  // 默认
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        batch_idx <= 0;
                        state <= LOAD_BATCH;
                    end
                end

                LOAD_BATCH: begin
                    batch_data <= data_in[batch_idx*BLOCK_SIZE*DATA_WIDTH +: BLOCK_SIZE*DATA_WIDTH];
                    sorter_start <= 1;
                    state <= WAIT_SORT;
                end

                WAIT_SORT: begin
                    if (sorter_valid) begin
                        topk_buffer[batch_idx] <= sorter_out;
                        batch_idx <= batch_idx + 1;
                        state <= CHECK_MORE;
                    end
                end

                CHECK_MORE: begin
                    if (batch_idx < NUM_BATCHES) state <= LOAD_BATCH;
                    else begin
                        // 所有批次 top-K 生成完成
                        merge_idx_a <= 0;
                        merge_idx_b <= 1;
                        num_active <= NUM_BATCHES;
                        state <= MERGE_PHASE;
                    end
                end

                MERGE_PHASE: begin
                    if (num_active == 1) state <= DONE_STATE;
                    else begin
                        batch_data <= {topk_buffer[merge_idx_a], topk_buffer[merge_idx_b]};
                        sorter_start <= 1;
                        state <= WAIT_MERGE;
                    end
                end

                WAIT_MERGE: begin
                    if (sorter_valid) begin
                        topk_buffer[merge_idx_a] <= sorter_out;
                        merge_idx_a <= merge_idx_a + 2;
                        merge_idx_b <= merge_idx_b + 2;
                        state <= CHECK_MERGE;
                    end
                end

                CHECK_MERGE: begin
                    if (merge_idx_b >= num_active) begin
                        num_active <= (num_active + 1) >> 1; // 每轮归并减少一半
                        merge_idx_a <= 0;
                        merge_idx_b <= 1;
                    end
                    state <= MERGE_PHASE;
                end

                DONE_STATE: begin
                    topk_out <= topk_buffer[0];
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule



module topk_decoder #(
    parameter DATA_WIDTH = 16,
    parameter K = 4
)(
    input  wire [DATA_WIDTH*K-1:0] packed_topk,
    output wire [DATA_WIDTH-1:0]   topk0,
    output wire [DATA_WIDTH-1:0]   topk1,
    output wire [DATA_WIDTH-1:0]   topk2,
    output wire [DATA_WIDTH-1:0]   topk3
);

    assign topk0 = packed_topk[0*DATA_WIDTH +: DATA_WIDTH];
    assign topk1 = packed_topk[1*DATA_WIDTH +: DATA_WIDTH];
    assign topk2 = packed_topk[2*DATA_WIDTH +: DATA_WIDTH];
    assign topk3 = packed_topk[3*DATA_WIDTH +: DATA_WIDTH];

endmodule
