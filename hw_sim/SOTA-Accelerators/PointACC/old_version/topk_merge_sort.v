// Module 2: 双路归并单元
// ============================================
module merge_unit #(
    parameter DATA_WIDTH = 16,
    parameter K = 4
)(
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

    localparam IDLE = 0, LOAD = 1, MERGE = 2, OUTPUT = 3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid <= 0;
            output_done <= 0;
            i <= 0; j <= 0; out_idx <= 0;
            merged_len <= 0;
        end else begin
            valid <= 0;  // 默认清零
            output_done <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        i <= 0; j <= 0; out_idx <= 0;
                        state <= LOAD;
                    end
                end

                LOAD: begin
                    a_buf[0] <= list_a[0*DATA_WIDTH +: DATA_WIDTH];
                    a_buf[1] <= list_a[1*DATA_WIDTH +: DATA_WIDTH];
                    a_buf[2] <= list_a[2*DATA_WIDTH +: DATA_WIDTH];
                    a_buf[3] <= list_a[3*DATA_WIDTH +: DATA_WIDTH];
                    b_buf[0] <= list_b[0*DATA_WIDTH +: DATA_WIDTH];
                    b_buf[1] <= list_b[1*DATA_WIDTH +: DATA_WIDTH];
                    b_buf[2] <= list_b[2*DATA_WIDTH +: DATA_WIDTH];
                    b_buf[3] <= list_b[3*DATA_WIDTH +: DATA_WIDTH];
                    state <= MERGE;
                end

                MERGE: begin
                    if (out_idx >= K || (i >= len_a && j >= len_b)) begin
                        state <= OUTPUT;
                    end else if (i >= len_a) begin
                        out_buf[out_idx] <= b_buf[j];
                        j <= j + 1;
                        out_idx <= out_idx + 1;
                    end else if (j >= len_b) begin
                        out_buf[out_idx] <= a_buf[i];
                        i <= i + 1;
                        out_idx <= out_idx + 1;
                    end else if (a_buf[i] < b_buf[j]) begin
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

    // valid 延迟一个周期
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid <= 0;
        else valid <= output_done;
    end
endmodule



// ============================================
// Module 3: 顶层控制器（时分复用merge unit）
// ============================================
module topk_merge_sort #(
    parameter DATA_WIDTH = 16,
    parameter BLOCK_SIZE = 8,
    parameter NUM_BLOCKS = 4,
    parameter K = 4
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [DATA_WIDTH*BLOCK_SIZE*NUM_BLOCKS-1:0] data_in,
    output reg [DATA_WIDTH*K-1:0] topk_out,
    output reg done
);
    // 块缓存
    reg [DATA_WIDTH*K-1:0] blocks [0:NUM_BLOCKS-1];
    reg [3:0] block_lens [0:NUM_BLOCKS-1];

    // block_sorter 实例
    wire [DATA_WIDTH*K-1:0] sorted_block [0:NUM_BLOCKS-1];
    wire sorter_valid [0:NUM_BLOCKS-1];
    reg sorter_start [0:NUM_BLOCKS-1];

    genvar g;
    generate
        for (g = 0; g < NUM_BLOCKS; g = g + 1) begin : gen_sorters
            block_sorter #(
                .DATA_WIDTH(DATA_WIDTH),
                .BLOCK_SIZE(BLOCK_SIZE),
                .K(K)
            ) sorter_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(sorter_start[g]),
                .data_in(data_in[g*BLOCK_SIZE*DATA_WIDTH +: BLOCK_SIZE*DATA_WIDTH]),
                .data_out(sorted_block[g]),
                .valid(sorter_valid[g])
            );  // 这里每个block会输出每个sorter排序的结果
        end
    endgenerate

    // 归并单元接口
    reg merge_start;
    wire merge_valid;
    wire [DATA_WIDTH*K-1:0] merge_out;
    wire [3:0] merge_out_len;
    reg [3:0] merge_idx_a, merge_idx_b;

    merge_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .K(K)
    ) merger (
        .clk(clk),
        .rst_n(rst_n),
        .start(merge_start),
        .list_a(blocks[merge_idx_a]),
        .list_b(blocks[merge_idx_b]),
        .len_a(block_lens[merge_idx_a]),
        .len_b(block_lens[merge_idx_b]),
        .merged(merge_out),
        .merged_len(merge_out_len),
        .valid(merge_valid)
    );

    // 控制状态机
    reg [3:0] state;
    reg [3:0] num_active_blocks;
    reg all_sorters_done;
    integer i;

    localparam IDLE = 0, START_SORT = 1, WAIT_SORT = 2,
               MERGE_PHASE = 3, WAIT_MERGE = 4, UPDATE = 5, DONE = 6;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 0;
            merge_start <= 0;
            all_sorters_done <= 0;
            num_active_blocks <= 0;
            for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
                sorter_start[i] <= 0;
            end
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        num_active_blocks <= NUM_BLOCKS;
                        for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
                            sorter_start[i] <= 1;
                        end
                        state <= START_SORT;
                    end
                end

                START_SORT: begin
                    for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
                        sorter_start[i] <= 0;
                    end
                    state <= WAIT_SORT;
                end

                WAIT_SORT: begin
                    all_sorters_done = 1;
                    for (i = 0; i < NUM_BLOCKS; i = i + 1) begin
                        if (!sorter_valid[i]) all_sorters_done = 0;
                        if (sorter_valid[i]) begin
                            blocks[i] <= sorted_block[i];
                            block_lens[i] <= K;
                        end
                    end
                    if (all_sorters_done) begin
                        merge_idx_a <= 0;
                        merge_idx_b <= 1;
                        state <= MERGE_PHASE;
                    end
                end

                MERGE_PHASE: begin
                    if (num_active_blocks == 1) begin
                        state <= DONE;
                    end else begin
                        merge_start <= 1;
                        state <= WAIT_MERGE;
                    end
                end

                WAIT_MERGE: begin
                    merge_start <= 0;
                    if (merge_valid) begin
                        blocks[merge_idx_a] <= merge_out;
                        block_lens[merge_idx_a] <= merge_out_len;
                        state <= UPDATE;
                    end
                end

                UPDATE: begin
                    if (merge_idx_b + 1 < num_active_blocks) begin
                        merge_idx_a <= merge_idx_a + 1;
                        merge_idx_b <= merge_idx_b + 2;
                    end else begin
                        num_active_blocks <= (num_active_blocks + 1) >> 1;
                        merge_idx_a <= 0;
                        merge_idx_b <= 1;
                    end
                    state <= MERGE_PHASE;
                end

                DONE: begin
                    topk_out <= blocks[0];
                    done <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule

