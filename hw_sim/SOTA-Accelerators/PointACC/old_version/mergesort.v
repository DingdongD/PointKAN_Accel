// 归并排序 - Verilog实现（标准端口声明）
// 采用迭代式归并排序，支持8个32位数据排序

module merge_sort #(
    parameter DATA_WIDTH = 32,
    parameter ARRAY_SIZE = 8
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    // 使用拼接的向量而非数组
    input wire [DATA_WIDTH*ARRAY_SIZE-1:0] data_in,
    output reg [DATA_WIDTH*ARRAY_SIZE-1:0] data_out,
    output reg done
);

    // 状态定义
    localparam IDLE = 3'd0;
    localparam MERGE = 3'd1;
    localparam COPY = 3'd2;
    localparam DONE = 3'd3;
    
    reg [2:0] state, next_state;
    reg [DATA_WIDTH-1:0] work_array [0:ARRAY_SIZE-1];
    reg [DATA_WIDTH-1:0] temp_array [0:ARRAY_SIZE-1];
    
    reg [3:0] merge_size;
    reg [3:0] left_start;
    reg [3:0] copy_idx;
    
    integer i;
    
    // 输入数据解包到数组
    always @(*) begin
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            if (state == IDLE && start) begin
                work_array[i] = data_in[i*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
    
    // 输出数据打包
    always @(*) begin
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            data_out[i*DATA_WIDTH +: DATA_WIDTH] = work_array[i];
        end
    end
    
    // 状态机
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            merge_size <= 4'd1;
            left_start <= 4'd0;
            copy_idx <= 4'd0;
        end
        else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        merge_size <= 4'd1;
                        left_start <= 4'd0;
                    end
                end
                
                MERGE: begin
                    // 执行归并操作
                    perform_merge();
                    
                    // 移动到下一个归并块
                    if (left_start + (merge_size << 1) < ARRAY_SIZE) begin
                        left_start <= left_start + (merge_size << 1);
                    end
                    else begin
                        copy_idx <= 4'd0;
                    end
                end
                
                COPY: begin
                    // 复制temp_array回work_array
                    if (copy_idx < ARRAY_SIZE) begin
                        work_array[copy_idx] <= temp_array[copy_idx];
                        copy_idx <= copy_idx + 1;
                    end
                    else begin
                        merge_size <= merge_size << 1;
                        left_start <= 4'd0;
                    end
                end
                
                DONE: begin
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // 次态逻辑
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start)
                    next_state = MERGE;
            end
            
            MERGE: begin
                if (left_start + (merge_size << 1) < ARRAY_SIZE)
                    next_state = MERGE;
                else
                    next_state = COPY;
            end
            
            COPY: begin
                if (copy_idx >= ARRAY_SIZE) begin
                    if (merge_size >= ARRAY_SIZE)
                        next_state = DONE;
                    else
                        next_state = MERGE;
                end
            end
            
            DONE: begin
                if (!start)
                    next_state = IDLE;
            end
        endcase
    end
    
    // 归并两个有序块
    task perform_merge;
        reg [3:0] mid, right_end;
        reg [3:0] i_idx, j_idx, k_idx;
        reg left_valid, right_valid;
    begin
        mid = left_start + merge_size - 1;
        right_end = (left_start + (merge_size << 1) - 1 < ARRAY_SIZE) ? 
                    (left_start + (merge_size << 1) - 1) : (ARRAY_SIZE - 1);
        
        i_idx = left_start;
        j_idx = mid + 1;
        k_idx = left_start;
        
        while (k_idx <= right_end) begin
            left_valid = (i_idx <= mid);
            right_valid = (j_idx <= right_end);
            
            if (left_valid && right_valid) begin
                if (work_array[i_idx] <= work_array[j_idx]) begin
                    temp_array[k_idx] = work_array[i_idx];
                    i_idx = i_idx + 1;
                end
                else begin
                    temp_array[k_idx] = work_array[j_idx];
                    j_idx = j_idx + 1;
                end
            end
            else if (left_valid) begin
                temp_array[k_idx] = work_array[i_idx];
                i_idx = i_idx + 1;
            end
            else begin
                temp_array[k_idx] = work_array[j_idx];
                j_idx = j_idx + 1;
            end
            k_idx = k_idx + 1;
        end
    end
    endtask

endmodule
