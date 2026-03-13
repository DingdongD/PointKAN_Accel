
// 测试模块
`include "mergesort.v"
module merge_sort_tb;
    parameter DATA_WIDTH = 32;
    parameter ARRAY_SIZE = 8;
    
    reg clk;
    reg rst_n;
    reg start;
    reg [DATA_WIDTH*ARRAY_SIZE-1:0] data_in;
    wire [DATA_WIDTH*ARRAY_SIZE-1:0] data_out;
    wire done;
    
    integer i;
    reg [DATA_WIDTH-1:0] input_array [0:ARRAY_SIZE-1];
    reg [DATA_WIDTH-1:0] output_array [0:ARRAY_SIZE-1];
    
    // 实例化归并排序模块
    merge_sort #(
        .DATA_WIDTH(DATA_WIDTH),
        .ARRAY_SIZE(ARRAY_SIZE)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .data_out(data_out),
        .done(done)
    );
    
    // 时钟生成
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // 打包输入数据
    always @(*) begin
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            data_in[i*DATA_WIDTH +: DATA_WIDTH] = input_array[i];
        end
    end
    
    // 解包输出数据
    always @(*) begin
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            output_array[i] = data_out[i*DATA_WIDTH +: DATA_WIDTH];
        end
    end
    
    // 测试激励
    initial begin
        $display("=== Star to Merge-sort ===");
        
        // 初始化
        rst_n = 0;
        start = 0;
        
        // 输入测试数据
        input_array[0] = 32'd64;
        input_array[1] = 32'd34;
        input_array[2] = 32'd25;
        input_array[3] = 32'd12;
        input_array[4] = 32'd22;
        input_array[5] = 32'd11;
        input_array[6] = 32'd90;
        input_array[7] = 32'd88;
        
        #20 rst_n = 1;
        
        $display("\nBefore Sorting:");
        for (i = 0; i < ARRAY_SIZE; i = i + 1)
            $write("%d ", input_array[i]);
        $display("");
        
        #10 start = 1;
        #10 start = 0;
        
        // 等待排序完成
        wait(done);
        #20;
        
        // 显示结果
        $display("\nAfter Sorting:");
        for (i = 0; i < ARRAY_SIZE; i = i + 1)
            $write("%d ", output_array[i]);
        $display("\n");
        
        // 验证结果
        $display("Validation: ");
        if (output_array[0] == 11 && output_array[1] == 12 && 
            output_array[2] == 22 && output_array[3] == 25 &&
            output_array[4] == 34 && output_array[5] == 64 &&
            output_array[6] == 88 && output_array[7] == 90)
            $display("Success");
        else
            $display("Failed");
        
        #50 $finish;
    end
    
    // 波形文件生成
    initial begin
        $dumpfile("merge_sort.vcd");
        $dumpvars(0, merge_sort_tb);
    end

endmodule