`include "unified_im2col.v"
`timescale 1ns/1ps

module tb_im2col_2d;

    parameter DIMENSION  = 2;
    parameter DATA_WIDTH = 8;
    parameter INPUT_H    = 5;
    parameter INPUT_W    = 5;
    parameter INPUT_C    = 2;
    parameter KERNEL_H   = 3;
    parameter KERNEL_W   = 3;
    parameter STRIDE     = 1;
    parameter PADDING    = 1;

    parameter OUTPUT_H   = (INPUT_H + 2*PADDING - KERNEL_H)/STRIDE + 1;
    parameter OUTPUT_W   = (INPUT_W + 2*PADDING - KERNEL_W)/STRIDE + 1;
    parameter COL_HEIGHT = KERNEL_H * KERNEL_W * INPUT_C;
    parameter COL_WIDTH  = OUTPUT_H * OUTPUT_W;
    parameter ADDR_WIDTH = clog2_safe(INPUT_H * INPUT_W * INPUT_C);

    parameter CLK_PERIOD = 10;

    function integer clog2_safe;
        input integer value;
        integer tmp;
        begin
            if (value <= 1) begin
                clog2_safe = 1;
            end else begin
                tmp = value - 1;
                clog2_safe = 0;
                while (tmp > 0) begin
                    tmp = tmp >> 1;
                    clog2_safe = clog2_safe + 1;
                end
            end
        end
    endfunction

    reg clk;
    reg rst_n;
    reg start;
    wire done;
    wire busy;

    wire [ADDR_WIDTH-1:0] ifm_addr;
    wire                  ifm_rd_en;
    reg  [DATA_WIDTH-1:0] ifm_data;
    reg                   ifm_valid;

    wire [clog2_safe(COL_HEIGHT)-1:0] col_row_addr;
    wire [clog2_safe(COL_WIDTH)-1:0]  col_col_addr;
    wire [DATA_WIDTH-1:0]             col_data;
    wire                              col_wr_en;
    wire                              col_valid;

    reg [DATA_WIDTH-1:0] input_feature_map [0:INPUT_H*INPUT_W*INPUT_C-1];
    reg [DATA_WIDTH-1:0] output_col_matrix [0:COL_HEIGHT-1][0:COL_WIDTH-1];
    reg [DATA_WIDTH-1:0] golden_col_matrix [0:COL_HEIGHT-1][0:COL_WIDTH-1];

    integer cycle_count;

    im2col #(
        .DIMENSION  (DIMENSION),
        .DATA_WIDTH (DATA_WIDTH),
        .INPUT_H    (INPUT_H),
        .INPUT_W    (INPUT_W),
        .INPUT_C    (INPUT_C),
        .KERNEL_H   (KERNEL_H),
        .KERNEL_W   (KERNEL_W),
        .STRIDE     (STRIDE),
        .PADDING    (PADDING)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .done        (done),
        .busy        (busy),
        .ifm_addr    (ifm_addr),
        .ifm_rd_en   (ifm_rd_en),
        .ifm_data    (ifm_data),
        .ifm_valid   (ifm_valid),
        .col_row_addr(col_row_addr),
        .col_col_addr(col_col_addr),
        .col_data    (col_data),
        .col_wr_en   (col_wr_en),
        .col_valid   (col_valid)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    reg [ADDR_WIDTH-1:0] ifm_addr_reg;
    reg                  ifm_rd_en_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ifm_addr_reg  <= 0;
            ifm_rd_en_reg <= 0;
        end else begin
            ifm_addr_reg  <= ifm_addr;
            ifm_rd_en_reg <= ifm_rd_en;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ifm_data  <= 0;
            ifm_valid <= 1'b0;
        end else begin
            if (ifm_rd_en_reg) begin
                ifm_data  <= input_feature_map[ifm_addr_reg];
                ifm_valid <= 1'b1;
            end else begin
                ifm_data  <= 0;
                ifm_valid <= 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (col_wr_en && col_valid) begin
            output_col_matrix[col_row_addr][col_col_addr] <= col_data;
        end
    end

    task initialize_input_feature_map;
        integer i;
    begin
        for (i = 0; i < INPUT_H*INPUT_W*INPUT_C; i = i + 1) begin
            input_feature_map[i] = i[DATA_WIDTH-1:0];
        end
    end
    endtask

    task clear_output_matrix;
        integer r, c;
    begin
        for (r = 0; r < COL_HEIGHT; r = r + 1)
            for (c = 0; c < COL_WIDTH; c = c + 1)
                output_col_matrix[r][c] = 0;
    end
    endtask

    task generate_golden_model;
        integer oh, ow, kh, kw, c;
        integer ih, iw;
        integer row_idx, col_idx;
        integer addr;
        reg [DATA_WIDTH-1:0] value;
    begin
        for (row_idx = 0; row_idx < COL_HEIGHT; row_idx = row_idx + 1) begin
            for (col_idx = 0; col_idx < COL_WIDTH; col_idx = col_idx + 1) begin
                golden_col_matrix[row_idx][col_idx] = 0;
            end
        end

        for (oh = 0; oh < OUTPUT_H; oh = oh + 1) begin
            for (ow = 0; ow < OUTPUT_W; ow = ow + 1) begin
                col_idx = oh * OUTPUT_W + ow;

                for (kh = 0; kh < KERNEL_H; kh = kh + 1) begin
                    for (kw = 0; kw < KERNEL_W; kw = kw + 1) begin
                        for (c = 0; c < INPUT_C; c = c + 1) begin
                            row_idx = (kh * KERNEL_W + kw) * INPUT_C + c;

                            ih = oh * STRIDE + kh - PADDING;
                            iw = ow * STRIDE + kw - PADDING;

                            if (ih < 0 || ih >= INPUT_H || iw < 0 || iw >= INPUT_W) begin
                                value = 0;
                            end else begin
                                addr  = (ih * INPUT_W + iw) * INPUT_C + c;
                                value = input_feature_map[addr];
                            end

                            golden_col_matrix[row_idx][col_idx] = value;
                        end
                    end
                end
            end
        end
    end
    endtask

    task check_results;
        integer row, col;
        integer error_count;
    begin
        error_count = 0;
        for (row = 0; row < COL_HEIGHT; row = row + 1) begin
            for (col = 0; col < COL_WIDTH; col = col + 1) begin
                if (output_col_matrix[row][col] !== golden_col_matrix[row][col]) begin
                    if (error_count < 20) begin
                        $display("2D ERROR [%0d][%0d]: DUT=%0d GOLDEN=%0d",
                                 row, col,
                                 output_col_matrix[row][col],
                                 golden_col_matrix[row][col]);
                    end
                    error_count = error_count + 1;
                end
            end
        end

        if (error_count == 0)
            $display("*** 2D TEST PASSED ***");
        else
            $display("*** 2D TEST FAILED: %0d mismatches ***", error_count);
    end
    endtask

    task display_output_matrix;
        integer row, col;
    begin
        $display("2D Output col matrix:");
        for (row = 0; row < COL_HEIGHT; row = row + 1) begin
            $write("row %02d : ", row);
            for (col = 0; col < COL_WIDTH; col = col + 1)
                $write("%3d ", output_col_matrix[row][col]);
            $write("\n");
        end
    end
    endtask

    initial begin
        rst_n = 0;
        start = 0;

        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);

        initialize_input_feature_map();
        clear_output_matrix();
        generate_golden_model();

        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        cycle_count = 0;
        while (!done) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
            if (cycle_count > 20000) begin
                $display("2D ERROR: timeout");
                $finish;
            end
        end

        repeat (5) @(posedge clk);

        $display("2D cycles = %0d", cycle_count);
        display_output_matrix();
        check_results();

        #(CLK_PERIOD*10);
        $finish;
    end

    initial begin
        $dumpfile("tb_im2col_2d.vcd");
        $dumpvars(0, tb_im2col_2d);
    end

endmodule