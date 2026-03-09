// ============================================================================
// Unified Im2col Module: Support 1D / 2D Convolution
// DIMENSION = 1 : 1D conv (slide along W dimension, require INPUT_H=1,KERNEL_H=1)
// DIMENSION = 2 : 2D conv
// Data layout in IFM memory: [H][W][C], linear addr = (h*W + w)*C + c
// ============================================================================

`timescale 1ns/1ps

module im2col #(
    parameter integer DIMENSION  = 2,   // 1 or 2
    parameter integer DATA_WIDTH = 8,
    parameter integer INPUT_H    = 32,
    parameter integer INPUT_W    = 32,
    parameter integer INPUT_C    = 3,
    parameter integer KERNEL_H   = 3,
    parameter integer KERNEL_W   = 3,
    parameter integer STRIDE     = 1,
    parameter integer PADDING    = 0,

    parameter integer OUTPUT_H   = (DIMENSION == 1) ?
                                   1 :
                                   ((INPUT_H + 2*PADDING - KERNEL_H) / STRIDE + 1),

    parameter integer OUTPUT_W   = ((INPUT_W + 2*PADDING - KERNEL_W) / STRIDE + 1),

    parameter integer COL_HEIGHT = KERNEL_H * KERNEL_W * INPUT_C,
    parameter integer COL_WIDTH  = OUTPUT_H * OUTPUT_W,
    parameter integer ADDR_WIDTH = CLOG2_SAFE(INPUT_H * INPUT_W * INPUT_C)
)(
    input  wire                           clk,
    input  wire                           rst_n,

    input  wire                           start,
    output reg                            done,
    output reg                            busy,

    // IFM SRAM-like read interface
    output reg  [ADDR_WIDTH-1:0]          ifm_addr,
    output reg                            ifm_rd_en,
    input  wire [DATA_WIDTH-1:0]          ifm_data,
    input  wire                           ifm_valid,

    // Column buffer write interface
    output reg  [CLOG2_SAFE(COL_HEIGHT)-1:0] col_row_addr,
    output reg  [CLOG2_SAFE(COL_WIDTH)-1:0]  col_col_addr,
    output reg  [DATA_WIDTH-1:0]             col_data,
    output reg                               col_wr_en,
    output reg                               col_valid
);

    // =========================================================================
    // Safe clog2
    // =========================================================================
    function integer CLOG2_SAFE;
        input integer value;
        integer tmp;
        begin
            if (value <= 1) begin
                CLOG2_SAFE = 1;
            end else begin
                tmp = value - 1;
                CLOG2_SAFE = 0;
                while (tmp > 0) begin
                    tmp = tmp >> 1;
                    CLOG2_SAFE = CLOG2_SAFE + 1;
                end
            end
        end
    endfunction

    // =========================================================================
    // State machine
    // =========================================================================
    localparam [2:0] IDLE       = 3'b000;
    localparam [2:0] CALC_ADDR  = 3'b001;
    localparam [2:0] READ_REQ   = 3'b010;
    localparam [2:0] WAIT_DATA  = 3'b011;
    localparam [2:0] WRITE_DATA = 3'b100;
    localparam [2:0] FINISH     = 3'b101;

    reg [2:0] state, next_state;

    // =========================================================================
    // Counters
    // =========================================================================
    reg [CLOG2_SAFE(OUTPUT_H)-1:0] out_h_cnt;
    reg [CLOG2_SAFE(OUTPUT_W)-1:0] out_w_cnt;
    reg [CLOG2_SAFE(KERNEL_H)-1:0] kernel_h_cnt;
    reg [CLOG2_SAFE(KERNEL_W)-1:0] kernel_w_cnt;
    reg [CLOG2_SAFE(INPUT_C)-1:0]  channel_cnt;

    // =========================================================================
    // Internal calculation registers
    // =========================================================================
    reg signed [31:0] input_h_calc;
    reg signed [31:0] input_w_calc;
    reg               is_padding;

    reg [CLOG2_SAFE(COL_HEIGHT)-1:0] col_row_calc;
    reg [CLOG2_SAFE(COL_WIDTH)-1:0]  col_col_calc;

    reg [DATA_WIDTH-1:0] data_pipeline;

    // last-element detection
    wire last_element;
    assign last_element =
        (out_h_cnt    == OUTPUT_H - 1) &&
        (out_w_cnt    == OUTPUT_W - 1) &&
        (kernel_h_cnt == KERNEL_H - 1) &&
        (kernel_w_cnt == KERNEL_W - 1) &&
        (channel_cnt  == INPUT_C - 1);

    // =========================================================================
    // Optional parameter checks (simulation only)
    // =========================================================================
    initial begin
        if (DIMENSION != 1 && DIMENSION != 2) begin
            $display("ERROR: DIMENSION must be 1 or 2.");
            $finish;
        end
        if (DIMENSION == 1) begin
            if (INPUT_H != 1) begin
                $display("ERROR: In DIMENSION=1 mode, INPUT_H must be 1.");
                $finish;
            end
            if (KERNEL_H != 1) begin
                $display("ERROR: In DIMENSION=1 mode, KERNEL_H must be 1.");
                $finish;
            end
        end
    end

    // =========================================================================
    // FSM sequential
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // =========================================================================
    // FSM combinational
    // =========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start) next_state = CALC_ADDR;
            end

            CALC_ADDR: begin
                next_state = READ_REQ;
            end

            READ_REQ: begin
                if (is_padding)
                    next_state = WRITE_DATA;  // padding directly writes zero
                else
                    next_state = WAIT_DATA;
            end

            WAIT_DATA: begin
                if (ifm_valid)
                    next_state = WRITE_DATA;
            end

            WRITE_DATA: begin
                if (last_element)
                    next_state = FINISH;
                else
                    next_state = CALC_ADDR;
            end

            FINISH: begin
                next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end

    // =========================================================================
    // Counter update (nested loop progression)
    // loop order:
    //   for oh
    //     for ow
    //       for kh
    //         for kw
    //           for c
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_h_cnt    <= 0;
            out_w_cnt    <= 0;
            kernel_h_cnt <= 0;
            kernel_w_cnt <= 0;
            channel_cnt  <= 0;
        end else begin
            if (state == IDLE && start) begin
                out_h_cnt    <= 0;
                out_w_cnt    <= 0;
                kernel_h_cnt <= 0;
                kernel_w_cnt <= 0;
                channel_cnt  <= 0;
            end else if (state == WRITE_DATA) begin
                if (channel_cnt == INPUT_C - 1) begin
                    channel_cnt <= 0;
                    if (kernel_w_cnt == KERNEL_W - 1) begin
                        kernel_w_cnt <= 0;
                        if (kernel_h_cnt == KERNEL_H - 1) begin
                            kernel_h_cnt <= 0;
                            if (out_w_cnt == OUTPUT_W - 1) begin
                                out_w_cnt <= 0;
                                if (out_h_cnt == OUTPUT_H - 1)
                                    out_h_cnt <= 0;
                                else
                                    out_h_cnt <= out_h_cnt + 1'b1;
                            end else begin
                                out_w_cnt <= out_w_cnt + 1'b1;
                            end
                        end else begin
                            kernel_h_cnt <= kernel_h_cnt + 1'b1;
                        end
                    end else begin
                        kernel_w_cnt <= kernel_w_cnt + 1'b1;
                    end
                end else begin
                    channel_cnt <= channel_cnt + 1'b1;
                end
            end
        end
    end
    
    integer oh_base;
    integer ow_base;
    integer ih_tmp;
    integer iw_tmp;
    integer addr_tmp;
    // =========================================================================
    // Address / position calculation
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        
        begin
            if (!rst_n) begin
                input_h_calc <= 0;
                input_w_calc <= 0;
                is_padding   <= 1'b0;
                ifm_addr     <= {ADDR_WIDTH{1'b0}};
                col_row_calc <= {CLOG2_SAFE(COL_HEIGHT){1'b0}};
                col_col_calc <= {CLOG2_SAFE(COL_WIDTH){1'b0}};
            end else if (state == CALC_ADDR) begin
                oh_base = (DIMENSION == 1) ? 0 : (out_h_cnt * STRIDE);
                ow_base = out_w_cnt * STRIDE;

                ih_tmp = oh_base + kernel_h_cnt - PADDING;
                iw_tmp = ow_base + kernel_w_cnt - PADDING;

                input_h_calc <= ih_tmp;
                input_w_calc <= iw_tmp;

                if (DIMENSION == 1) begin
                    // Only W dimension is meaningful, H is fixed to 0
                    if ((iw_tmp < 0) || (iw_tmp >= INPUT_W)) begin
                        is_padding <= 1'b1;
                    end else begin
                        is_padding <= 1'b0;
                        addr_tmp = ((0 * INPUT_W) + iw_tmp) * INPUT_C + channel_cnt;
                        ifm_addr <= addr_tmp[ADDR_WIDTH-1:0];
                    end
                end else begin
                    if ((ih_tmp < 0) || (ih_tmp >= INPUT_H) ||
                        (iw_tmp < 0) || (iw_tmp >= INPUT_W)) begin
                        is_padding <= 1'b1;
                    end else begin
                        is_padding <= 1'b0;
                        addr_tmp = ((ih_tmp * INPUT_W) + iw_tmp) * INPUT_C + channel_cnt;
                        ifm_addr <= addr_tmp[ADDR_WIDTH-1:0];
                    end
                end

                col_row_calc <= ((kernel_h_cnt * KERNEL_W + kernel_w_cnt) * INPUT_C + channel_cnt);
                col_col_calc <= (out_h_cnt * OUTPUT_W + out_w_cnt);
            end
        end
    end

    // =========================================================================
    // IFM read request
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ifm_rd_en <= 1'b0;
        end else begin
            if (state == READ_REQ && !is_padding)
                ifm_rd_en <= 1'b1;
            else
                ifm_rd_en <= 1'b0;
        end
    end

    // =========================================================================
    // Pipeline capture
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_pipeline <= {DATA_WIDTH{1'b0}};
        end else begin
            if (state == WAIT_DATA && ifm_valid) begin
                data_pipeline <= ifm_data;
            end else if (state == READ_REQ && is_padding) begin
                data_pipeline <= {DATA_WIDTH{1'b0}};
            end
        end
    end

    // =========================================================================
    // Output write
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_data     <= {DATA_WIDTH{1'b0}};
            col_row_addr <= {CLOG2_SAFE(COL_HEIGHT){1'b0}};
            col_col_addr <= {CLOG2_SAFE(COL_WIDTH){1'b0}};
            col_wr_en    <= 1'b0;
            col_valid    <= 1'b0;
        end else begin
            if (state == WRITE_DATA) begin
                col_data     <= data_pipeline;
                col_row_addr <= col_row_calc;
                col_col_addr <= col_col_calc;
                col_wr_en    <= 1'b1;
                col_valid    <= 1'b1;
            end else begin
                col_wr_en    <= 1'b0;
                col_valid    <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Busy / done
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    done <= 1'b0;
                end

                FINISH: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end

                default: begin
                    busy <= 1'b1;
                    done <= 1'b0;
                end
            endcase
        end
    end

endmodule