module op_layer_norm #(
    parameter MAX_DIM      = 512,
    parameter RSQRT_LUT_AW = 8,
    parameter DIM_W        = 10
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] x_in,
    input  wire        valid_in,
    input  wire        last_in,     
    input  wire [31:0] weight_in,
    input  wire [31:0] bias_in,
    output reg  [31:0] y_out,
    output reg         valid_out,
    output wire        ready_out
);
    localparam [31:0] FP32_EPS   = 32'h358637BD; 
    localparam [31:0] FP32_ZERO  = 32'h00000000;
    localparam [31:0] FP32_THREE = 32'h40400000; 
    localparam SC_IDLE     = 3'd0;
    localparam SC_ACCUM    = 3'd1;  
    localparam SC_VARIANCE = 3'd2;  
    localparam SC_RSQRT    = 3'd3;  
    localparam SC_DONE     = 3'd4;  
    localparam SO_IDLE      = 1'd0;
    localparam SO_NORMALIZE = 1'd1; 
    reg [2:0] sc_state;
    reg       so_state;
    assign ready_out = (sc_state == SC_IDLE) || (sc_state == SC_ACCUM);
    reg [31:0] x_buffer_A [0:MAX_DIM-1];
    reg [31:0] x_buffer_B [0:MAX_DIM-1];
    reg        wr_buf_sel;
    reg        rd_buf_sel;
    reg [DIM_W-1:0] rd_ptr_o;
    reg [DIM_W-1:0] dim_count;
    wire [31:0] buf_rd_data =
        (rd_buf_sel == 1'b0) ? x_buffer_A[rd_ptr_o]
                             : x_buffer_B[rd_ptr_o];
    reg [31:0] inv_count_val;
    always @(*) begin
        case (dim_count)
            10'd1:   inv_count_val = 32'h3F800000; 
            10'd2:   inv_count_val = 32'h3F000000; 
            10'd3:   inv_count_val = 32'h3EAAAAAB; 
            10'd4:   inv_count_val = 32'h3E800000; 
            10'd8:   inv_count_val = 32'h3E000000; 
            10'd16:  inv_count_val = 32'h3D800000; 
            10'd24:  inv_count_val = 32'h3D2AAAAB; 
            10'd32:  inv_count_val = 32'h3D000000; 
            10'd64:  inv_count_val = 32'h3C800000; 
            10'd128: inv_count_val = 32'h3C000000; 
            10'd256: inv_count_val = 32'h3B800000; 
            10'd512: inv_count_val = 32'h3B000000; 
            default: inv_count_val = 32'h00000000; 
        endcase
    end
    reg  [31:0] mul_a,  mul_b;  reg  mul_valid;
    wire [31:0] mul_result;     wire mul_done;
    fp32_mul u_mul (
        .clk(clk), .rst_n(rst_n),
        .a(mul_a), .b(mul_b), .valid_in(mul_valid),
        .result(mul_result), .valid_out(mul_done)
    );
    reg  [31:0] add_a,  add_b;  reg  add_valid;
    wire [31:0] add_result;     wire add_done;
    fp32_add u_add (
        .clk(clk), .rst_n(rst_n),
        .a(add_a), .b(add_b), .valid_in(add_valid),
        .result(add_result), .valid_out(add_done)
    );
    reg  [31:0] mul2_a, mul2_b; reg  mul2_valid;
    wire [31:0] mul2_result;    wire mul2_done;
    fp32_mul u_mul2 (
        .clk(clk), .rst_n(rst_n),
        .a(mul2_a), .b(mul2_b), .valid_in(mul2_valid),
        .result(mul2_result), .valid_out(mul2_done)
    );
    reg  [31:0] add2_a, add2_b; reg  add2_valid;
    wire [31:0] add2_result;    wire add2_done;
    fp32_add u_add2 (
        .clk(clk), .rst_n(rst_n),
        .a(add2_a), .b(add2_b), .valid_in(add2_valid),
        .result(add2_result), .valid_out(add2_done)
    );
    reg [31:0] rsqrt_input_c; 
    reg [31:0] rsqrt_lut_out;
    always @(*) begin : rsqrt_lut_logic
        reg [7:0]  r_exp;
        reg [7:0]  result_exp;
        reg [22:0] result_mant;
        r_exp       = rsqrt_input_c[30:23];
        result_exp  = 8'd0;   
        result_mant = 23'd0;  
        if (r_exp == 8'd0) begin
            rsqrt_lut_out = 32'h7F800000; 
        end else begin
            if (r_exp[0] == 1'b1) begin
                result_exp  = (9'd379 - {1'b0, r_exp}) >> 1;
                result_mant = 23'h400000 - {1'b0, rsqrt_input_c[22:1]};
            end else begin
                result_exp  = (9'd380 - {1'b0, r_exp}) >> 1;
                result_mant = 23'h3504F3 - {1'b0, rsqrt_input_c[22:1]};
            end
            rsqrt_lut_out = {1'b0, result_exp, result_mant};
        end
    end
    reg        handshake_valid;
    reg [31:0] hs_mean;
    reg [31:0] hs_inv_std;
    reg [DIM_W-1:0] hs_dim_count;
    reg        hs_buf_id;           
    wire handshake_ready    = (so_state == SO_IDLE);
    wire both_bufs_occupied = (so_state == SO_NORMALIZE) && handshake_valid;
    reg [DIM_W-1:0] wr_ptr;       
    reg [DIM_W-1:0] rd_ptr_c;     
    reg [3:0]       sc_sub_phase; 
    reg [31:0] sum_acc;    
    reg [31:0] sum_sq_acc; 
    reg [31:0] var_reg_c;    
    reg [31:0] inv_std_c;    
    reg [DIM_W-1:0] dim_count_o;  
    reg [31:0]      mean_o;       
    reg [31:0]      inv_std_o;    
    reg [3:0]       so_sub_phase; 
    reg [31:0] weight_d1, bias_d1, bias_d2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sc_state        <= SC_IDLE;
            wr_ptr          <= {DIM_W{1'b0}};
            dim_count       <= {DIM_W{1'b0}};
            rd_ptr_c        <= {DIM_W{1'b0}};
            sc_sub_phase    <= 4'd0;
            sum_acc    <= FP32_ZERO;
            sum_sq_acc      <= FP32_ZERO;
            var_reg_c       <= FP32_ZERO;
            inv_std_c       <= FP32_ZERO;
            rsqrt_input_c   <= FP32_ZERO;
            wr_buf_sel      <= 1'b0;
            handshake_valid <= 1'b0;
            hs_mean         <= FP32_ZERO;
            hs_inv_std      <= FP32_ZERO;
            hs_dim_count    <= {DIM_W{1'b0}};
            hs_buf_id       <= 1'b0;
            mul_a           <= 32'd0;
            mul_b           <= 32'd0;
            mul_valid       <= 1'b0;
            add_a           <= 32'd0;
            add_b           <= 32'd0;
            add_valid       <= 1'b0;
        end else begin
            mul_valid <= 1'b0;
            add_valid <= 1'b0;
            if (handshake_valid && handshake_ready)
                handshake_valid <= 1'b0;
            case (sc_state)
                SC_IDLE: begin
                    if (valid_in) begin
                        if (wr_buf_sel == 1'b0) x_buffer_A[0] <= x_in;
                        else                     x_buffer_B[0] <= x_in;
                        wr_ptr       <= {{(DIM_W-1){1'b0}}, 1'b1};
                        dim_count    <= {{(DIM_W-1){1'b0}}, 1'b1};
                        sum_acc <= FP32_ZERO;
                        sum_sq_acc   <= FP32_ZERO;
                        if (last_in) begin
                            sc_state     <= SC_VARIANCE;
                            sc_sub_phase <= 4'd0;
                        end else begin
                            sc_state <= SC_ACCUM;
                        end
                    end
                end
                SC_ACCUM: begin
                    if (valid_in) begin
                        if (wr_buf_sel == 1'b0) x_buffer_A[wr_ptr] <= x_in;
                        else                     x_buffer_B[wr_ptr] <= x_in;
                        wr_ptr    <= wr_ptr + 1'b1;
                        dim_count <= dim_count + 1'b1;
                        if (last_in) begin
                            sc_state     <= SC_VARIANCE;
                            sc_sub_phase <= 4'd0;
                        end
                    end
                end
                SC_VARIANCE: begin
                    case (sc_sub_phase)
                        4'd0: begin
                            sum_acc <= FP32_ZERO;
                            sum_sq_acc   <= FP32_ZERO;
                            rd_ptr_c     <= {DIM_W{1'b0}};
                            sc_sub_phase <= 4'd1;
                        end
                        4'd1: begin
                            add_a <= sum_acc;
                            add_b <= (wr_buf_sel == 1'b0)
                                     ? x_buffer_A[rd_ptr_c]
                                     : x_buffer_B[rd_ptr_c];
                            add_valid <= 1'b1;
                            mul_a <= (wr_buf_sel == 1'b0)
                                     ? x_buffer_A[rd_ptr_c]
                                     : x_buffer_B[rd_ptr_c];
                            mul_b <= (wr_buf_sel == 1'b0)
                                     ? x_buffer_A[rd_ptr_c]
                                     : x_buffer_B[rd_ptr_c];
                            mul_valid <= 1'b1;
                            sc_sub_phase <= 4'd2;
                        end
                        4'd2: begin
                            if (add_done && mul_done) begin
                                sum_acc <= add_result; 
                                add_a        <= sum_sq_acc;
                                add_b        <= mul_result;
                                add_valid    <= 1'b1;
                                sc_sub_phase <= 4'd3;
                            end
                        end
                        4'd3: begin
                            if (add_done) begin
                                sum_sq_acc <= add_result; 
                                rd_ptr_c   <= rd_ptr_c + 1'b1;
                                if (rd_ptr_c + 1'b1 >= dim_count)
                                    sc_sub_phase <= 4'd4; 
                                else
                                    sc_sub_phase <= 4'd1; 
                            end
                        end
                        4'd4: begin
                            mul_a        <= sum_acc;
                            mul_b        <= inv_count_val;
                            mul_valid    <= 1'b1;
                            sc_sub_phase <= 4'd5;
                        end
                        4'd5: begin
                            if (mul_done) begin
                                sum_acc <= mul_result; 
                                mul_a        <= sum_sq_acc;
                                mul_b        <= inv_count_val;
                                mul_valid    <= 1'b1;
                                sc_sub_phase <= 4'd6;
                            end
                        end
                        4'd6: begin
                            if (mul_done) begin
                                var_reg_c    <= mul_result; 
                                mul_a        <= sum_acc;
                                mul_b        <= sum_acc;
                                mul_valid    <= 1'b1;
                                sc_sub_phase <= 4'd7;
                            end
                        end
                        4'd7: begin
                            if (mul_done) begin
                                add_a        <= var_reg_c;
                                add_b        <= {1'b1, mul_result[30:0]};
                                add_valid    <= 1'b1;
                                sc_sub_phase <= 4'd8;
                            end
                        end
                        4'd8: begin
                            if (add_done) begin
                                var_reg_c    <= add_result;
                                add_a        <= add_result;
                                add_b        <= FP32_EPS;
                                add_valid    <= 1'b1;
                                sc_sub_phase <= 4'd9;
                            end
                        end
                        4'd9: begin
                            if (add_done) begin
                                rsqrt_input_c <= add_result; 
                                sc_state      <= SC_RSQRT;
                                sc_sub_phase  <= 4'd0;
                            end
                        end
                        default: sc_sub_phase <= 4'd0;
                    endcase
                end
                SC_RSQRT: begin
                    case (sc_sub_phase)
                        4'd0: begin
                            inv_std_c    <= rsqrt_lut_out; 
                            sc_sub_phase <= 4'd1;
                        end
                        4'd1: begin
                            mul_a        <= inv_std_c;
                            mul_b        <= inv_std_c;
                            mul_valid    <= 1'b1;
                            sc_sub_phase <= 4'd2;
                        end
                        4'd2: begin
                            if (mul_done) begin
                                mul_a        <= rsqrt_input_c;
                                mul_b        <= mul_result;  
                                mul_valid    <= 1'b1;
                                sc_sub_phase <= 4'd3;
                            end
                        end
                        4'd3: begin
                            if (mul_done) begin
                                add_a        <= FP32_THREE;
                                add_b        <= {1'b1, mul_result[30:0]};
                                add_valid    <= 1'b1;
                                sc_sub_phase <= 4'd4;
                            end
                        end
                        4'd4: begin
                            if (add_done) begin
                                mul_a        <= inv_std_c;
                                mul_b        <= add_result;
                                mul_valid    <= 1'b1;
                                sc_sub_phase <= 4'd5;
                            end
                        end
                        4'd5: begin
                            if (mul_done) begin
                                if (mul_result[30:23] == 8'hFF)
                                    inv_std_c <= mul_result; 
                                else if (mul_result[30:23] > 8'd1)
                                    inv_std_c <= {mul_result[31],
                                                  mul_result[30:23] - 8'd1,
                                                  mul_result[22:0]};
                                else
                                    inv_std_c <= FP32_ZERO; 
                                sc_sub_phase <= 4'd6; 
                            end
                        end
                        4'd6: begin
                            mul_a        <= inv_std_c;
                            mul_b        <= inv_std_c;
                            mul_valid    <= 1'b1;
                            sc_sub_phase <= 4'd7;
                        end
                        4'd7: begin
                            if (mul_done) begin
                                mul_a        <= rsqrt_input_c;
                                mul_b        <= mul_result;  
                                mul_valid    <= 1'b1;
                                sc_sub_phase <= 4'd8;
                            end
                        end
                        4'd8: begin
                            if (mul_done) begin
                                add_a        <= FP32_THREE;
                                add_b        <= {1'b1, mul_result[30:0]};
                                add_valid    <= 1'b1;
                                sc_sub_phase <= 4'd9;
                            end
                        end
                        4'd9: begin
                            if (add_done) begin
                                mul_a        <= inv_std_c;
                                mul_b        <= add_result;
                                mul_valid    <= 1'b1;
                                sc_sub_phase <= 4'd10;
                            end
                        end
                        4'd10: begin
                            if (mul_done) begin
                                if (mul_result[30:23] == 8'hFF)
                                    inv_std_c <= mul_result; 
                                else if (mul_result[30:23] > 8'd1)
                                    inv_std_c <= {mul_result[31],
                                                  mul_result[30:23] - 8'd1,
                                                  mul_result[22:0]};
                                else
                                    inv_std_c <= FP32_ZERO; 
                                sc_state     <= SC_DONE;
                                sc_sub_phase <= 4'd0;
                            end
                        end
                        default: sc_sub_phase <= 4'd0;
                    endcase
                end
                SC_DONE: begin
                    if (!both_bufs_occupied) begin
                        hs_mean         <= sum_acc;
                        hs_inv_std      <= inv_std_c;
                        hs_dim_count    <= dim_count;
                        hs_buf_id       <= wr_buf_sel;
                        handshake_valid <= 1'b1; 
                        wr_buf_sel   <= ~wr_buf_sel; 
                        wr_ptr       <= {DIM_W{1'b0}};
                        dim_count    <= {DIM_W{1'b0}};
                        sc_sub_phase <= 4'd0;
                        sc_state     <= SC_IDLE;
                    end
                end
                default: sc_state <= SC_IDLE;
            endcase
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            so_state     <= SO_IDLE;
            rd_ptr_o     <= {DIM_W{1'b0}};
            dim_count_o  <= {DIM_W{1'b0}};
            mean_o       <= FP32_ZERO;
            inv_std_o    <= FP32_ZERO;
            rd_buf_sel   <= 1'b1;  
            so_sub_phase <= 4'd0;
            weight_d1    <= 32'd0;
            bias_d1      <= 32'd0;
            bias_d2      <= 32'd0;
            y_out        <= 32'd0;
            valid_out    <= 1'b0;
            mul2_a       <= 32'd0;
            mul2_b       <= 32'd0;
            mul2_valid   <= 1'b0;
            add2_a       <= 32'd0;
            add2_b       <= 32'd0;
            add2_valid   <= 1'b0;
        end else begin
            valid_out  <= 1'b0;
            mul2_valid <= 1'b0;
            add2_valid <= 1'b0;
            case (so_state)
                SO_IDLE: begin
                    if (handshake_valid) begin
                        mean_o       <= hs_mean;
                        inv_std_o    <= hs_inv_std;
                        dim_count_o  <= hs_dim_count;
                        rd_buf_sel   <= hs_buf_id;
                        rd_ptr_o     <= {DIM_W{1'b0}};
                        so_sub_phase <= 4'd0;
                        so_state     <= SO_NORMALIZE;
                    end
                end
                SO_NORMALIZE: begin
                    case (so_sub_phase)
                        4'd0: begin
                            if (rd_ptr_o < dim_count_o) begin
                                add2_a       <= buf_rd_data;          
                                add2_b       <= {~mean_o[31], mean_o[30:0]}; 
                                add2_valid   <= 1'b1;
                                weight_d1    <= weight_in;
                                bias_d1      <= bias_in;
                                so_sub_phase <= 4'd1;
                            end else begin
                                so_state <= SO_IDLE;
                            end
                        end
                        4'd1: begin
                            if (add2_done) begin
                                mul2_a       <= add2_result;
                                mul2_b       <= inv_std_o;
                                mul2_valid   <= 1'b1;
                                so_sub_phase <= 4'd2;
                            end
                        end
                        4'd2: begin
                            if (mul2_done) begin
                                mul2_a       <= mul2_result;
                                mul2_b       <= weight_d1;
                                mul2_valid   <= 1'b1;
                                bias_d2      <= bias_d1;
                                so_sub_phase <= 4'd3;
                            end
                        end
                        4'd3: begin
                            if (mul2_done) begin
                                add2_a       <= mul2_result;
                                add2_b       <= bias_d2;
                                add2_valid   <= 1'b1;
                                so_sub_phase <= 4'd4;
                            end
                        end
                        4'd4: begin
                            if (add2_done) begin
                                y_out        <= add2_result;
                                valid_out    <= 1'b1;
                                rd_ptr_o     <= rd_ptr_o + 1'b1;
                                so_sub_phase <= 4'd5;
                            end
                        end
                        4'd5: begin
                            so_sub_phase <= 4'd0;
                        end
                        default: so_sub_phase <= 4'd0;
                    endcase
                end
                default: so_state <= SO_IDLE;
            endcase
        end
    end
endmodule