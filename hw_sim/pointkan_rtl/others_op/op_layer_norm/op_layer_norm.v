

module op_layer_norm_noreg #(
    parameter DIM_W = 10,
    parameter PAR   = 32
)(
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire [PAR*32-1:0]    x_in,
    input  wire [DIM_W-1:0]     dim_count_in,
    input  wire                  valid_in,

    output reg                   norm_rd,

    input  wire [PAR*32-1:0]    weight_in,
    input  wire [PAR*32-1:0]    bias_in,

    output reg  [PAR*32-1:0]    y_out,
    output reg                   valid_out,
    output wire                  ready_out
);

    localparam [31:0] FP32_EPS   = 32'h358637BD;
    localparam [31:0] FP32_ZERO  = 32'h00000000;
    localparam [31:0] FP32_THREE = 32'h40400000;

    localparam PTR_W = DIM_W - $clog2(PAR);

    localparam S_IDLE   = 3'd0;
    localparam S_RECV   = 3'd1;
    localparam S_TAIL   = 3'd2;
    localparam S_RSQRT  = 3'd3;
    localparam S_NORM   = 3'd4;

    reg [2:0]       state;
    reg [3:0]       sub;
    reg [PTR_W-1:0] batch_ptr;
    reg [PTR_W-1:0] total_batches;
    reg [DIM_W-1:0] dim_count;

    assign ready_out = (state == S_IDLE)
                     || (state == S_RECV && sub == 4'd0);

    reg [PAR*32-1:0] x_batch;

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
            default: inv_count_val = FP32_ZERO;
        endcase
    end

    reg  [PAR*32-1:0] mul_a, mul_b;
    wire [PAR*32-1:0] mul_r;
    reg  [PAR*32-1:0] add_a, add_b;
    wire [PAR*32-1:0] add_r;

    genvar gi;
    generate
        for (gi = 0; gi < PAR; gi = gi + 1) begin : gen_alu
            fp32_mul u_mul (
                .a(mul_a[gi*32 +: 32]), .b(mul_b[gi*32 +: 32]),
                .result(mul_r[gi*32 +: 32])
            );
            fp32_add u_add (
                .a(add_a[gi*32 +: 32]), .b(add_b[gi*32 +: 32]),
                .result(add_r[gi*32 +: 32])
            );
        end
    endgenerate

    reg [31:0] rsqrt_input;
    reg [31:0] rsqrt_lut_out;
    always @(*) begin : rsqrt_lut
        reg [7:0] r_exp, re;
        reg [22:0] rm;
        r_exp = rsqrt_input[30:23];
        re = 8'd0; rm = 23'd0;
        if (r_exp == 8'd0) begin
            rsqrt_lut_out = 32'h7F800000;
        end else begin
            if (r_exp[0]) begin
                re = (9'd379 - {1'b0, r_exp}) >> 1;
                rm = 23'h400000 - {1'b0, rsqrt_input[22:1]};
            end else begin
                re = (9'd380 - {1'b0, r_exp}) >> 1;
                rm = 23'h3504F3 - {1'b0, rsqrt_input[22:1]};
            end
            rsqrt_lut_out = {1'b0, re, rm};
        end
    end

    reg [31:0] sum_acc, sq_acc;
    reg [31:0] mean_reg, var_reg, inv_std;
    wire [31:0] neg_mean = {~mean_reg[31], mean_reg[30:0]};

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            sub           <= 4'd0;
            batch_ptr     <= {PTR_W{1'b0}};
            total_batches <= {PTR_W{1'b0}};
            dim_count     <= {DIM_W{1'b0}};
            x_batch       <= {(PAR*32){1'b0}};
            sum_acc       <= FP32_ZERO;
            sq_acc        <= FP32_ZERO;
            mean_reg      <= FP32_ZERO;
            var_reg       <= FP32_ZERO;
            inv_std       <= FP32_ZERO;
            rsqrt_input   <= FP32_ZERO;
            y_out         <= {(PAR*32){1'b0}};
            valid_out     <= 1'b0;
            norm_rd       <= 1'b0;
            mul_a         <= {(PAR*32){1'b0}};
            mul_b         <= {(PAR*32){1'b0}};
            add_a         <= {(PAR*32){1'b0}};
            add_b         <= {(PAR*32){1'b0}};
        end else begin
            valid_out <= 1'b0;
            norm_rd   <= 1'b0;

            case (state)

            S_IDLE: begin
                if (valid_in) begin
                    dim_count     <= dim_count_in;
                    total_batches <= dim_count_in[DIM_W-1:$clog2(PAR)];
                    batch_ptr     <= {PTR_W{1'b0}};
                    sum_acc       <= FP32_ZERO;
                    sq_acc        <= FP32_ZERO;

                    x_batch <= x_in;
                    mul_a   <= x_in;
                    mul_b   <= x_in;

                    state <= S_RECV;
                    sub   <= 4'd1;
                end
            end

            S_RECV: begin
                case (sub)

                4'd0: begin
                    if (valid_in) begin

                        sum_acc <= add_r[0*32 +: 32];
                        sq_acc  <= add_r[1*32 +: 32];

                        x_batch <= x_in;
                        mul_a   <= x_in;
                        mul_b   <= x_in;

                        sub <= 4'd1;
                    end
                end

                4'd1: begin
                    for (i = 0; i < 16; i = i + 1) begin

                        add_a[i*32 +: 32]      <= x_batch[(2*i)*32 +: 32];
                        add_b[i*32 +: 32]      <= x_batch[(2*i+1)*32 +: 32];

                        add_a[(16+i)*32 +: 32] <= mul_r[(2*i)*32 +: 32];
                        add_b[(16+i)*32 +: 32] <= mul_r[(2*i+1)*32 +: 32];
                    end
                    sub <= 4'd2;
                end

                4'd2: begin
                    for (i = 0; i < 8; i = i + 1) begin
                        add_a[i*32 +: 32]     <= add_r[(2*i)*32 +: 32];
                        add_b[i*32 +: 32]     <= add_r[(2*i+1)*32 +: 32];
                        add_a[(8+i)*32 +: 32] <= add_r[(16+2*i)*32 +: 32];
                        add_b[(8+i)*32 +: 32] <= add_r[(17+2*i)*32 +: 32];
                    end
                    sub <= 4'd3;
                end

                4'd3: begin
                    for (i = 0; i < 4; i = i + 1) begin
                        add_a[i*32 +: 32]     <= add_r[(2*i)*32 +: 32];
                        add_b[i*32 +: 32]     <= add_r[(2*i+1)*32 +: 32];
                        add_a[(4+i)*32 +: 32] <= add_r[(8+2*i)*32 +: 32];
                        add_b[(4+i)*32 +: 32] <= add_r[(9+2*i)*32 +: 32];
                    end
                    sub <= 4'd4;
                end

                4'd4: begin
                    add_a[0*32 +: 32] <= add_r[0*32 +: 32];
                    add_b[0*32 +: 32] <= add_r[1*32 +: 32];
                    add_a[1*32 +: 32] <= add_r[2*32 +: 32];
                    add_b[1*32 +: 32] <= add_r[3*32 +: 32];
                    add_a[2*32 +: 32] <= add_r[4*32 +: 32];
                    add_b[2*32 +: 32] <= add_r[5*32 +: 32];
                    add_a[3*32 +: 32] <= add_r[6*32 +: 32];
                    add_b[3*32 +: 32] <= add_r[7*32 +: 32];
                    sub <= 4'd5;
                end

                4'd5: begin
                    add_a[0*32 +: 32] <= add_r[0*32 +: 32];
                    add_b[0*32 +: 32] <= add_r[1*32 +: 32];
                    add_a[1*32 +: 32] <= add_r[2*32 +: 32];
                    add_b[1*32 +: 32] <= add_r[3*32 +: 32];
                    sub <= 4'd6;
                end

                4'd6: begin
                    add_a[0*32 +: 32] <= sum_acc;
                    add_b[0*32 +: 32] <= add_r[0*32 +: 32];
                    add_a[1*32 +: 32] <= sq_acc;
                    add_b[1*32 +: 32] <= add_r[1*32 +: 32];

                    batch_ptr <= batch_ptr + 1'b1;
                    if (batch_ptr + 1'b1 >= total_batches) begin
                        state <= S_TAIL;
                        sub   <= 4'd0;
                    end else begin
                        sub <= 4'd0;
                    end
                end

                default: sub <= 4'd0;
                endcase
            end

            S_TAIL: begin
                case (sub)
                4'd0: begin
                    sum_acc <= add_r[0*32 +: 32];
                    sq_acc  <= add_r[1*32 +: 32];
                    mul_a[0*32 +: 32] <= add_r[0*32 +: 32];
                    mul_b[0*32 +: 32] <= inv_count_val;
                    sub <= 4'd1;
                end
                4'd1: begin
                    mean_reg <= mul_r[0*32 +: 32];
                    mul_a[0*32 +: 32] <= sq_acc;
                    mul_b[0*32 +: 32] <= inv_count_val;
                    sub <= 4'd2;
                end
                4'd2: begin
                    var_reg <= mul_r[0*32 +: 32];
                    mul_a[0*32 +: 32] <= mean_reg;
                    mul_b[0*32 +: 32] <= mean_reg;
                    sub <= 4'd3;
                end
                4'd3: begin
                    add_a[0*32 +: 32] <= var_reg;
                    add_b[0*32 +: 32] <= {1'b1, mul_r[30:0]};
                    sub <= 4'd4;
                end
                4'd4: begin
                    var_reg <= add_r[0*32 +: 32];
                    add_a[0*32 +: 32] <= add_r[0*32 +: 32];
                    add_b[0*32 +: 32] <= FP32_EPS;
                    sub <= 4'd5;
                end
                4'd5: begin
                    rsqrt_input <= add_r[0*32 +: 32];
                    state <= S_RSQRT;
                    sub   <= 4'd0;
                end
                default: sub <= 4'd0;
                endcase
            end

            S_RSQRT: begin
                case (sub)
                4'd0: begin
                    inv_std <= rsqrt_lut_out;
                    sub <= 4'd1;
                end
                4'd1: begin
                    mul_a[0*32 +: 32] <= inv_std;
                    mul_b[0*32 +: 32] <= inv_std;
                    sub <= 4'd2;
                end
                4'd2: begin
                    mul_a[0*32 +: 32] <= rsqrt_input;
                    mul_b[0*32 +: 32] <= mul_r[0*32 +: 32];
                    sub <= 4'd3;
                end
                4'd3: begin
                    add_a[0*32 +: 32] <= FP32_THREE;
                    add_b[0*32 +: 32] <= {1'b1, mul_r[30:0]};
                    sub <= 4'd4;
                end
                4'd4: begin
                    mul_a[0*32 +: 32] <= inv_std;
                    mul_b[0*32 +: 32] <= add_r[0*32 +: 32];
                    sub <= 4'd5;
                end
                4'd5: begin
                    if (mul_r[30:23] == 8'hFF)
                        inv_std <= mul_r[0*32 +: 32];
                    else if (mul_r[30:23] > 8'd1)
                        inv_std <= {mul_r[31], mul_r[30:23] - 8'd1, mul_r[22:0]};
                    else
                        inv_std <= FP32_ZERO;
                    sub <= 4'd6;
                end
                4'd6: begin
                    mul_a[0*32 +: 32] <= inv_std;
                    mul_b[0*32 +: 32] <= inv_std;
                    sub <= 4'd7;
                end
                4'd7: begin
                    mul_a[0*32 +: 32] <= rsqrt_input;
                    mul_b[0*32 +: 32] <= mul_r[0*32 +: 32];
                    sub <= 4'd8;
                end
                4'd8: begin
                    add_a[0*32 +: 32] <= FP32_THREE;
                    add_b[0*32 +: 32] <= {1'b1, mul_r[30:0]};
                    sub <= 4'd9;
                end
                4'd9: begin
                    mul_a[0*32 +: 32] <= inv_std;
                    mul_b[0*32 +: 32] <= add_r[0*32 +: 32];
                    sub <= 4'd10;
                end
                4'd10: begin
                    if (mul_r[30:23] == 8'hFF)
                        inv_std <= mul_r[0*32 +: 32];
                    else if (mul_r[30:23] > 8'd1)
                        inv_std <= {mul_r[31], mul_r[30:23] - 8'd1, mul_r[22:0]};
                    else
                        inv_std <= FP32_ZERO;
                    batch_ptr <= {PTR_W{1'b0}};
                    state     <= S_NORM;
                    sub       <= 4'd0;
                end
                default: sub <= 4'd0;
                endcase
            end

            S_NORM: begin
                case (sub)
                4'd0: begin

                    for (i = 0; i < PAR; i = i + 1) begin
                        add_a[i*32 +: 32] <= x_in[i*32 +: 32];
                        add_b[i*32 +: 32] <= neg_mean;
                    end
                    norm_rd <= 1'b1;
                    sub     <= 4'd1;
                end
                4'd1: begin
                    for (i = 0; i < PAR; i = i + 1) begin
                        mul_a[i*32 +: 32] <= add_r[i*32 +: 32];
                        mul_b[i*32 +: 32] <= inv_std;
                    end
                    sub <= 4'd2;
                end
                4'd2: begin
                    for (i = 0; i < PAR; i = i + 1) begin
                        mul_a[i*32 +: 32] <= mul_r[i*32 +: 32];
                        mul_b[i*32 +: 32] <= weight_in[i*32 +: 32];
                    end
                    sub <= 4'd3;
                end
                4'd3: begin
                    for (i = 0; i < PAR; i = i + 1) begin
                        add_a[i*32 +: 32] <= mul_r[i*32 +: 32];
                        add_b[i*32 +: 32] <= bias_in[i*32 +: 32];
                    end
                    sub <= 4'd4;
                end
                4'd4: begin
                    y_out     <= add_r;
                    valid_out <= 1'b1;
                    batch_ptr <= batch_ptr + 1'b1;
                    if (batch_ptr + 1'b1 >= total_batches) begin
                        state <= S_IDLE;
                        sub   <= 4'd0;
                    end else begin
                        sub <= 4'd0;
                    end
                end
                default: sub <= 4'd0;
                endcase
            end

            default: state <= S_IDLE;

            endcase
        end
    end

endmodule