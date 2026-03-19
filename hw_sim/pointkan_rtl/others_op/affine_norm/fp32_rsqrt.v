

module fp32_rsqrt (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] x_in,
    input  wire        valid_in,
    output reg  [31:0] y_out,
    output reg         valid_out
);

    localparam [31:0] FP_THREE_HALF = 32'h3FC00000;

    reg  [31:0] mul_a, mul_b;
    reg         mul_v;
    wire [31:0] mul_r;
    wire        mul_d;
    fp32_mul u_mul (
        .clk(clk), .rst_n(rst_n),
        .a(mul_a), .b(mul_b), .valid_in(mul_v),
        .result(mul_r), .valid_out(mul_d)
    );

    reg  [31:0] add_a, add_b;
    reg         add_v;
    wire [31:0] add_r;
    wire        add_d;
    fp32_add u_add (
        .clk(clk), .rst_n(rst_n),
        .a(add_a), .b(add_b), .valid_in(add_v),
        .result(add_r), .valid_out(add_d)
    );

    localparam [3:0]
        RS_IDLE     = 4'd0,
        RS_SQ1      = 4'd1,
        RS_HXY1     = 4'd2,
        RS_T1       = 4'd3,
        RS_YT1      = 4'd4,
        RS_SQ2      = 4'd5,
        RS_HXY2     = 4'd6,
        RS_T2       = 4'd7,
        RS_YT2      = 4'd8,
        RS_OUT      = 4'd9;

    reg [3:0]  state;
    reg [31:0] y_reg;
    reg [31:0] halfx;
    reg [31:0] mean_r2;

    wire [31:0] y0_est = 32'h5F3759DF - {1'b0, x_in[31:1]};

    wire [31:0] halfx_comb = {x_in[31], x_in[30:23] - 8'd1, x_in[22:0]};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= RS_IDLE;
            valid_out <= 1'b0;
            y_out     <= 32'd0;
            mul_v     <= 1'b0;
            add_v     <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            mul_v     <= 1'b0;
            add_v     <= 1'b0;

            case (state)
                RS_IDLE: begin
                    if (valid_in) begin
                        y_reg <= y0_est;
                        halfx <= halfx_comb;

                        mul_a <= y0_est; mul_b <= y0_est; mul_v <= 1'b1;
                        state <= RS_SQ1;
                    end
                end

                RS_SQ1: begin
                    if (mul_d) begin

                        mul_a <= halfx; mul_b <= mul_r; mul_v <= 1'b1;
                        state <= RS_HXY1;
                    end
                end
                RS_HXY1: begin
                    if (mul_d) begin

                        add_a <= FP_THREE_HALF;
                        add_b <= {~mul_r[31], mul_r[30:0]};
                        add_v <= 1'b1;
                        state <= RS_T1;
                    end
                end
                RS_T1: begin
                    if (add_d) begin

                        mul_a <= y_reg; mul_b <= add_r; mul_v <= 1'b1;
                        state <= RS_YT1;
                    end
                end
                RS_YT1: begin
                    if (mul_d) begin

                        y_reg <= mul_r;
                        mul_a <= mul_r; mul_b <= mul_r; mul_v <= 1'b1;
                        state <= RS_SQ2;
                    end
                end

                RS_SQ2: begin
                    if (mul_d) begin
                        mul_a <= halfx; mul_b <= mul_r; mul_v <= 1'b1;
                        state <= RS_HXY2;
                    end
                end
                RS_HXY2: begin
                    if (mul_d) begin
                        add_a <= FP_THREE_HALF;
                        add_b <= {~mul_r[31], mul_r[30:0]};
                        add_v <= 1'b1;
                        state <= RS_T2;
                    end
                end
                RS_T2: begin
                    if (add_d) begin
                        mul_a <= y_reg; mul_b <= add_r; mul_v <= 1'b1;
                        state <= RS_YT2;
                    end
                end
                RS_YT2: begin
                    if (mul_d) begin
                        y_out <= mul_r;
                        state <= RS_OUT;
                    end
                end

                RS_OUT: begin
                    valid_out <= 1'b1;
                    state     <= RS_IDLE;
                end

                default: state <= RS_IDLE;
            endcase
        end
    end

endmodule