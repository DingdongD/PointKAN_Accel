

module op_batch_norm_1d (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] x_in,
    input  wire [31:0] gamma_in,
    input  wire [31:0] beta_in,
    input  wire        valid_in,
    output wire [31:0] y_out,
    output wire        valid_out
);

    wire [31:0] mul_result;
    wire        mul_valid;

    fp32_mul u_mul (
        .clk       (clk),
        .rst_n     (rst_n),
        .a         (x_in),
        .b         (gamma_in),
        .valid_in  (valid_in),
        .result    (mul_result),
        .valid_out (mul_valid)
    );

    reg [31:0] beta_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            beta_d1 <= 32'd0;
        else
            beta_d1 <= beta_in;
    end

    fp32_add u_add (
        .clk       (clk),
        .rst_n     (rst_n),
        .a         (mul_result),
        .b         (beta_d1),
        .valid_in  (mul_valid),
        .result    (y_out),
        .valid_out (valid_out)
    );

endmodule
