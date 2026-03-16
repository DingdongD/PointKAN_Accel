module quantize_pipeline #(
    parameter Q_WIDTH  = 12,
    parameter CLIP_MAX = (1 << (Q_WIDTH - 1)) - 1
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              ena,
    input  wire [31:0]       fp_in,
    input  wire [31:0]       scale,
    input  wire [Q_WIDTH-1:0] zp,
    input  wire              use_asym,
    output reg  [Q_WIDTH-1:0] q_out,
    output reg               sat,
    output reg               out_valid
);

    localparam EXT_W = Q_WIDTH + 4;

    wire [31:0] mul_out;
    wire        mul_valid;

    fp32_mul u_mul (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (ena),
        .a        (fp_in),
        .b        (scale),
        .result   (mul_out),
        .valid_out(mul_valid)
    );

    wire [Q_WIDTH-1:0] fxp_raw;
    wire               fxp_ovf;
    wire               fxp_valid;

    float2fxp_pipe #(
        .WOI(Q_WIDTH),
        .WOF(0)
    ) u_f2fxp (
        .rstn     (rst_n),
        .clk      (clk),
        .ena      (mul_valid),
        .in       (mul_out),
        .out      (fxp_raw),
        .overflow (fxp_ovf),
        .out_valid(fxp_valid)
    );

    wire signed [EXT_W-1:0] fxp_ext = {{4{fxp_raw[Q_WIDTH-1]}}, fxp_raw};
    wire signed [EXT_W-1:0] zp_ext  = use_asym
        ? {{4{zp[Q_WIDTH-1]}}, zp}
        : {EXT_W{1'b0}};

    wire signed [EXT_W-1:0] add_sum;
    wire                     add_ovf;
    wire                     add_valid;

    fixed_add_pipeline #(
        .WIDTH(EXT_W),
        .FRAC_BITS(0)
    ) u_add (
        .clk      (clk),
        .rst_n    (rst_n),
        .ena      (fxp_valid),
        .a        (fxp_ext),
        .b        (zp_ext),
        .sum      (add_sum),
        .overflow (add_ovf),
        .out_valid(add_valid)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_out     <= {Q_WIDTH{1'b0}};
            sat       <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= add_valid;
            if (add_valid) begin
                if (add_sum > CLIP_MAX) begin
                    q_out <= CLIP_MAX[Q_WIDTH-1:0];
                    sat   <= 1'b1;
                end else if (add_sum < 0) begin
                    q_out <= {Q_WIDTH{1'b0}};
                    sat   <= 1'b1;
                end else begin
                    q_out <= add_sum[Q_WIDTH-1:0];
                    sat   <= 1'b0;
                end
            end
        end
    end

endmodule

module fixed_add_pipeline #(
    parameter WIDTH     = 16,
    parameter FRAC_BITS = 0
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    ena,
    input  wire signed [WIDTH-1:0] a,
    input  wire signed [WIDTH-1:0] b,
    output reg  signed [WIDTH-1:0] sum,
    output reg                     overflow,
    output reg                     out_valid
);

    wire signed [WIDTH:0] ext_sum = {a[WIDTH-1], a} + {b[WIDTH-1], b};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum       <= 0;
            overflow  <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= ena;
            if (ena) begin
                sum      <= ext_sum[WIDTH-1:0];
                overflow <= (ext_sum[WIDTH] != ext_sum[WIDTH-1]);
            end
        end
    end

endmodule