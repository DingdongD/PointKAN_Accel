
module dequantize_pipeline #(
    parameter N    = 16,
    parameter FRAC = 8
)(
    input  wire         clk,
    input  wire         rstn,
    input  wire [N-1:0] fxp_in,
    input  wire         valid_in,
    input  wire [31:0]  scale,
    output wire [31:0]  fp_out,
    output wire         valid_out
);

    wire [31:0] fp_val;
    wire        fxp2f_valid;

    fxp2float_pipeline #(
        .N(N),
        .FRAC(FRAC)
    ) u_fxp2f (
        .clk      (clk),
        .rstn     (rstn),
        .fxp_in   (fxp_in),
        .valid_in (valid_in),
        .fp_out   (fp_val),
        .valid_out(fxp2f_valid)
    );

    fp32_mul u_fpmul (
        .clk      (clk),
        .rst_n    (rstn),
        .valid_in (fxp2f_valid),
        .a        (fp_val),
        .b        (scale),
        .result   (fp_out),
        .valid_out(valid_out)
    );

endmodule