module fxp2float_pipeline #(
    parameter N    = 16,
    parameter FRAC = 8
)(
    input  wire         clk,
    input  wire         rstn,
    input  wire [N-1:0] fxp_in,
    input  wire         valid_in,
    output reg  [31:0]  fp_out,
    output reg          valid_out
);

    reg [N-1:0] abs_val_s1;
    reg         sign_s1;
    reg         valid_s1;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            abs_val_s1 <= 0;
            sign_s1    <= 0;
            valid_s1   <= 0;
        end else begin
            sign_s1    <= fxp_in[N-1];
            abs_val_s1 <= fxp_in[N-1] ? -fxp_in : fxp_in;
            valid_s1   <= valid_in;
        end
    end

    reg [N-1:0] abs_val_s2;
    reg         sign_s2;
    reg         valid_s2;
    reg [4:0]   msb_pos_s2;

    wire [4:0] msb_pos_comb;
    fxp2float_find_msb #(.N(N)) u_find_msb (
        .val(abs_val_s1),
        .msb_pos(msb_pos_comb)
    );

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            abs_val_s2 <= 0;
            sign_s2    <= 0;
            valid_s2   <= 0;
            msb_pos_s2 <= 0;
        end else begin
            abs_val_s2 <= abs_val_s1;
            sign_s2    <= sign_s1;
            valid_s2   <= valid_s1;
            msb_pos_s2 <= msb_pos_comb;
        end
    end

    reg [7:0]  exponent_s3;
    reg [22:0] mantissa_s3;
    reg        sign_s3;
    reg        valid_s3;

    wire signed [31:0] shift_val = 32'sd23 - {27'd0, msb_pos_s2};
    wire [31:0] mant_shifted = (shift_val >= 0)
        ? ({16'd0, abs_val_s2} << shift_val[4:0])
        : ({16'd0, abs_val_s2} >> (-shift_val[4:0]));

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            exponent_s3 <= 0;
            mantissa_s3 <= 0;
            sign_s3     <= 0;
            valid_s3    <= 0;
        end else begin
            sign_s3  <= sign_s2;
            valid_s3 <= valid_s2;

            if (abs_val_s2 == 0) begin
                exponent_s3 <= 0;
                mantissa_s3 <= 0;
            end else begin
                exponent_s3 <= 8'd127 + {3'd0, msb_pos_s2} - FRAC[7:0];
                mantissa_s3 <= mant_shifted[22:0];
            end
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            fp_out    <= 0;
            valid_out <= 0;
        end else begin
            fp_out    <= {sign_s3, exponent_s3, mantissa_s3};
            valid_out <= valid_s3;
        end
    end

endmodule

module fxp2float_find_msb #(
    parameter N = 16
)(
    input  wire [N-1:0] val,
    output reg  [4:0]   msb_pos
);
    integer i;
    always @(*) begin
        msb_pos = 5'd0;
        for (i = 0; i < N; i = i + 1) begin
            if (val[i])
                msb_pos = i[4:0];
        end
    end
endmodule