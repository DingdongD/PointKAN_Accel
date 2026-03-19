

module fp32_floor_comb (
    input  wire [31:0] p,
    output reg  [31:0] out
);

    wire        sign    = p[31];
    wire [7:0]  exp_val = p[30:23];
    wire [22:0] mant    = p[22:0];

    wire [23:0] full_mant = {1'b1, mant};

    wire [7:0] shift = exp_val - 8'd127;

    reg [23:0] imask;
    reg [23:0] ipart;
    reg        has_frac;

    reg [23:0] inc_val;
    reg [23:0] inc_result;

    always @(*) begin

        imask      = 24'hFFFFFF;
        ipart      = 24'd0;
        has_frac   = 1'b0;
        inc_val    = 24'd0;
        inc_result = 24'd0;
        out        = 32'd0;

        if (exp_val == 8'd0 && mant == 23'd0) begin
            out = 32'h00000000;
        end

        else if (exp_val == 8'd0 || exp_val < 8'd127) begin
            out = sign ? 32'hBF800000 : 32'h00000000;
        end

        else if (shift >= 8'd23) begin
            out = p;
        end

        else begin
            imask    = 24'hFFFFFF << (8'd23 - shift);
            ipart    = full_mant & imask;
            has_frac = (full_mant != ipart);

            if (sign && has_frac) begin

                inc_val    = 24'd1 << (8'd23 - shift);
                inc_result = ipart + inc_val;

                if (inc_result[23]) begin

                    out = {1'b1, exp_val, inc_result[22:0]};
                end else begin

                    out = {1'b1, exp_val + 8'd1, 23'd0};
                end
            end else begin

                out = {sign, exp_val, ipart[22:0]};
            end
        end
    end

endmodule