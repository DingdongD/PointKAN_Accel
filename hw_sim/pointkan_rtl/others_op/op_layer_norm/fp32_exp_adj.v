

module fp32_exp_adj (
    input  wire [31:0] val,
    input  wire [31:0] n_fp,
    output reg  [31:0] out
);

    wire        val_sign = val[31];
    wire [7:0]  val_exp  = val[30:23];

    wire        n_sign = n_fp[31];
    wire [7:0]  n_exp  = n_fp[30:23];
    wire [22:0] n_mant = n_fp[22:0];

    wire [7:0]  n_shift = n_exp - 8'd127;
    wire [7:0]  raw_hi  = {1'b1, n_mant[22:16]};

    reg  [8:0]  n_abs;
    reg  [9:0]  new_exp;

    always @(*) begin

        n_abs   = 9'd0;
        new_exp = 10'd0;
        out     = val;

        if (val[30:0] == 31'd0) begin
            out = val;
        end

        else if (n_exp < 8'd127) begin
            out = val;
        end

        else begin
            if (n_shift >= 8'd8)
                n_abs = 9'd255;
            else
                n_abs = {1'b0, raw_hi} >> (4'd7 - n_shift[3:0]);

            if (n_sign) begin

                if ({1'b0, val_exp} < {1'b0, n_abs[7:0]}) begin
                    out = {val_sign, 31'd0};
                end else begin
                    new_exp = {2'b00, val_exp} - {1'b0, n_abs};
                    if (new_exp == 10'd0)
                        out = {val_sign, 31'd0};
                    else
                        out = {val_sign, new_exp[7:0], val[22:0]};
                end
            end else begin

                new_exp = {2'b00, val_exp} + {1'b0, n_abs};
                if (new_exp >= 10'd255)
                    out = {val_sign, 8'hFF, 23'd0};
                else
                    out = {val_sign, new_exp[7:0], val[22:0]};
            end
        end
    end

endmodule