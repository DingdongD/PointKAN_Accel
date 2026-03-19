module float2fxp_pipe #(
    parameter WOI   = 8,
    parameter WOF   = 8,
    parameter ROUND = 1
)(
    input  wire               rstn,
    input  wire               clk,
    input  wire [31:0]        in,
    input  wire               ena,
    output reg  [WOI+WOF-1:0] out,
    output reg                overflow,
    output reg                out_valid
);

    localparam TOTAL_W = WOI + WOF;
    localparam [TOTAL_W-1:0] ONEO = 1;

    localparam SHIFT_W = $clog2(TOTAL_W + 1);

    localparam SEXP_W = SHIFT_W + 1;
    localparam signed [SEXP_W-1:0] POS_LIMIT =  TOTAL_W;
    localparam signed [SEXP_W-1:0] NEG_LIMIT = -TOTAL_W;

    initial {out, overflow, out_valid} = 0;

    wire        sign;
    wire [ 7:0] exp;
    wire [23:0] val;

    assign {sign, exp, val[22:0]} = in;
    assign val[23] = |exp;

    reg                signinit;
    reg                roundinit;
    reg signed [SEXP_W-1:0] expinit;
    reg [TOTAL_W-1:0]  outinit;
    reg                valid_s1;

    wire [8:0] woi_bias = WOI[8:0] + 9'd126;
    wire       exp_is_special = (exp == 8'd255) || ({1'b0, exp} > woi_bias);
    wire signed [9:0] exp_diff = {2'b00, exp} - {1'b0, woi_bias};

    generate
        if (TOTAL_W - 1 >= 23) begin : gen_wide
            always @(posedge clk or negedge rstn)
                if (~rstn) begin
                    outinit   <= 0;
                    roundinit <= 1'b0;
                    valid_s1  <= 1'b0;
                end else if (ena) begin
                    outinit <= 0;
                    outinit[TOTAL_W-1 : TOTAL_W-1-23] <= val;
                    roundinit <= 1'b0;
                    valid_s1  <= 1'b1;
                end else
                    valid_s1 <= 1'b0;
        end else begin : gen_narrow
            always @(posedge clk or negedge rstn)
                if (~rstn) begin
                    outinit   <= 0;
                    roundinit <= 1'b0;
                    valid_s1  <= 1'b0;
                end else if (ena) begin
                    outinit   <= val[23 : 23-(TOTAL_W-1)];
                    roundinit <= (ROUND && val[23-(TOTAL_W-1)-1]);
                    valid_s1  <= 1'b1;
                end else
                    valid_s1 <= 1'b0;
        end
    endgenerate

    always @(posedge clk or negedge rstn)
        if (~rstn) begin
            signinit <= 1'b0;
            expinit  <= {SEXP_W{1'b0}};
        end else if (ena) begin
            signinit <= sign;
            if (exp_is_special)
                expinit <= {SEXP_W{1'b0}};
            else if (exp_diff > POS_LIMIT)
                expinit <= POS_LIMIT;
            else if (exp_diff < NEG_LIMIT)
                expinit <= NEG_LIMIT;
            else
                expinit <= exp_diff[SEXP_W-1:0];
        end

    reg               sign_s2;
    reg [TOTAL_W-1:0] shifted_s2;
    reg               round_s2;
    reg               valid_s2;

    wire [SHIFT_W-1:0] lsh = expinit[SHIFT_W-1:0];
    wire [SHIFT_W-1:0] rsh = (-expinit);

    reg round_pick;
    integer ri;
    always @(*) begin
        round_pick = 1'b0;
        for (ri = 1; ri < TOTAL_W; ri = ri + 1) begin
            if (rsh == ri[SHIFT_W-1:0])
                round_pick = outinit[ri-1];
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (~rstn) begin
            sign_s2    <= 1'b0;
            shifted_s2 <= 0;
            round_s2   <= 1'b0;
            valid_s2   <= 1'b0;
        end else begin
            sign_s2  <= signinit;
            valid_s2 <= valid_s1;

            if (valid_s1) begin
                if (expinit > 0) begin
                    shifted_s2 <= outinit << lsh;
                    round_s2   <= 1'b0;
                end else if (expinit < 0) begin
                    shifted_s2 <= outinit >> rsh;
                    round_s2   <= (ROUND != 0) ? round_pick : 1'b0;
                end else begin
                    shifted_s2 <= outinit;
                    round_s2   <= roundinit;
                end
            end
        end
    end

    reg               signl;
    reg [TOTAL_W-1:0] outl;
    reg               valid_s3;

    always @(posedge clk or negedge rstn)
        if (~rstn) begin
            outl     <= 0;
            signl    <= 1'b0;
            valid_s3 <= 1'b0;
        end else begin
            valid_s3 <= valid_s2;
            if (valid_s2) begin : blk_round
                reg [TOTAL_W-1:0] outt;
                outt = shifted_s2;
                if (ROUND & round_s2 & ~(&outt))
                    outt = outt + 1;
                if (sign_s2) begin
                    signl <= (outt != 0);
                    outt  = (~outt) + ONEO;
                end else
                    signl <= 1'b0;
                outl <= outt;
            end
        end

    always @(posedge clk or negedge rstn)
        if (~rstn) begin
            out       <= 0;
            overflow  <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= valid_s3;
            if (valid_s3) begin
                out      <= outl;
                overflow <= 1'b0;
                if (signl) begin
                    if (~outl[TOTAL_W-1]) begin
                        out[TOTAL_W-1]   <= 1'b1;
                        out[TOTAL_W-2:0] <= 0;
                        overflow         <= 1'b1;
                    end
                end else begin
                    if (outl[TOTAL_W-1]) begin
                        out[TOTAL_W-1]   <= 1'b0;
                        out[TOTAL_W-2:0] <= {TOTAL_W{1'b1}};
                        overflow         <= 1'b1;
                    end
                end
            end
        end

endmodule