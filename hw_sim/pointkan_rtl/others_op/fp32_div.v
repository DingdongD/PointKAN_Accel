

module fp32_div (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire        valid_in,
    output reg  [31:0] result,
    output reg         valid_out
);

    localparam S_IDLE     = 3'd0;
    localparam S_SPECIAL  = 3'd1;
    localparam S_ITER_MUL = 3'd2;
    localparam S_ITER_UPD = 3'd3;
    localparam S_QUOT     = 3'd4;
    localparam S_NORM     = 3'd5;

    reg [2:0]  state;
    reg [1:0]  iter_count;

    reg [31:0] a_reg;
    reg [31:0] b_reg;
    reg        sign_reg;
    reg [9:0]  exp_reg;

    reg [24:0] a_norm;
    reg [24:0] b_norm;
    reg [24:0] r_est;
    reg [24:0] f_reg;

    wire a_is_zero = (a_reg[30:23] == 8'd0)  && (a_reg[22:0] == 23'd0);
    wire b_is_zero = (b_reg[30:23] == 8'd0)  && (b_reg[22:0] == 23'd0);
    wire a_is_inf  = (a_reg[30:23] == 8'hFF) && (a_reg[22:0] == 23'd0);
    wire b_is_inf  = (b_reg[30:23] == 8'hFF) && (b_reg[22:0] == 23'd0);
    wire a_is_nan  = (a_reg[30:23] == 8'hFF) && (a_reg[22:0] != 23'd0);
    wire b_is_nan  = (b_reg[30:23] == 8'hFF) && (b_reg[22:0] != 23'd0);

    wire [49:0] mul_br = {25'b0, b_norm} * {25'b0, r_est};
    wire [49:0] mul_ar = {25'b0, a_norm} * {25'b0, r_est};
    wire [49:0] mul_fr = {25'b0, f_reg}  * {25'b0, r_est};

    wire [25:0] two_fp = 26'h2000000;
    wire [25:0] br_trunc = {1'b0, mul_br[48:24]};
    wire [25:0] f_val = two_fp - br_trunc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            result     <= 32'd0;
            valid_out  <= 1'b0;
            iter_count <= 2'd0;
            a_reg      <= 32'd0;
            b_reg      <= 32'd0;
            sign_reg   <= 1'b0;
            exp_reg    <= 10'd0;
            a_norm     <= 25'd0;
            b_norm     <= 25'd0;
            r_est      <= 25'd0;
            f_reg      <= 25'd0;
        end else begin
            valid_out <= 1'b0;

            case (state)

                S_IDLE: begin
                    if (valid_in) begin
                        a_reg    <= a;
                        b_reg    <= b;
                        sign_reg <= a[31] ^ b[31];
                        state    <= S_SPECIAL;
                    end
                end

                S_SPECIAL: begin

                    if (a_is_nan || b_is_nan || (a_is_inf && b_is_inf)) begin

                        result    <= {sign_reg, 8'hFF, 23'h400000};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (a_is_zero && b_is_zero) begin
                        result    <= 32'd0;
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (a_is_zero) begin
                        result    <= {sign_reg, 31'd0};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (b_is_zero) begin
                        result    <= {sign_reg, 8'hFF, 23'd0};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (a_is_inf) begin
                        result    <= {sign_reg, 8'hFF, 23'd0};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else if (b_is_inf) begin
                        result    <= {sign_reg, 31'd0};
                        valid_out <= 1'b1;
                        state     <= S_IDLE;
                    end else begin

                        exp_reg <= {2'b0, a_reg[30:23]} - {2'b0, b_reg[30:23]} + 10'd127;

                        a_norm <= {1'b1, a_reg[22:0], 1'b0};
                        b_norm <= {1'b1, b_reg[22:0], 1'b0};

                        r_est  <= 25'h1000000 - {2'b00, b_reg[22:0]};

                        iter_count <= 2'd0;
                        state      <= S_ITER_MUL;
                    end
                end

                S_ITER_MUL: begin

                    f_reg <= f_val[24:0];
                    state <= S_ITER_UPD;
                end

                S_ITER_UPD: begin

                    r_est <= mul_fr[48:24];

                    if (iter_count == 2'd2) begin

                        state <= S_QUOT;
                    end else begin
                        iter_count <= iter_count + 2'd1;
                        state      <= S_ITER_MUL;
                    end
                end

                S_QUOT: begin

                    a_norm <= mul_ar[48:24];
                    state  <= S_NORM;
                end

                S_NORM: begin
                    begin : normalize_block
                        reg [9:0]  final_exp;
                        reg [22:0] final_mant;

                        if (a_norm[24]) begin

                            final_exp  = exp_reg;
                            final_mant = a_norm[23:1];
                        end else if (a_norm[23]) begin

                            final_exp  = exp_reg - 10'd1;
                            final_mant = a_norm[22:0];
                        end else begin

                            final_exp  = exp_reg - 10'd2;
                            final_mant = {a_norm[21:0], 1'b0};
                        end

                        if (final_exp[9] || final_exp == 10'd0) begin
                            result <= {sign_reg, 31'd0};
                        end else if (final_exp >= 10'd255) begin
                            result <= {sign_reg, 8'hFF, 23'd0};
                        end else begin
                            result <= {sign_reg, final_exp[7:0], final_mant};
                        end
                    end

                    valid_out <= 1'b1;
                    state     <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule