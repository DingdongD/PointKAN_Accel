

module op_spool_v4 #(
    parameter K = 24
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire [K*32-1:0]   x_in,
    input  wire              valid_in,
    output reg  [31:0]       y_out,
    output reg               valid_out
);

    localparam P = K;

    localparam [31:0] FP_ZERO    = 32'h00000000;
    localparam [31:0] FP_NEG_INF = 32'hFF800000;
    localparam [31:0] LOG2E      = 32'h3FB8AA3B;
    localparam [31:0] COEFF_C0   = 32'h3F800000;
    localparam [31:0] COEFF_C1   = 32'h3F317218;
    localparam [31:0] COEFF_C2   = 32'h3E75FDF0;

    localparam [4:0] S_IDLE        = 5'd0;
    localparam [4:0] S_SUB         = 5'd1;
    localparam [4:0] S_MUL_P       = 5'd2;
    localparam [4:0] S_SPLIT_LATCH = 5'd3;
    localparam [4:0] S_SPLIT_F     = 5'd4;
    localparam [4:0] S_MUL_C2F     = 5'd5;
    localparam [4:0] S_ADD_C1      = 5'd6;
    localparam [4:0] S_MUL_TF      = 5'd7;
    localparam [4:0] S_ADD_C0      = 5'd8;
    localparam [4:0] S_MUL_EX      = 5'd9;
    localparam [4:0] S_RED1        = 5'd10;
    localparam [4:0] S_RED2        = 5'd11;
    localparam [4:0] S_RED3        = 5'd12;
    localparam [4:0] S_RED4        = 5'd13;
    localparam [4:0] S_RED5        = 5'd14;
    localparam [4:0] S_FIRE_DIV    = 5'd15;
    localparam [4:0] S_WAIT_DIV    = 5'd16;

    reg [4:0] state;

    reg [K*32-1:0] xr;

    wire latch_input = valid_in
                     && (state == S_IDLE || state == S_WAIT_DIV);

    always @(posedge clk) begin
        if (latch_input)
            xr <= x_in;
    end

    wire [12*32-1:0] cmp_l1;
    wire [ 6*32-1:0] cmp_l2;
    wire [ 3*32-1:0] cmp_l3;
    wire [31:0]      cmp_l4;
    wire [31:0]      cmp_max;

    genvar gi;

    generate
        for (gi = 0; gi < 12; gi = gi + 1) begin : gen_cmp_l1
            fp32_fmax u_fmax (
                .a  (x_in[(2*gi  )*32 +: 32]),
                .b  (x_in[(2*gi+1)*32 +: 32]),
                .out(cmp_l1[gi*32 +: 32])
            );
        end
    endgenerate

    generate
        for (gi = 0; gi < 6; gi = gi + 1) begin : gen_cmp_l2
            fp32_fmax u_fmax (
                .a  (cmp_l1[(2*gi  )*32 +: 32]),
                .b  (cmp_l1[(2*gi+1)*32 +: 32]),
                .out(cmp_l2[gi*32 +: 32])
            );
        end
    endgenerate

    generate
        for (gi = 0; gi < 3; gi = gi + 1) begin : gen_cmp_l3
            fp32_fmax u_fmax (
                .a  (cmp_l2[(2*gi  )*32 +: 32]),
                .b  (cmp_l2[(2*gi+1)*32 +: 32]),
                .out(cmp_l3[gi*32 +: 32])
            );
        end
    endgenerate

    fp32_fmax u_cmp_l4 (
        .a(cmp_l3[0*32 +: 32]), .b(cmp_l3[1*32 +: 32]), .out(cmp_l4)
    );
    fp32_fmax u_cmp_l5 (
        .a(cmp_l4), .b(cmp_l3[2*32 +: 32]), .out(cmp_max)
    );

    reg  [P*32-1:0] add_a;
    reg  [P*32-1:0] add_b;
    wire [P*32-1:0] add_out;

    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : gen_add
            fp32_add u_add (
                .a     (add_a[gi*32 +: 32]),
                .b     (add_b[gi*32 +: 32]),
                .result(add_out[gi*32 +: 32])
            );
        end
    endgenerate

    reg  [P*32-1:0] mul_a;
    reg  [P*32-1:0] mul_b;
    wire [P*32-1:0] mul_out;

    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : gen_mul
            fp32_mul u_mul (
                .a     (mul_a[gi*32 +: 32]),
                .b     (mul_b[gi*32 +: 32]),
                .result(mul_out[gi*32 +: 32])
            );
        end
    endgenerate

    reg  [31:0] div_a, div_b;
    reg         div_fire;
    wire [31:0] div_out;
    wire        div_done;

    fp32_div u_div (
        .clk(clk), .rst_n(rst_n),
        .a(div_a), .b(div_b), .valid_in(div_fire),
        .result(div_out), .valid_out(div_done)
    );

    reg [P*32-1:0] n_reg;
    reg [P*32-1:0] f_reg;
    reg [P*32-1:0] ev_reg;
    reg [P*32-1:0] xo_reg;
    reg [P*32-1:0] p_latch;

    reg [31:0] num_pass;
    reg [31:0] den_pass;

    reg [31:0] x_max;
    wire [31:0] neg_xmax = {~x_max[31], x_max[30:0]};

    reg div_pending;

    wire [P*32-1:0] floor_val;
    wire [P*32-1:0] neg_floor;

    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : gen_floor
            fp32_floor_comb u_floor (
                .p  (p_latch[gi*32 +: 32]),
                .out(floor_val[gi*32 +: 32])
            );
            assign neg_floor[gi*32 +: 32] = {
                ~floor_val[gi*32 + 31],
                 floor_val[gi*32 +: 31]
            };
        end
    endgenerate

    wire [P*32-1:0] eadj_val;

    generate
        for (gi = 0; gi < P; gi = gi + 1) begin : gen_eadj
            fp32_exp_adj u_eadj (
                .val (add_out[gi*32 +: 32]),
                .n_fp(n_reg[gi*32 +: 32]),
                .out (eadj_val[gi*32 +: 32])
            );
        end
    endgenerate

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            x_max       <= FP_NEG_INF;
            y_out       <= FP_ZERO;
            valid_out   <= 1'b0;
            div_fire    <= 1'b0;
            div_pending <= 1'b0;
            div_a       <= FP_ZERO;
            div_b       <= FP_ZERO;
            add_a       <= {(P*32){1'b0}};
            add_b       <= {(P*32){1'b0}};
            mul_a       <= {(P*32){1'b0}};
            mul_b       <= {(P*32){1'b0}};
            n_reg       <= {(P*32){1'b0}};
            f_reg       <= {(P*32){1'b0}};
            ev_reg      <= {(P*32){1'b0}};
            xo_reg      <= {(P*32){1'b0}};
            p_latch     <= {(P*32){1'b0}};
            num_pass    <= FP_ZERO;
            den_pass    <= FP_ZERO;
        end else begin

            valid_out <= 1'b0;
            div_fire  <= 1'b0;

            case (state)

            S_IDLE: begin
                if (valid_in) begin
                    x_max <= cmp_max;
                    state <= S_SUB;
                end
            end

            S_SUB: begin
                for (i = 0; i < P; i = i + 1) begin
                    add_a[i*32 +: 32]  <= xr[i*32 +: 32];
                    add_b[i*32 +: 32]  <= neg_xmax;
                    xo_reg[i*32 +: 32] <= xr[i*32 +: 32];
                end
                state <= S_MUL_P;
            end

            S_MUL_P: begin
                for (i = 0; i < P; i = i + 1) begin
                    mul_a[i*32 +: 32] <= add_out[i*32 +: 32];
                    mul_b[i*32 +: 32] <= LOG2E;
                end
                state <= S_SPLIT_LATCH;
            end

            S_SPLIT_LATCH: begin
                p_latch <= mul_out;
                state   <= S_SPLIT_F;
            end

            S_SPLIT_F: begin
                for (i = 0; i < P; i = i + 1) begin
                    n_reg[i*32 +: 32] <= floor_val[i*32 +: 32];
                    add_a[i*32 +: 32] <= p_latch[i*32 +: 32];
                    add_b[i*32 +: 32] <= neg_floor[i*32 +: 32];
                end
                state <= S_MUL_C2F;
            end

            S_MUL_C2F: begin
                for (i = 0; i < P; i = i + 1) begin
                    f_reg[i*32 +: 32] <= add_out[i*32 +: 32];
                    mul_a[i*32 +: 32] <= COEFF_C2;
                    mul_b[i*32 +: 32] <= add_out[i*32 +: 32];
                end
                state <= S_ADD_C1;
            end

            S_ADD_C1: begin
                for (i = 0; i < P; i = i + 1) begin
                    add_a[i*32 +: 32] <= mul_out[i*32 +: 32];
                    add_b[i*32 +: 32] <= COEFF_C1;
                end
                state <= S_MUL_TF;
            end

            S_MUL_TF: begin
                for (i = 0; i < P; i = i + 1) begin
                    mul_a[i*32 +: 32] <= add_out[i*32 +: 32];
                    mul_b[i*32 +: 32] <= f_reg[i*32 +: 32];
                end
                state <= S_ADD_C0;
            end

            S_ADD_C0: begin
                for (i = 0; i < P; i = i + 1) begin
                    add_a[i*32 +: 32] <= mul_out[i*32 +: 32];
                    add_b[i*32 +: 32] <= COEFF_C0;
                end

                ev_reg <= eadj_val;
                state  <= S_MUL_EX;
            end

            S_MUL_EX: begin
                for (i = 0; i < P; i = i + 1) begin
                    mul_a[i*32 +: 32] <= ev_reg[i*32 +: 32];
                    mul_b[i*32 +: 32] <= xo_reg[i*32 +: 32];
                end
                state <= S_RED1;
            end

            S_RED1: begin
                for (i = 0; i < 12; i = i + 1) begin

                    add_a[i*32 +: 32]      <= mul_out[(2*i)*32 +: 32];
                    add_b[i*32 +: 32]      <= mul_out[(2*i+1)*32 +: 32];

                    add_a[(12+i)*32 +: 32] <= ev_reg[(2*i)*32 +: 32];
                    add_b[(12+i)*32 +: 32] <= ev_reg[(2*i+1)*32 +: 32];
                end
                state <= S_RED2;
            end

            S_RED2: begin
                for (i = 0; i < 6; i = i + 1) begin
                    add_a[i*32 +: 32]     <= add_out[(2*i)*32 +: 32];
                    add_b[i*32 +: 32]     <= add_out[(2*i+1)*32 +: 32];
                    add_a[(6+i)*32 +: 32] <= add_out[(12+2*i)*32 +: 32];
                    add_b[(6+i)*32 +: 32] <= add_out[(13+2*i)*32 +: 32];
                end
                state <= S_RED3;
            end

            S_RED3: begin
                for (i = 0; i < 3; i = i + 1) begin
                    add_a[i*32 +: 32]     <= add_out[(2*i)*32 +: 32];
                    add_b[i*32 +: 32]     <= add_out[(2*i+1)*32 +: 32];
                    add_a[(3+i)*32 +: 32] <= add_out[(6+2*i)*32 +: 32];
                    add_b[(3+i)*32 +: 32] <= add_out[(7+2*i)*32 +: 32];
                end
                state <= S_RED4;
            end

            S_RED4: begin

                add_a[0*32 +: 32] <= add_out[0*32 +: 32];
                add_b[0*32 +: 32] <= add_out[1*32 +: 32];
                num_pass           <= add_out[2*32 +: 32];

                add_a[1*32 +: 32] <= add_out[3*32 +: 32];
                add_b[1*32 +: 32] <= add_out[4*32 +: 32];
                den_pass           <= add_out[5*32 +: 32];

                state <= S_RED5;
            end

            S_RED5: begin
                add_a[0*32 +: 32] <= add_out[0*32 +: 32];
                add_b[0*32 +: 32] <= num_pass;
                add_a[1*32 +: 32] <= add_out[1*32 +: 32];
                add_b[1*32 +: 32] <= den_pass;
                state <= S_FIRE_DIV;
            end

            S_FIRE_DIV: begin
                div_a       <= add_out[0*32 +: 32];
                div_b       <= add_out[1*32 +: 32];
                div_fire    <= 1'b1;
                div_pending <= 1'b1;
                state       <= S_WAIT_DIV;
            end

            S_WAIT_DIV: begin
                if (valid_in) begin
                    x_max <= cmp_max;
                    state <= S_SUB;
                end else if (!div_pending) begin
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;

            endcase

            if (div_done && div_pending) begin
                y_out       <= div_out;
                valid_out   <= 1'b1;
                div_pending <= 1'b0;
            end

        end
    end

endmodule
