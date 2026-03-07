// GRF: Generalized Rational Function (strictly pipelined)
// y = P/Q
// P = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
// Q = 1 + b0*x + b1*x^2 + b2*x^3 + b3*x^4

// iverilog -o tb_grf.vvp -s tb_grf tb_grf.v grf.v fp32_mul.v fp32_adder.v div_newton.v

`timescale 1ns/1ps

module grf #(
    parameter [1:0] RM = 2'b00
) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        ena,
    input  wire [31:0] x,
    input  wire [31:0] a0, a1, a2, a3, a4, a5,
    input  wire [31:0] b0, b1, b2, b3,
    output wire [31:0] y,
    output wire        valid
);

    localparam [31:0] FP32_ONE = 32'h3f800000;

    // =========================================================
    // Stage 0 : input capture
    // =========================================================
    reg [31:0] x_s0;
    reg [31:0] a0_s0, a1_s0, a2_s0, a3_s0, a4_s0, a5_s0;
    reg [31:0] b0_s0, b1_s0, b2_s0, b3_s0;
    reg        v_s0;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x_s0  <= 32'b0;
            a0_s0 <= 32'b0; a1_s0 <= 32'b0; a2_s0 <= 32'b0;
            a3_s0 <= 32'b0; a4_s0 <= 32'b0; a5_s0 <= 32'b0;
            b0_s0 <= 32'b0; b1_s0 <= 32'b0; b2_s0 <= 32'b0; b3_s0 <= 32'b0;
            v_s0  <= 1'b0;
        end else begin
            if (ena) begin
                x_s0  <= x;
                a0_s0 <= a0; a1_s0 <= a1; a2_s0 <= a2;
                a3_s0 <= a3; a4_s0 <= a4; a5_s0 <= a5;
                b0_s0 <= b0; b1_s0 <= b1; b2_s0 <= b2; b3_s0 <= b3;
            end
            v_s0 <= ena;
        end
    end

    // =========================================================
    // Delay original sample / coeffs by 9 cycles
    // so they align with x5 output (3 mul stages: x2/x3/x5)
    // =========================================================
    reg [31:0] x_d1,  x_d2,  x_d3,  x_d4,  x_d5,  x_d6,  x_d7,  x_d8,  x_d9;
    reg [31:0] a0_d1, a0_d2, a0_d3, a0_d4, a0_d5, a0_d6, a0_d7, a0_d8, a0_d9;
    reg [31:0] a1_d1, a1_d2, a1_d3, a1_d4, a1_d5, a1_d6, a1_d7, a1_d8, a1_d9;
    reg [31:0] a2_d1, a2_d2, a2_d3, a2_d4, a2_d5, a2_d6, a2_d7, a2_d8, a2_d9;
    reg [31:0] a3_d1, a3_d2, a3_d3, a3_d4, a3_d5, a3_d6, a3_d7, a3_d8, a3_d9;
    reg [31:0] a4_d1, a4_d2, a4_d3, a4_d4, a4_d5, a4_d6, a4_d7, a4_d8, a4_d9;
    reg [31:0] a5_d1, a5_d2, a5_d3, a5_d4, a5_d5, a5_d6, a5_d7, a5_d8, a5_d9;
    reg [31:0] b0_d1, b0_d2, b0_d3, b0_d4, b0_d5, b0_d6, b0_d7, b0_d8, b0_d9;
    reg [31:0] b1_d1, b1_d2, b1_d3, b1_d4, b1_d5, b1_d6, b1_d7, b1_d8, b1_d9;
    reg [31:0] b2_d1, b2_d2, b2_d3, b2_d4, b2_d5, b2_d6, b2_d7, b2_d8, b2_d9;
    reg [31:0] b3_d1, b3_d2, b3_d3, b3_d4, b3_d5, b3_d6, b3_d7, b3_d8, b3_d9;
    reg        v_d1,  v_d2,  v_d3,  v_d4,  v_d5,  v_d6,  v_d7,  v_d8,  v_d9;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x_d1 <= 32'b0;  x_d2 <= 32'b0;  x_d3 <= 32'b0;  x_d4 <= 32'b0;  x_d5 <= 32'b0;
            x_d6 <= 32'b0;  x_d7 <= 32'b0;  x_d8 <= 32'b0;  x_d9 <= 32'b0;

            a0_d1 <= 32'b0; a0_d2 <= 32'b0; a0_d3 <= 32'b0; a0_d4 <= 32'b0; a0_d5 <= 32'b0;
            a0_d6 <= 32'b0; a0_d7 <= 32'b0; a0_d8 <= 32'b0; a0_d9 <= 32'b0;

            a1_d1 <= 32'b0; a1_d2 <= 32'b0; a1_d3 <= 32'b0; a1_d4 <= 32'b0; a1_d5 <= 32'b0;
            a1_d6 <= 32'b0; a1_d7 <= 32'b0; a1_d8 <= 32'b0; a1_d9 <= 32'b0;

            a2_d1 <= 32'b0; a2_d2 <= 32'b0; a2_d3 <= 32'b0; a2_d4 <= 32'b0; a2_d5 <= 32'b0;
            a2_d6 <= 32'b0; a2_d7 <= 32'b0; a2_d8 <= 32'b0; a2_d9 <= 32'b0;

            a3_d1 <= 32'b0; a3_d2 <= 32'b0; a3_d3 <= 32'b0; a3_d4 <= 32'b0; a3_d5 <= 32'b0;
            a3_d6 <= 32'b0; a3_d7 <= 32'b0; a3_d8 <= 32'b0; a3_d9 <= 32'b0;

            a4_d1 <= 32'b0; a4_d2 <= 32'b0; a4_d3 <= 32'b0; a4_d4 <= 32'b0; a4_d5 <= 32'b0;
            a4_d6 <= 32'b0; a4_d7 <= 32'b0; a4_d8 <= 32'b0; a4_d9 <= 32'b0;

            a5_d1 <= 32'b0; a5_d2 <= 32'b0; a5_d3 <= 32'b0; a5_d4 <= 32'b0; a5_d5 <= 32'b0;
            a5_d6 <= 32'b0; a5_d7 <= 32'b0; a5_d8 <= 32'b0; a5_d9 <= 32'b0;

            b0_d1 <= 32'b0; b0_d2 <= 32'b0; b0_d3 <= 32'b0; b0_d4 <= 32'b0; b0_d5 <= 32'b0;
            b0_d6 <= 32'b0; b0_d7 <= 32'b0; b0_d8 <= 32'b0; b0_d9 <= 32'b0;

            b1_d1 <= 32'b0; b1_d2 <= 32'b0; b1_d3 <= 32'b0; b1_d4 <= 32'b0; b1_d5 <= 32'b0;
            b1_d6 <= 32'b0; b1_d7 <= 32'b0; b1_d8 <= 32'b0; b1_d9 <= 32'b0;

            b2_d1 <= 32'b0; b2_d2 <= 32'b0; b2_d3 <= 32'b0; b2_d4 <= 32'b0; b2_d5 <= 32'b0;
            b2_d6 <= 32'b0; b2_d7 <= 32'b0; b2_d8 <= 32'b0; b2_d9 <= 32'b0;

            b3_d1 <= 32'b0; b3_d2 <= 32'b0; b3_d3 <= 32'b0; b3_d4 <= 32'b0; b3_d5 <= 32'b0;
            b3_d6 <= 32'b0; b3_d7 <= 32'b0; b3_d8 <= 32'b0; b3_d9 <= 32'b0;

            v_d1 <= 1'b0; v_d2 <= 1'b0; v_d3 <= 1'b0; v_d4 <= 1'b0; v_d5 <= 1'b0;
            v_d6 <= 1'b0; v_d7 <= 1'b0; v_d8 <= 1'b0; v_d9 <= 1'b0;
        end else begin
            x_d1 <= x_s0; x_d2 <= x_d1; x_d3 <= x_d2; x_d4 <= x_d3; x_d5 <= x_d4;
            x_d6 <= x_d5; x_d7 <= x_d6; x_d8 <= x_d7; x_d9 <= x_d8;

            a0_d1 <= a0_s0; a0_d2 <= a0_d1; a0_d3 <= a0_d2; a0_d4 <= a0_d3; a0_d5 <= a0_d4;
            a0_d6 <= a0_d5; a0_d7 <= a0_d6; a0_d8 <= a0_d7; a0_d9 <= a0_d8;

            a1_d1 <= a1_s0; a1_d2 <= a1_d1; a1_d3 <= a1_d2; a1_d4 <= a1_d3; a1_d5 <= a1_d4;
            a1_d6 <= a1_d5; a1_d7 <= a1_d6; a1_d8 <= a1_d7; a1_d9 <= a1_d8;

            a2_d1 <= a2_s0; a2_d2 <= a2_d1; a2_d3 <= a2_d2; a2_d4 <= a2_d3; a2_d5 <= a2_d4;
            a2_d6 <= a2_d5; a2_d7 <= a2_d6; a2_d8 <= a2_d7; a2_d9 <= a2_d8;

            a3_d1 <= a3_s0; a3_d2 <= a3_d1; a3_d3 <= a3_d2; a3_d4 <= a3_d3; a3_d5 <= a3_d4;
            a3_d6 <= a3_d5; a3_d7 <= a3_d6; a3_d8 <= a3_d7; a3_d9 <= a3_d8;

            a4_d1 <= a4_s0; a4_d2 <= a4_d1; a4_d3 <= a4_d2; a4_d4 <= a4_d3; a4_d5 <= a4_d4;
            a4_d6 <= a4_d5; a4_d7 <= a4_d6; a4_d8 <= a4_d7; a4_d9 <= a4_d8;

            a5_d1 <= a5_s0; a5_d2 <= a5_d1; a5_d3 <= a5_d2; a5_d4 <= a5_d3; a5_d5 <= a5_d4;
            a5_d6 <= a5_d5; a5_d7 <= a5_d6; a5_d8 <= a5_d7; a5_d9 <= a5_d8;

            b0_d1 <= b0_s0; b0_d2 <= b0_d1; b0_d3 <= b0_d2; b0_d4 <= b0_d3; b0_d5 <= b0_d4;
            b0_d6 <= b0_d5; b0_d7 <= b0_d6; b0_d8 <= b0_d7; b0_d9 <= b0_d8;

            b1_d1 <= b1_s0; b1_d2 <= b1_d1; b1_d3 <= b1_d2; b1_d4 <= b1_d3; b1_d5 <= b1_d4;
            b1_d6 <= b1_d5; b1_d7 <= b1_d6; b1_d8 <= b1_d7; b1_d9 <= b1_d8;

            b2_d1 <= b2_s0; b2_d2 <= b2_d1; b2_d3 <= b2_d2; b2_d4 <= b2_d3; b2_d5 <= b2_d4;
            b2_d6 <= b2_d5; b2_d7 <= b2_d6; b2_d8 <= b2_d7; b2_d9 <= b2_d8;

            b3_d1 <= b3_s0; b3_d2 <= b3_d1; b3_d3 <= b3_d2; b3_d4 <= b3_d3; b3_d5 <= b3_d4;
            b3_d6 <= b3_d5; b3_d7 <= b3_d6; b3_d8 <= b3_d7; b3_d9 <= b3_d8;

            v_d1 <= v_s0; v_d2 <= v_d1; v_d3 <= v_d2; v_d4 <= v_d3; v_d5 <= v_d4;
            v_d6 <= v_d5; v_d7 <= v_d6; v_d8 <= v_d7; v_d9 <= v_d8;
        end
    end

    // =========================================================
    // Power chain
    // x2 = x*x
    // x3 = x2*x
    // x4 = x2*x2
    // x5 = x3*x2
    // =========================================================

    // x delayed by 3 cycles for x3 multiply
    reg [31:0] x_for_x3_d1, x_for_x3_d2, x_for_x3_d3;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x_for_x3_d1 <= 32'b0;
            x_for_x3_d2 <= 32'b0;
            x_for_x3_d3 <= 32'b0;
        end else begin
            x_for_x3_d1 <= x_s0;
            x_for_x3_d2 <= x_for_x3_d1;
            x_for_x3_d3 <= x_for_x3_d2;
        end
    end

    wire [31:0] x2_out;
    wire        x2_valid;
    fp32_mul u_mul_x2 (
        .clk(clk), .rstn(rstn), .ena(v_s0),
        .a(x_s0), .b(x_s0), .rm(RM),
        .s(x2_out), .valid(x2_valid)
    );

    wire [31:0] x3_out, x4_out;
    wire        x3_valid, x4_valid;
    fp32_mul u_mul_x3 (
        .clk(clk), .rstn(rstn), .ena(x2_valid),
        .a(x2_out), .b(x_for_x3_d3), .rm(RM),
        .s(x3_out), .valid(x3_valid)
    );
    fp32_mul u_mul_x4 (
        .clk(clk), .rstn(rstn), .ena(x2_valid),
        .a(x2_out), .b(x2_out), .rm(RM),
        .s(x4_out), .valid(x4_valid)
    );

    // delay x2 by 3 cycles so it aligns with x3_out for x5 = x3 * x2
    reg [31:0] x2_for_x5_d1, x2_for_x5_d2, x2_for_x5_d3;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x2_for_x5_d1 <= 32'b0;
            x2_for_x5_d2 <= 32'b0;
            x2_for_x5_d3 <= 32'b0;
        end else begin
            x2_for_x5_d1 <= x2_out;
            x2_for_x5_d2 <= x2_for_x5_d1;
            x2_for_x5_d3 <= x2_for_x5_d2;
        end
    end

    wire [31:0] x5_out;
    wire        x5_valid;
    fp32_mul u_mul_x5 (
        .clk(clk), .rstn(rstn), .ena(x3_valid),
        .a(x3_out), .b(x2_for_x5_d3), .rm(RM),
        .s(x5_out), .valid(x5_valid)
    );

    // =========================================================
    // Align x2/x3/x4 to x5 timing
    // x2 needs +6 cycles after x2_out
    // x3/x4 need +3 cycles after x3_out/x4_out
    // =========================================================
    reg [31:0] x2_dly1, x2_dly2, x2_dly3, x2_dly4, x2_dly5, x2_dly6;
    reg [31:0] x3_dly1, x3_dly2, x3_dly3;
    reg [31:0] x4_dly1, x4_dly2, x4_dly3;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x2_dly1 <= 32'b0; x2_dly2 <= 32'b0; x2_dly3 <= 32'b0;
            x2_dly4 <= 32'b0; x2_dly5 <= 32'b0; x2_dly6 <= 32'b0;
            x3_dly1 <= 32'b0; x3_dly2 <= 32'b0; x3_dly3 <= 32'b0;
            x4_dly1 <= 32'b0; x4_dly2 <= 32'b0; x4_dly3 <= 32'b0;
        end else begin
            x2_dly1 <= x2_out;    x2_dly2 <= x2_dly1; x2_dly3 <= x2_dly2;
            x2_dly4 <= x2_dly3;   x2_dly5 <= x2_dly4; x2_dly6 <= x2_dly5;

            x3_dly1 <= x3_out;    x3_dly2 <= x3_dly1; x3_dly3 <= x3_dly2;
            x4_dly1 <= x4_out;    x4_dly2 <= x4_dly1; x4_dly3 <= x4_dly2;
        end
    end

    // =========================================================
    // Stage ALIGN : latch the whole aligned bundle
    // IMPORTANT:
    //   first latch aligned inputs when x5_valid arrives
    //   then on NEXT cycle use valid_align to start coef muls
    // =========================================================
    reg [31:0] x_aln, x2_aln, x3_aln, x4_aln, x5_aln;
    reg [31:0] a0_aln, a1_aln, a2_aln, a3_aln, a4_aln, a5_aln;
    reg [31:0] b0_aln, b1_aln, b2_aln, b3_aln;
    reg        valid_align;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x_aln <= 32'b0;  x2_aln <= 32'b0; x3_aln <= 32'b0; x4_aln <= 32'b0; x5_aln <= 32'b0;
            a0_aln <= 32'b0; a1_aln <= 32'b0; a2_aln <= 32'b0; a3_aln <= 32'b0; a4_aln <= 32'b0; a5_aln <= 32'b0;
            b0_aln <= 32'b0; b1_aln <= 32'b0; b2_aln <= 32'b0; b3_aln <= 32'b0;
            valid_align <= 1'b0;
        end else begin
            if (x5_valid) begin
                x_aln  <= x_d9;
                x2_aln <= x2_dly6;
                x3_aln <= x3_dly3;
                x4_aln <= x4_dly3;
                x5_aln <= x5_out;

                a0_aln <= a0_d9; a1_aln <= a1_d9; a2_aln <= a2_d9;
                a3_aln <= a3_d9; a4_aln <= a4_d9; a5_aln <= a5_d9;

                b0_aln <= b0_d9; b1_aln <= b1_d9; b2_aln <= b2_d9; b3_aln <= b3_d9;
            end
            valid_align <= x5_valid;
        end
    end

    // =========================================================
    // Coefficient products
    // launched one full cycle after alignment latch
    // =========================================================
    wire [31:0] p1_f, p2_f, p3_f, p4_f, p5_f;
    wire [31:0] q1_f, q2_f, q3_f, q4_f;
    wire        coef_valid;

    fp32_mul u_p1 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(a1_aln), .b(x_aln),  .rm(RM), .s(p1_f), .valid(coef_valid));
    fp32_mul u_p2 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(a2_aln), .b(x2_aln), .rm(RM), .s(p2_f), .valid());
    fp32_mul u_p3 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(a3_aln), .b(x3_aln), .rm(RM), .s(p3_f), .valid());
    fp32_mul u_p4 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(a4_aln), .b(x4_aln), .rm(RM), .s(p4_f), .valid());
    fp32_mul u_p5 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(a5_aln), .b(x5_aln), .rm(RM), .s(p5_f), .valid());

    fp32_mul u_q1 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(b0_aln), .b(x_aln),  .rm(RM), .s(q1_f), .valid());
    fp32_mul u_q2 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(b1_aln), .b(x2_aln), .rm(RM), .s(q2_f), .valid());
    fp32_mul u_q3 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(b2_aln), .b(x3_aln), .rm(RM), .s(q3_f), .valid());
    fp32_mul u_q4 (.clk(clk), .rstn(rstn), .ena(valid_align), .a(b3_aln), .b(x4_aln), .rm(RM), .s(q4_f), .valid());

    // delay a0 by 3 cycles to align with p1..p5 at coef_valid
    reg [31:0] a0_coef_d1, a0_coef_d2, a0_coef_d3;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            a0_coef_d1 <= 32'b0;
            a0_coef_d2 <= 32'b0;
            a0_coef_d3 <= 32'b0;
        end else begin
            a0_coef_d1 <= a0_aln;
            a0_coef_d2 <= a0_coef_d1;
            a0_coef_d3 <= a0_coef_d2;
        end
    end

    // =========================================================
    // Adder tree - Level 1
    // P: (a0+p1), (p2+p3), (p4+p5)
    // Q: (1+q1), (q2+q3), and delay q4 by 3 cycles
    // =========================================================
    wire [31:0] sum_p1_l1, sum_p2_l1, sum_p3_l1;
    wire [31:0] sum_q1_l1, sum_q2_l1;
    wire        l1_valid;

    fp32_adder u_add_p1 (
        .clk(clk), .rstn(rstn), .ena(coef_valid),
        .a(a0_coef_d3), .b(p1_f), .rm(RM), .sel(1'b0),
        .s(sum_p1_l1), .valid(l1_valid)
    );

    fp32_adder u_add_p2 (
        .clk(clk), .rstn(rstn), .ena(coef_valid),
        .a(p2_f), .b(p3_f), .rm(RM), .sel(1'b0),
        .s(sum_p2_l1), .valid()
    );

    fp32_adder u_add_p3 (
        .clk(clk), .rstn(rstn), .ena(coef_valid),
        .a(p4_f), .b(p5_f), .rm(RM), .sel(1'b0),
        .s(sum_p3_l1), .valid()
    );

    fp32_adder u_add_q1 (
        .clk(clk), .rstn(rstn), .ena(coef_valid),
        .a(FP32_ONE), .b(q1_f), .rm(RM), .sel(1'b0),
        .s(sum_q1_l1), .valid()
    );

    fp32_adder u_add_q2 (
        .clk(clk), .rstn(rstn), .ena(coef_valid),
        .a(q2_f), .b(q3_f), .rm(RM), .sel(1'b0),
        .s(sum_q2_l1), .valid()
    );

    // delay sum_p3 and q4 by 3 cycles so they align with level-2 outputs
    reg [31:0] sum_p3_d1, sum_p3_d2, sum_p3_d3;
    reg [31:0] q4_d1, q4_d2, q4_d3;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            sum_p3_d1 <= 32'b0; sum_p3_d2 <= 32'b0; sum_p3_d3 <= 32'b0;
            q4_d1     <= 32'b0; q4_d2     <= 32'b0; q4_d3     <= 32'b0;
        end else begin
            sum_p3_d1 <= sum_p3_l1;
            sum_p3_d2 <= sum_p3_d1;
            sum_p3_d3 <= sum_p3_d2;

            q4_d1 <= q4_f;
            q4_d2 <= q4_d1;
            q4_d3 <= q4_d2;
        end
    end

    // =========================================================
    // Adder tree - Level 2
    // P: sum_p1_l1 + sum_p2_l1
    // Q: sum_q1_l1 + sum_q2_l1
    // =========================================================
    wire [31:0] sum_p_l2, sum_q_l2;
    wire        l2_valid_p, l2_valid_q;

    fp32_adder u_add_p_l2 (
        .clk(clk), .rstn(rstn), .ena(l1_valid),
        .a(sum_p1_l1), .b(sum_p2_l1), .rm(RM), .sel(1'b0),
        .s(sum_p_l2), .valid(l2_valid_p)
    );

    fp32_adder u_add_q_l2 (
        .clk(clk), .rstn(rstn), .ena(l1_valid),
        .a(sum_q1_l1), .b(sum_q2_l1), .rm(RM), .sel(1'b0),
        .s(sum_q_l2), .valid(l2_valid_q)
    );

    // =========================================================
    // Adder tree - Level 3
    // P = sum_p_l2 + delayed(sum_p3_l1)
    // Q = sum_q_l2 + delayed(q4)
    // =========================================================
    wire [31:0] P_fin, Q_fin;
    wire        pq_valid_p, pq_valid_q;

    fp32_adder u_add_p_l3 (
        .clk(clk), .rstn(rstn), .ena(l2_valid_p),
        .a(sum_p_l2), .b(sum_p3_d3), .rm(RM), .sel(1'b0),
        .s(P_fin), .valid(pq_valid_p)
    );

    fp32_adder u_add_q_l3 (
        .clk(clk), .rstn(rstn), .ena(l2_valid_q),
        .a(sum_q_l2), .b(q4_d3), .rm(RM), .sel(1'b0),
        .s(Q_fin), .valid(pq_valid_q)
    );

    // =========================================================
    // Latch P and Q together
    // =========================================================
    reg [31:0] P_reg, Q_reg;
    reg        valid_pq;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            P_reg    <= 32'b0;
            Q_reg    <= 32'b0;
            valid_pq <= 1'b0;
        end else begin
            if (pq_valid_p && pq_valid_q) begin
                P_reg    <= P_fin;
                Q_reg    <= Q_fin;
                valid_pq <= 1'b1;
            end else begin
                valid_pq <= 1'b0;
            end
        end
    end

    // =========================================================
    // Divider launch
    // IMPORTANT:
    //   P_reg / Q_reg are captured first,
    //   then one cycle later start divider.
    // =========================================================
    reg valid_pq_d1;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            valid_pq_d1 <= 1'b0;
        end else begin
            valid_pq_d1 <= valid_pq;
        end
    end

    wire fdiv_start = valid_pq_d1;

    wire        div_busy, div_stall;
    wire [25:0] reg_x;
    wire [4:0]  count;

    fdiv_newton u_div (
        .a(P_reg),
        .b(Q_reg),
        .rm(RM),
        .fdiv(fdiv_start),
        .ena(1'b1),
        .clk(clk),
        .clrn(rstn),
        .s(y),
        .reg_x(reg_x),
        .count(count),
        .busy(div_busy),
        .stall(div_stall),
        .valid(valid)
    );

endmodule