

`timescale 1ns/1ps

module grf (
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

    reg [31:0] x_d1,  x_d2,  x_d3;
    reg [31:0] a0_d1, a0_d2, a0_d3;
    reg [31:0] a1_d1, a1_d2, a1_d3;
    reg [31:0] a2_d1, a2_d2, a2_d3;
    reg [31:0] a3_d1, a3_d2, a3_d3;
    reg [31:0] a4_d1, a4_d2, a4_d3;
    reg [31:0] a5_d1, a5_d2, a5_d3;
    reg [31:0] b0_d1, b0_d2, b0_d3;
    reg [31:0] b1_d1, b1_d2, b1_d3;
    reg [31:0] b2_d1, b2_d2, b2_d3;
    reg [31:0] b3_d1, b3_d2, b3_d3;
    reg        v_d1,  v_d2,  v_d3;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x_d1  <= 0; x_d2  <= 0; x_d3  <= 0;
            a0_d1 <= 0; a0_d2 <= 0; a0_d3 <= 0;
            a1_d1 <= 0; a1_d2 <= 0; a1_d3 <= 0;
            a2_d1 <= 0; a2_d2 <= 0; a2_d3 <= 0;
            a3_d1 <= 0; a3_d2 <= 0; a3_d3 <= 0;
            a4_d1 <= 0; a4_d2 <= 0; a4_d3 <= 0;
            a5_d1 <= 0; a5_d2 <= 0; a5_d3 <= 0;
            b0_d1 <= 0; b0_d2 <= 0; b0_d3 <= 0;
            b1_d1 <= 0; b1_d2 <= 0; b1_d3 <= 0;
            b2_d1 <= 0; b2_d2 <= 0; b2_d3 <= 0;
            b3_d1 <= 0; b3_d2 <= 0; b3_d3 <= 0;
            v_d1  <= 0; v_d2  <= 0; v_d3  <= 0;
        end else begin
            x_d1  <= x_s0;  x_d2  <= x_d1;  x_d3  <= x_d2;
            a0_d1 <= a0_s0; a0_d2 <= a0_d1; a0_d3 <= a0_d2;
            a1_d1 <= a1_s0; a1_d2 <= a1_d1; a1_d3 <= a1_d2;
            a2_d1 <= a2_s0; a2_d2 <= a2_d1; a2_d3 <= a2_d2;
            a3_d1 <= a3_s0; a3_d2 <= a3_d1; a3_d3 <= a3_d2;
            a4_d1 <= a4_s0; a4_d2 <= a4_d1; a4_d3 <= a4_d2;
            a5_d1 <= a5_s0; a5_d2 <= a5_d1; a5_d3 <= a5_d2;
            b0_d1 <= b0_s0; b0_d2 <= b0_d1; b0_d3 <= b0_d2;
            b1_d1 <= b1_s0; b1_d2 <= b1_d1; b1_d3 <= b1_d2;
            b2_d1 <= b2_s0; b2_d2 <= b2_d1; b2_d3 <= b2_d2;
            b3_d1 <= b3_s0; b3_d2 <= b3_d1; b3_d3 <= b3_d2;
            v_d1  <= v_s0;  v_d2  <= v_d1;  v_d3  <= v_d2;
        end
    end

    reg [31:0] x_for_x3;
    always @(posedge clk or negedge rstn)
        if (!rstn) x_for_x3 <= 0;
        else       x_for_x3 <= x_s0;

    wire [31:0] x2_out;
    wire        x2_valid;
    fp32_mul u_mul_x2 (
        .clk(clk), .rst_n(rstn), .valid_in(v_s0),
        .a(x_s0), .b(x_s0),
        .result(x2_out), .valid_out(x2_valid)
    );

    wire [31:0] x3_out, x4_out;
    wire        x3_valid, x4_valid;
    fp32_mul u_mul_x3 (
        .clk(clk), .rst_n(rstn), .valid_in(x2_valid),
        .a(x2_out), .b(x_for_x3),
        .result(x3_out), .valid_out(x3_valid)
    );
    fp32_mul u_mul_x4 (
        .clk(clk), .rst_n(rstn), .valid_in(x2_valid),
        .a(x2_out), .b(x2_out),
        .result(x4_out), .valid_out(x4_valid)
    );

    reg [31:0] x2_for_x5;
    always @(posedge clk or negedge rstn)
        if (!rstn) x2_for_x5 <= 0;
        else       x2_for_x5 <= x2_out;

    wire [31:0] x5_out;
    wire        x5_valid;
    fp32_mul u_mul_x5 (
        .clk(clk), .rst_n(rstn), .valid_in(x3_valid),
        .a(x3_out), .b(x2_for_x5),
        .result(x5_out), .valid_out(x5_valid)
    );

    reg [31:0] x2_dly1, x2_dly2;
    reg [31:0] x3_dly1;
    reg [31:0] x4_dly1;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x2_dly1 <= 0; x2_dly2 <= 0;
            x3_dly1 <= 0; x4_dly1 <= 0;
        end else begin
            x2_dly1 <= x2_out;  x2_dly2 <= x2_dly1;
            x3_dly1 <= x3_out;  x4_dly1 <= x4_out;
        end
    end

    reg [31:0] x_aln, x2_aln, x3_aln, x4_aln, x5_aln;
    reg [31:0] a0_aln, a1_aln, a2_aln, a3_aln, a4_aln, a5_aln;
    reg [31:0] b0_aln, b1_aln, b2_aln, b3_aln;
    reg        valid_align;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            x_aln  <= 0; x2_aln <= 0; x3_aln <= 0; x4_aln <= 0; x5_aln <= 0;
            a0_aln <= 0; a1_aln <= 0; a2_aln <= 0;
            a3_aln <= 0; a4_aln <= 0; a5_aln <= 0;
            b0_aln <= 0; b1_aln <= 0; b2_aln <= 0; b3_aln <= 0;
            valid_align <= 0;
        end else begin
            if (x5_valid) begin
                x_aln  <= x_d3;
                x2_aln <= x2_dly2;
                x3_aln <= x3_dly1;
                x4_aln <= x4_dly1;
                x5_aln <= x5_out;
                a0_aln <= a0_d3; a1_aln <= a1_d3; a2_aln <= a2_d3;
                a3_aln <= a3_d3; a4_aln <= a4_d3; a5_aln <= a5_d3;
                b0_aln <= b0_d3; b1_aln <= b1_d3; b2_aln <= b2_d3; b3_aln <= b3_d3;
            end
            valid_align <= x5_valid;
        end
    end

    wire [31:0] p1_f, p2_f, p3_f, p4_f, p5_f;
    wire [31:0] q1_f, q2_f, q3_f, q4_f;
    wire        coef_valid;

    fp32_mul u_p1 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(a1_aln), .b(x_aln),  .result(p1_f), .valid_out(coef_valid));
    fp32_mul u_p2 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(a2_aln), .b(x2_aln), .result(p2_f), .valid_out());
    fp32_mul u_p3 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(a3_aln), .b(x3_aln), .result(p3_f), .valid_out());
    fp32_mul u_p4 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(a4_aln), .b(x4_aln), .result(p4_f), .valid_out());
    fp32_mul u_p5 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(a5_aln), .b(x5_aln), .result(p5_f), .valid_out());

    fp32_mul u_q1 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(b0_aln), .b(x_aln),  .result(q1_f), .valid_out());
    fp32_mul u_q2 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(b1_aln), .b(x2_aln), .result(q2_f), .valid_out());
    fp32_mul u_q3 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(b2_aln), .b(x3_aln), .result(q3_f), .valid_out());
    fp32_mul u_q4 (.clk(clk), .rst_n(rstn), .valid_in(valid_align), .a(b3_aln), .b(x4_aln), .result(q4_f), .valid_out());

    reg [31:0] a0_coef;
    always @(posedge clk or negedge rstn)
        if (!rstn) a0_coef <= 0;
        else       a0_coef <= a0_aln;

    wire [31:0] sum_p1_l1, sum_p2_l1, sum_p3_l1;
    wire [31:0] sum_q1_l1, sum_q2_l1;
    wire        l1_valid;

    fp32_add u_add_p1 (.clk(clk), .rst_n(rstn), .valid_in(coef_valid), .a(a0_coef),  .b(p1_f), .result(sum_p1_l1), .valid_out(l1_valid));
    fp32_add u_add_p2 (.clk(clk), .rst_n(rstn), .valid_in(coef_valid), .a(p2_f),     .b(p3_f), .result(sum_p2_l1), .valid_out());
    fp32_add u_add_p3 (.clk(clk), .rst_n(rstn), .valid_in(coef_valid), .a(p4_f),     .b(p5_f), .result(sum_p3_l1), .valid_out());
    fp32_add u_add_q1 (.clk(clk), .rst_n(rstn), .valid_in(coef_valid), .a(FP32_ONE), .b(q1_f), .result(sum_q1_l1), .valid_out());
    fp32_add u_add_q2 (.clk(clk), .rst_n(rstn), .valid_in(coef_valid), .a(q2_f),     .b(q3_f), .result(sum_q2_l1), .valid_out());

    reg [31:0] sum_p3_d1, sum_p3_d2;
    reg [31:0] q4_d1, q4_d2;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            sum_p3_d1 <= 0; sum_p3_d2 <= 0;
            q4_d1     <= 0; q4_d2     <= 0;
        end else begin
            sum_p3_d1 <= sum_p3_l1; sum_p3_d2 <= sum_p3_d1;
            q4_d1     <= q4_f;      q4_d2     <= q4_d1;
        end
    end

    wire [31:0] sum_p_l2, sum_q_l2;
    wire        l2_valid;

    fp32_add u_add_p_l2 (.clk(clk), .rst_n(rstn), .valid_in(l1_valid), .a(sum_p1_l1), .b(sum_p2_l1), .result(sum_p_l2), .valid_out(l2_valid));
    fp32_add u_add_q_l2 (.clk(clk), .rst_n(rstn), .valid_in(l1_valid), .a(sum_q1_l1), .b(sum_q2_l1), .result(sum_q_l2), .valid_out());

    wire [31:0] P_fin, Q_fin;
    wire        pq_valid_p, pq_valid_q;

    fp32_add u_add_p_l3 (.clk(clk), .rst_n(rstn), .valid_in(l2_valid), .a(sum_p_l2), .b(sum_p3_d2), .result(P_fin), .valid_out(pq_valid_p));
    fp32_add u_add_q_l3 (.clk(clk), .rst_n(rstn), .valid_in(l2_valid), .a(sum_q_l2), .b(q4_d2),     .result(Q_fin), .valid_out(pq_valid_q));

    reg [31:0] P_reg, Q_reg;
    reg        valid_pq;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            P_reg    <= 0;
            Q_reg    <= 0;
            valid_pq <= 0;
        end else begin
            if (pq_valid_p && pq_valid_q) begin
                P_reg    <= P_fin;
                Q_reg    <= Q_fin;
                valid_pq <= 1'b1;
            end else
                valid_pq <= 1'b0;
        end
    end

    reg valid_pq_d1;
    always @(posedge clk or negedge rstn)
        if (!rstn) valid_pq_d1 <= 0;
        else       valid_pq_d1 <= valid_pq;

    fp32_div u_div (
        .clk      (clk),
        .rst_n    (rstn),
        .valid_in (valid_pq_d1),
        .a        (P_reg),
        .b        (Q_reg),
        .result   (y),
        .valid_out(valid)
    );

endmodule