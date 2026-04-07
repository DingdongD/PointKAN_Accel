module op_spool #(
    parameter K = 24,
    parameter P = 4
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire [K*32-1:0]   x_in,      
    input  wire              valid_in,
    output reg  [31:0]       y_out,
    output reg               valid_out
);
    localparam ROUNDS = K / P;  
    localparam [31:0] FP_ZERO    = 32'h00000000;
    localparam [31:0] FP_NEG_INF = 32'hFF800000;
    localparam [31:0] LOG2E      = 32'h3FB8AA3B; 
    localparam [31:0] COEFF_C0   = 32'h3F800000; 
    localparam [31:0] COEFF_C1   = 32'h3F317218; 
    localparam [31:0] COEFF_C2   = 32'h3E75FDF0; 
    localparam [4:0] S_IDLE    = 5'd0;
    localparam [4:0] S_PH1     = 5'd1;
    localparam [4:0] S_SUB     = 5'd2;
    localparam [4:0] S_MUL_P   = 5'd3;
    localparam [4:0] S_SPLIT_F = 5'd4;
    localparam [4:0] S_MUL_C2F = 5'd5;
    localparam [4:0] S_ADD_C1  = 5'd6;
    localparam [4:0] S_MUL_TF  = 5'd7;
    localparam [4:0] S_ADD_C0  = 5'd8;
    localparam [4:0] S_MUL_EX  = 5'd9;
    localparam [4:0] S_RED4    = 5'd10;
    localparam [4:0] S_RED2    = 5'd11;
    localparam [4:0] S_ACC     = 5'd12;
    localparam [4:0] S_RDONE   = 5'd13;
    localparam [4:0] S_DIV     = 5'd14;
    localparam [4:0] S_OUT     = 5'd15;
    reg [4:0] state;
    reg [2:0] round_cnt;
    reg [31:0] xr0,  xr1,  xr2,  xr3,  xr4,  xr5,  xr6,  xr7;
    reg [31:0] xr8,  xr9,  xr10, xr11, xr12, xr13, xr14, xr15;
    reg [31:0] xr16, xr17, xr18, xr19, xr20, xr21, xr22, xr23;
    always @(posedge clk) begin
        if (valid_in && state == S_IDLE) begin
            xr0  <= x_in[ 0*32 +: 32]; xr1  <= x_in[ 1*32 +: 32];
            xr2  <= x_in[ 2*32 +: 32]; xr3  <= x_in[ 3*32 +: 32];
            xr4  <= x_in[ 4*32 +: 32]; xr5  <= x_in[ 5*32 +: 32];
            xr6  <= x_in[ 6*32 +: 32]; xr7  <= x_in[ 7*32 +: 32];
            xr8  <= x_in[ 8*32 +: 32]; xr9  <= x_in[ 9*32 +: 32];
            xr10 <= x_in[10*32 +: 32]; xr11 <= x_in[11*32 +: 32];
            xr12 <= x_in[12*32 +: 32]; xr13 <= x_in[13*32 +: 32];
            xr14 <= x_in[14*32 +: 32]; xr15 <= x_in[15*32 +: 32];
            xr16 <= x_in[16*32 +: 32]; xr17 <= x_in[17*32 +: 32];
            xr18 <= x_in[18*32 +: 32]; xr19 <= x_in[19*32 +: 32];
            xr20 <= x_in[20*32 +: 32]; xr21 <= x_in[21*32 +: 32];
            xr22 <= x_in[22*32 +: 32]; xr23 <= x_in[23*32 +: 32];
        end
    end
    reg [31:0] sel0, sel1, sel2, sel3;
    always @(*) begin
        case (round_cnt)
            3'd0: begin sel0 = xr0;  sel1 = xr1;  sel2 = xr2;  sel3 = xr3;  end
            3'd1: begin sel0 = xr4;  sel1 = xr5;  sel2 = xr6;  sel3 = xr7;  end
            3'd2: begin sel0 = xr8;  sel1 = xr9;  sel2 = xr10; sel3 = xr11; end
            3'd3: begin sel0 = xr12; sel1 = xr13; sel2 = xr14; sel3 = xr15; end
            3'd4: begin sel0 = xr16; sel1 = xr17; sel2 = xr18; sel3 = xr19; end
            3'd5: begin sel0 = xr20; sel1 = xr21; sel2 = xr22; sel3 = xr23; end
            default: begin sel0 = FP_ZERO; sel1 = FP_ZERO; sel2 = FP_ZERO; sel3 = FP_ZERO; end
        endcase
    end
    function [31:0] fmax;
        input [31:0] a, b;
        reg ag;
        begin
            if (a[31] && !b[31])      ag = 1'b0;
            else if (!a[31] && b[31]) ag = 1'b1;
            else if (!a[31])          ag = (a[30:0] > b[30:0]);
            else                      ag = (a[30:0] < b[30:0]);
            fmax = ag ? a : b;
        end
    endfunction
    wire [31:0] m1a  = fmax(xr0,  xr1);
    wire [31:0] m1b  = fmax(xr2,  xr3);
    wire [31:0] m1c  = fmax(xr4,  xr5);
    wire [31:0] m1d  = fmax(xr6,  xr7);
    wire [31:0] m1e  = fmax(xr8,  xr9);
    wire [31:0] m1f  = fmax(xr10, xr11);
    wire [31:0] m1g  = fmax(xr12, xr13);
    wire [31:0] m1h  = fmax(xr14, xr15);
    wire [31:0] m1i  = fmax(xr16, xr17);
    wire [31:0] m1j  = fmax(xr18, xr19);
    wire [31:0] m1k  = fmax(xr20, xr21);
    wire [31:0] m1l  = fmax(xr22, xr23);
    wire [31:0] m2a = fmax(m1a, m1b);
    wire [31:0] m2b = fmax(m1c, m1d);
    wire [31:0] m2c = fmax(m1e, m1f);
    wire [31:0] m2d = fmax(m1g, m1h);
    wire [31:0] m2e = fmax(m1i, m1j);
    wire [31:0] m2f = fmax(m1k, m1l);
    wire [31:0] m3a = fmax(m2a, m2b);
    wire [31:0] m3b = fmax(m2c, m2d);
    wire [31:0] m3c = fmax(m2e, m2f);
    wire [31:0] m4a    = fmax(m3a, m3b);
    wire [31:0] cmp_max = fmax(m4a, m3c);
    reg [31:0] x_max;
    function [31:0] fp32_floor;
        input [31:0] p;
        reg        s;
        reg [7:0]  e;
        reg [22:0] m;
        reg [23:0] fm;
        reg [23:0] mask, ipart, inc;
        reg [7:0]  sh;
        reg        has_frac;
        begin
            s  = p[31];
            e  = p[30:23];
            m  = p[22:0];
            fm = {1'b1, m};
            if (e == 8'd0 && m == 23'd0) begin
                fp32_floor = 32'h00000000;
            end else if (e == 8'd0 || e < 8'd127) begin
                fp32_floor = s ? 32'hBF800000 : 32'h00000000;
            end else begin
                sh = e - 8'd127;
                if (sh >= 8'd23) begin
                    fp32_floor = p;
                end else begin
                    mask     = 24'hFFFFFF << (8'd23 - sh);
                    ipart    = fm & mask;
                    has_frac = (fm != ipart);
                    if (s && has_frac) begin
                        inc = ipart + (24'd1 << (8'd23 - sh));
                        if (inc[23]) begin
                            fp32_floor = {1'b1, e, inc[22:0]};
                        end else begin
                            fp32_floor = {1'b1, e + 8'd1, 23'd0};
                        end
                    end else begin
                        fp32_floor = {s, e, ipart[22:0]};
                    end
                end
            end
        end
    endfunction
    function [31:0] exp_adj;
        input [31:0] val;
        input [31:0] n_fp;
        reg [7:0]  ve, ne;
        reg [22:0] nm;
        reg        ns;
        reg [7:0]  sh;
        reg [8:0]  n_abs;
        reg [9:0]  new_e;
        begin
            ve = val[30:23];
            ns = n_fp[31];
            ne = n_fp[30:23];
            nm = n_fp[22:0];
            if (ne < 8'd127) begin
                exp_adj = val;
            end else if (val[30:0] == 31'd0) begin
                exp_adj = val;
            end else begin
                sh = ne - 8'd127;
                if (sh >= 8'd8)
                    n_abs = 9'd255;  
                else
                    n_abs = {1'b1, nm[22:16]} >> (4'd7 - sh[3:0]);
                if (ns) begin
                    if ({1'b0, ve} < {1'b0, n_abs[7:0]})
                        exp_adj = {val[31], 31'd0}; 
                    else begin
                        new_e = {2'b00, ve} - {1'b0, n_abs};
                        if (new_e == 10'd0)
                            exp_adj = {val[31], 31'd0};
                        else
                            exp_adj = {val[31], new_e[7:0], val[22:0]};
                    end
                end else begin
                    new_e = {2'b00, ve} + {1'b0, n_abs};
                    if (new_e >= 10'd255)
                        exp_adj = {val[31], 8'hFF, 23'd0}; 
                    else
                        exp_adj = {val[31], new_e[7:0], val[22:0]};
                end
            end
        end
    endfunction
    reg  [31:0] ma0, mb0, ma1, mb1, ma2, mb2, ma3, mb3;
    reg         mv0, mv1, mv2, mv3;
    wire [31:0] mr0, mr1, mr2, mr3;
    wire        md0, md1, md2, md3;
    fp32_mul u_m0 (.clk(clk), .rst_n(rst_n), .a(ma0), .b(mb0), .valid_in(mv0), .result(mr0), .valid_out(md0));
    fp32_mul u_m1 (.clk(clk), .rst_n(rst_n), .a(ma1), .b(mb1), .valid_in(mv1), .result(mr1), .valid_out(md1));
    fp32_mul u_m2 (.clk(clk), .rst_n(rst_n), .a(ma2), .b(mb2), .valid_in(mv2), .result(mr2), .valid_out(md2));
    fp32_mul u_m3 (.clk(clk), .rst_n(rst_n), .a(ma3), .b(mb3), .valid_in(mv3), .result(mr3), .valid_out(md3));
    reg  [31:0] aa0, ab0, aa1, ab1, aa2, ab2, aa3, ab3;
    reg         av0, av1, av2, av3;
    wire [31:0] ar0, ar1, ar2, ar3;
    wire        ad0, ad1, ad2, ad3;
    fp32_add u_a0 (.clk(clk), .rst_n(rst_n), .a(aa0), .b(ab0), .valid_in(av0), .result(ar0), .valid_out(ad0));
    fp32_add u_a1 (.clk(clk), .rst_n(rst_n), .a(aa1), .b(ab1), .valid_in(av1), .result(ar1), .valid_out(ad1));
    fp32_add u_a2 (.clk(clk), .rst_n(rst_n), .a(aa2), .b(ab2), .valid_in(av2), .result(ar2), .valid_out(ad2));
    fp32_add u_a3 (.clk(clk), .rst_n(rst_n), .a(aa3), .b(ab3), .valid_in(av3), .result(ar3), .valid_out(ad3));
    reg  [31:0] da, db;
    reg         dv;
    wire [31:0] dr;
    wire        dd;
    fp32_div u_div (.clk(clk), .rst_n(rst_n), .a(da), .b(db), .valid_in(dv), .result(dr), .valid_out(dd));
    reg [31:0] n_r0, n_r1, n_r2, n_r3;   
    reg [31:0] f_r0, f_r1, f_r2, f_r3;   
    reg [31:0] ev_r0, ev_r1, ev_r2, ev_r3; 
    reg [31:0] xo_r0, xo_r1, xo_r2, xo_r3; 
    reg [31:0] sum_num, sum_den;
    wire [31:0] floor0 = fp32_floor(mr0);
    wire [31:0] floor1 = fp32_floor(mr1);
    wire [31:0] floor2 = fp32_floor(mr2);
    wire [31:0] floor3 = fp32_floor(mr3);
    wire [31:0] neg_floor0 = {~floor0[31], floor0[30:0]};
    wire [31:0] neg_floor1 = {~floor1[31], floor1[30:0]};
    wire [31:0] neg_floor2 = {~floor2[31], floor2[30:0]};
    wire [31:0] neg_floor3 = {~floor3[31], floor3[30:0]};
    wire [31:0] eadj0 = exp_adj(ar0, n_r0);
    wire [31:0] eadj1 = exp_adj(ar1, n_r1);
    wire [31:0] eadj2 = exp_adj(ar2, n_r2);
    wire [31:0] eadj3 = exp_adj(ar3, n_r3);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            round_cnt <= 3'd0;
            x_max     <= FP_NEG_INF;
            sum_num   <= FP_ZERO;
            sum_den   <= FP_ZERO;
            y_out     <= FP_ZERO;
            valid_out <= 1'b0;
            dv  <= 1'b0;
            mv0 <= 1'b0; mv1 <= 1'b0; mv2 <= 1'b0; mv3 <= 1'b0;
            av0 <= 1'b0; av1 <= 1'b0; av2 <= 1'b0; av3 <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            dv  <= 1'b0;
            mv0 <= 1'b0; mv1 <= 1'b0; mv2 <= 1'b0; mv3 <= 1'b0;
            av0 <= 1'b0; av1 <= 1'b0; av2 <= 1'b0; av3 <= 1'b0;
            case (state)
            S_IDLE: begin
                if (valid_in)
                    state <= S_PH1;
            end
            S_PH1: begin
                x_max     <= cmp_max;
                round_cnt <= 3'd0;
                sum_num   <= FP_ZERO;
                sum_den   <= FP_ZERO;
                state     <= S_SUB;
            end
            S_SUB: begin
                aa0 <= sel0; ab0 <= {~x_max[31], x_max[30:0]}; av0 <= 1'b1;
                aa1 <= sel1; ab1 <= {~x_max[31], x_max[30:0]}; av1 <= 1'b1;
                aa2 <= sel2; ab2 <= {~x_max[31], x_max[30:0]}; av2 <= 1'b1;
                aa3 <= sel3; ab3 <= {~x_max[31], x_max[30:0]}; av3 <= 1'b1;
                xo_r0 <= sel0; xo_r1 <= sel1; xo_r2 <= sel2; xo_r3 <= sel3;
                state <= S_MUL_P;
            end
            S_MUL_P: begin
                if (ad0) begin
                    ma0 <= ar0; mb0 <= LOG2E; mv0 <= 1'b1;
                    ma1 <= ar1; mb1 <= LOG2E; mv1 <= 1'b1;
                    ma2 <= ar2; mb2 <= LOG2E; mv2 <= 1'b1;
                    ma3 <= ar3; mb3 <= LOG2E; mv3 <= 1'b1;
                    state <= S_SPLIT_F;
                end
            end
            S_SPLIT_F: begin
                if (md0) begin
                    n_r0 <= floor0; n_r1 <= floor1; n_r2 <= floor2; n_r3 <= floor3;
                    aa0 <= mr0; ab0 <= neg_floor0; av0 <= 1'b1;
                    aa1 <= mr1; ab1 <= neg_floor1; av1 <= 1'b1;
                    aa2 <= mr2; ab2 <= neg_floor2; av2 <= 1'b1;
                    aa3 <= mr3; ab3 <= neg_floor3; av3 <= 1'b1;
                    state <= S_MUL_C2F;
                end
            end
            S_MUL_C2F: begin
                if (ad0) begin
                    f_r0 <= ar0; f_r1 <= ar1; f_r2 <= ar2; f_r3 <= ar3;
                    ma0 <= COEFF_C2; mb0 <= ar0; mv0 <= 1'b1;
                    ma1 <= COEFF_C2; mb1 <= ar1; mv1 <= 1'b1;
                    ma2 <= COEFF_C2; mb2 <= ar2; mv2 <= 1'b1;
                    ma3 <= COEFF_C2; mb3 <= ar3; mv3 <= 1'b1;
                    state <= S_ADD_C1;
                end
            end
            S_ADD_C1: begin
                if (md0) begin
                    aa0 <= mr0; ab0 <= COEFF_C1; av0 <= 1'b1;
                    aa1 <= mr1; ab1 <= COEFF_C1; av1 <= 1'b1;
                    aa2 <= mr2; ab2 <= COEFF_C1; av2 <= 1'b1;
                    aa3 <= mr3; ab3 <= COEFF_C1; av3 <= 1'b1;
                    state <= S_MUL_TF;
                end
            end
            S_MUL_TF: begin
                if (ad0) begin
                    ma0 <= ar0; mb0 <= f_r0; mv0 <= 1'b1;
                    ma1 <= ar1; mb1 <= f_r1; mv1 <= 1'b1;
                    ma2 <= ar2; mb2 <= f_r2; mv2 <= 1'b1;
                    ma3 <= ar3; mb3 <= f_r3; mv3 <= 1'b1;
                    state <= S_ADD_C0;
                end
            end
            S_ADD_C0: begin
                if (md0) begin
                    aa0 <= mr0; ab0 <= COEFF_C0; av0 <= 1'b1;
                    aa1 <= mr1; ab1 <= COEFF_C0; av1 <= 1'b1;
                    aa2 <= mr2; ab2 <= COEFF_C0; av2 <= 1'b1;
                    aa3 <= mr3; ab3 <= COEFF_C0; av3 <= 1'b1;
                    state <= S_MUL_EX;
                end
            end
            S_MUL_EX: begin
                if (ad0) begin
                    ev_r0 <= eadj0; ev_r1 <= eadj1; ev_r2 <= eadj2; ev_r3 <= eadj3;
                    ma0 <= eadj0; mb0 <= xo_r0; mv0 <= 1'b1;
                    ma1 <= eadj1; mb1 <= xo_r1; mv1 <= 1'b1;
                    ma2 <= eadj2; mb2 <= xo_r2; mv2 <= 1'b1;
                    ma3 <= eadj3; mb3 <= xo_r3; mv3 <= 1'b1;
                    state <= S_RED4;
                end
            end
            S_RED4: begin
                if (md0) begin
                    aa0 <= mr0;   ab0 <= mr1;   av0 <= 1'b1;
                    aa1 <= mr2;   ab1 <= mr3;   av1 <= 1'b1;
                    aa2 <= ev_r0; ab2 <= ev_r1; av2 <= 1'b1;
                    aa3 <= ev_r2; ab3 <= ev_r3; av3 <= 1'b1;
                    state <= S_RED2;
                end
            end
            S_RED2: begin
                if (ad0) begin
                    aa0 <= ar0; ab0 <= ar1; av0 <= 1'b1;
                    aa2 <= ar2; ab2 <= ar3; av2 <= 1'b1;
                    state <= S_ACC;
                end
            end
            S_ACC: begin
                if (ad0) begin
                    aa0 <= sum_num; ab0 <= ar0; av0 <= 1'b1;
                    aa2 <= sum_den; ab2 <= ar2; av2 <= 1'b1;
                    state <= S_RDONE;
                end
            end
            S_RDONE: begin
                if (ad0) begin
                    sum_num <= ar0;
                    sum_den <= ar2;
                    if (round_cnt == ROUNDS[2:0] - 3'd1) begin
                        da <= ar0;
                        db <= ar2;
                        dv <= 1'b1;
                        state <= S_DIV;
                    end else begin
                        round_cnt <= round_cnt + 3'd1;
                        state     <= S_SUB;
                    end
                end
            end
            S_DIV: begin
                if (dd) begin
                    y_out <= dr;
                    state <= S_OUT;
                end
            end
            S_OUT: begin
                valid_out <= 1'b1;
                state     <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule