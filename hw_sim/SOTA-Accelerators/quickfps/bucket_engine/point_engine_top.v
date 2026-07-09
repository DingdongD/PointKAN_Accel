module point_engine_top #(
    parameter L      = 4,
    parameter R      = 4,
    parameter LIDX_W = 11,
    parameter BIDX_W = 9,
    parameter PIDX_W = 18,
    parameter NUMP_W = 16,
    parameter ADDR_W = 32,
    parameter MCNT_W = 8,
    parameter BF_W   = BIDX_W + ADDR_W + NUMP_W + 96 + MCNT_W,
    parameter FP_W   = BIDX_W + 96 + 32 + PIDX_W
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               bkt_empty,
    input  wire [BF_W-1:0]    bkt_rdata,
    output wire               bkt_ren,
    input  wire               fp_full,
    output wire               fp_wen,
    output wire [FP_W-1:0]    fp_wdata,
    output wire [BIDX_W-1:0]  mb_bid,
    output wire [7:0]         mb_k,
    input  wire [95:0]        mb_rdata,
    output wire [PIDX_W-1:0]  co_addr,
    input  wire [R*96-1:0]    co_rdata,
    output wire [PIDX_W-1:0]  di_raddr,
    input  wire [R*32-1:0]    di_rdata,
    output wire [R-1:0]       di_wstrb,
    output wire [PIDX_W-1:0]  di_waddr,
    output wire [R*32-1:0]    di_wdata
);
    localparam [31:0] INF32 = 32'h7F80_0000;
    localparam OFF_SX = BIDX_W + ADDR_W + NUMP_W;

    wire [BIDX_W-1:0] bf_bid  = bkt_rdata[0 +: BIDX_W];
    wire [ADDR_W-1:0] bf_ptr  = bkt_rdata[BIDX_W +: ADDR_W];
    wire [NUMP_W-1:0] bf_nump = bkt_rdata[BIDX_W+ADDR_W +: NUMP_W];
    wire [31:0]       bf_sx   = bkt_rdata[OFF_SX    +: 32];
    wire [31:0]       bf_sy   = bkt_rdata[OFF_SX+32 +: 32];
    wire [31:0]       bf_sz   = bkt_rdata[OFF_SX+64 +: 32];
    wire [MCNT_W-1:0] bf_mcnt = bkt_rdata[OFF_SX+96 +: MCNT_W];

    localparam T_IDLE=0, T_LDMERGE=1, T_STREAM=2, T_COLLECT=3, T_PUSH=4;
    reg [2:0]         st;
    reg [BIDX_W-1:0]  r_bid;
    reg [PIDX_W-1:0]  r_base;
    reg [NUMP_W-1:0]  r_nump;
    reg [31:0]        r_sx,r_sy,r_sz;
    reg [7:0]         r_k, r_npass, pass;
    reg [NUMP_W-1:0]  r_nbatch;
    reg [3:0]         c;
    reg [NUMP_W-1:0]  in_batch, cap_batch;
    reg [1:0]         collect_wait;
    reg [FP_W-1:0]    fp_hold;

    wire last_pass = (pass == r_npass - 8'd1);
    assign bkt_ren = (st==T_IDLE) && !bkt_empty;

    wire [L*32-1:0]     m_x, m_y, m_z;
    wire [R*LIDX_W-1:0] in_idx;
    wire [R*32-1:0]     in_x, in_y, in_z, in_dist;
    wire [LIDX_W-1:0]   far_idx;
    wire [31:0]         far_dist;
    wire                far_valid;
    wire [R-1:0]        o_row_valid;
    wire [R*LIDX_W-1:0] o_row_idx;
    wire [R*32-1:0]     o_row_dist;

    reg  [7:0]  r_mcnt_ext;
    wire [7:0]  refi = pass*L + c;
    wire [95:0] merge_pt = (refi <  r_mcnt_ext) ? mb_rdata :
                           (refi == r_mcnt_ext) ? {r_sz,r_sy,r_sx} :
                                                  {INF32,INF32,INF32};
    assign mb_bid  = r_bid;
    assign mb_k    = refi;
    assign m_x     = {L{merge_pt[31:0]}};
    assign m_y     = {L{merge_pt[63:32]}};
    assign m_z     = {L{merge_pt[95:64]}};
    wire clr           = (st==T_LDMERGE) && last_pass && (c==4'd0);
    wire [L-1:0] m_load= (st==T_LDMERGE) ? ({{(L-1){1'b0}},1'b1} << c) : {L{1'b0}};

    wire streaming = (st==T_STREAM) && (in_batch < r_nbatch);
    assign co_addr  = (st==T_COLLECT) ? (r_base + {far_idx[LIDX_W-1:2],2'b00})
                                      : (r_base + in_batch*R);
    assign di_raddr = r_base + in_batch*R;

    genvar g;
    generate for (g=0; g<R; g=g+1) begin: g_lane
        assign in_idx [g*LIDX_W +: LIDX_W] = (in_batch*R + g);
        assign in_x   [g*32 +: 32] = co_rdata[g*96      +: 32];
        assign in_y   [g*32 +: 32] = co_rdata[g*96 + 32 +: 32];
        assign in_z   [g*32 +: 32] = co_rdata[g*96 + 64 +: 32];
        assign in_dist[g*32 +: 32] = di_rdata[g*32 +: 32];
    end endgenerate
    wire [R-1:0] in_valid = {R{streaming}};

    assign di_wstrb = (st==T_STREAM) ? o_row_valid : {R{1'b0}};
    assign di_waddr = r_base + o_row_idx[LIDX_W-1:0];
    assign di_wdata = o_row_dist;

    wire [1:0]  far_off = far_idx[1:0];
    wire [95:0] far_co  = co_rdata[far_off*96 +: 96];

    assign fp_wen   = (st==T_PUSH) && !fp_full;
    assign fp_wdata = fp_hold;

    point_engine #(.L(L), .LIDX_W(LIDX_W)) u_pe (
        .clk(clk), .rst_n(rst_n), .clr(clr),
        .m_load(m_load), .m_x(m_x), .m_y(m_y), .m_z(m_z),
        .in_valid(in_valid), .in_idx(in_idx),
        .in_x(in_x), .in_y(in_y), .in_z(in_z), .in_dist(in_dist),
        .far_idx(far_idx), .far_dist(far_dist), .far_valid(far_valid),
        .o_row_valid(o_row_valid), .o_row_idx(o_row_idx), .o_row_dist(o_row_dist)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) st<=T_IDLE;
        else begin
            case (st)
            T_IDLE: if (!bkt_empty) begin
                r_bid<=bf_bid; r_base<=bf_ptr[PIDX_W-1:0]; r_nump<=bf_nump;
                r_sx<=bf_sx; r_sy<=bf_sy; r_sz<=bf_sz;
                r_mcnt_ext<={{(8-MCNT_W){1'b0}}, bf_mcnt};
                r_k<= {{(8-MCNT_W){1'b0}}, bf_mcnt} + 8'd1;
                r_npass<= (({{(8-MCNT_W){1'b0}}, bf_mcnt} + 8'd1) + L - 1) / L;
                r_nbatch<= bf_nump / R;
                pass<=0; c<=4'd0; st<=T_LDMERGE;
            end
            T_LDMERGE: if (c==L-1) begin in_batch<=0; cap_batch<=0; st<=T_STREAM; end
                       else c<=c+4'd1;
            T_STREAM: begin
                if (in_batch < r_nbatch) in_batch <= in_batch + 1'b1;
                if (o_row_valid[0])      cap_batch <= cap_batch + 1'b1;
                if ((cap_batch + (o_row_valid[0]?1'b1:1'b0)) >= r_nbatch) begin
                    if (last_pass) begin collect_wait<=2'd2; st<=T_COLLECT; end
                    else begin pass<=pass+8'd1; c<=4'd0; st<=T_LDMERGE; end
                end
            end
            T_COLLECT: if (collect_wait!=2'd0) collect_wait<=collect_wait-1'b1;
                       else begin fp_hold <= {r_bid, r_base + far_idx, far_dist, far_co}; st<=T_PUSH; end
            T_PUSH: if (!fp_full) st<=T_IDLE;
            default: st<=T_IDLE;
            endcase
        end
    end
endmodule
