// FILE: merge_batch_32.v
`timescale 1ns / 1ps
module bitonic_merge_net #(
    parameter integer N             = 8,
    parameter integer DW            = 32,
    parameter integer IDX_W         = 16,
    parameter [1:0]   MODE_FPS_MAX1 = 2'd2,
    parameter integer STAGE_FIRST   = 0,
    parameter integer STAGE_LAST    = 0
)(
    input  wire [1:0]            mode_sel,
    input  wire [N*DW-1:0]      dist_in,
    input  wire [N*IDX_W-1:0]   idx_in,
    input  wire [N-1:0]         valid_in,
    output wire [N*DW-1:0]      dist_out,
    output wire [N*IDX_W-1:0]   idx_out,
    output wire [N-1:0]         valid_out
);
    localparam integer NUM_S = STAGE_LAST - STAGE_FIRST + 1;
    wire [DW-1:0]    stg_d [0:NUM_S][0:N-1];
    wire [IDX_W-1:0] stg_i [0:NUM_S][0:N-1];
    wire             stg_v [0:NUM_S][0:N-1];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : GEN_IN
            assign stg_d[0][g] = dist_in [g*DW    +: DW];
            assign stg_i[0][g] = idx_in  [g*IDX_W +: IDX_W];
            assign stg_v[0][g] = valid_in[g];
        end
    endgenerate
    genvar sr, blk, ci;
    generate
        for (sr = 0; sr < NUM_S; sr = sr + 1) begin : GEN_STG
            localparam integer SA     = STAGE_FIRST + sr;
            localparam integer STRIDE = N >> (SA + 1);
            localparam integer BLK_SZ = STRIDE << 1;
            for (blk = 0; blk < N / BLK_SZ; blk = blk + 1) begin : GEN_BLK
                for (ci = 0; ci < STRIDE; ci = ci + 1) begin : GEN_CAS
                    localparam integer LO = blk * BLK_SZ + ci;
                    localparam integer HI = LO + STRIDE;
                    wire [DW-1:0]    lo_d, hi_d;
                    wire [IDX_W-1:0] lo_x, hi_x;
                    wire             lo_v, hi_v;
                    assign lo_d = stg_d[sr][LO];
                    assign hi_d = stg_d[sr][HI];
                    assign lo_x = stg_i[sr][LO];
                    assign hi_x = stg_i[sr][HI];
                    assign lo_v = stg_v[sr][LO];
                    assign hi_v = stg_v[sr][HI];
                    wire keep_asc  = (lo_d < hi_d) ||
                                     ((lo_d == hi_d) && (lo_x <= hi_x));
                    wire keep_desc = (lo_d > hi_d) ||
                                     ((lo_d == hi_d) && (lo_x <= hi_x));
                    wire swap_w = (mode_sel == MODE_FPS_MAX1)
                                  ? ~keep_desc : ~keep_asc;
                    assign stg_d[sr+1][LO] = swap_w ? hi_d : lo_d;
                    assign stg_d[sr+1][HI] = swap_w ? lo_d : hi_d;
                    assign stg_i[sr+1][LO] = swap_w ? hi_x : lo_x;
                    assign stg_i[sr+1][HI] = swap_w ? lo_x : hi_x;
                    assign stg_v[sr+1][LO] = swap_w ? hi_v : lo_v;
                    assign stg_v[sr+1][HI] = swap_w ? lo_v : hi_v;
                end
            end
        end
    endgenerate
    generate
        for (g = 0; g < N; g = g + 1) begin : GEN_OUT
            assign dist_out [g*DW    +: DW]    = stg_d[NUM_S][g];
            assign idx_out  [g*IDX_W +: IDX_W] = stg_i[NUM_S][g];
            assign valid_out[g]                 = stg_v[NUM_S][g];
        end
    endgenerate
endmodule
module mergeK_raw #(
    parameter integer DW            = 32,
    parameter integer IDX_W         = 16,
    parameter integer K             = 64,
    parameter [1:0]   MODE_BQ_TOPK  = 2'd0,
    parameter [1:0]   MODE_KNN_TOPK = 2'd1,
    parameter [1:0]   MODE_FPS_MAX1 = 2'd2,
    parameter [DW-1:0] POS_INF      = 32'h7f7fffff,
    parameter [DW-1:0] SENTINEL_MIN = 32'h00000000
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    input  wire [1:0]               mode_sel,
    input  wire [K*DW-1:0]          a_dist_bus,
    input  wire [K*IDX_W-1:0]       a_idx_bus,
    input  wire [K-1:0]             a_valid_bus,
    input  wire [K*DW-1:0]          b_dist_bus,
    input  wire [K*IDX_W-1:0]       b_idx_bus,
    input  wire [K-1:0]             b_valid_bus,
    output reg  [K*DW-1:0]          o_dist_bus,
    output reg  [K*IDX_W-1:0]       o_idx_bus,
    output reg  [K-1:0]             o_valid_bus,
    output reg                      valid_out
);
    localparam integer N2    = 2 * K;
    localparam integer LOG_K = $clog2(K);
    localparam integer CUT   = 3;
    wire [N2*DW-1:0]    bton_d;
    wire [N2*IDX_W-1:0] bton_i;
    wire [N2-1:0]       bton_v;
    genvar g;
    generate
        for (g = 0; g < K; g = g + 1) begin : GEN_A
            assign bton_d[g*DW +: DW]       = a_dist_bus[g*DW +: DW];
            assign bton_i[g*IDX_W +: IDX_W] = a_idx_bus [g*IDX_W +: IDX_W];
            assign bton_v[g]                 = a_valid_bus[g];
        end
        for (g = 0; g < K; g = g + 1) begin : GEN_B_REV
            assign bton_d[(K+g)*DW +: DW]       = b_dist_bus[(K-1-g)*DW +: DW];
            assign bton_i[(K+g)*IDX_W +: IDX_W] = b_idx_bus [(K-1-g)*IDX_W +: IDX_W];
            assign bton_v[K+g]                   = b_valid_bus[K-1-g];
        end
    endgenerate
    wire [N2*DW-1:0]    hc0_d;
    wire [N2*IDX_W-1:0] hc0_i;
    wire [N2-1:0]       hc0_v;
    bitonic_merge_net #(
        .N(N2), .DW(DW), .IDX_W(IDX_W),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .STAGE_FIRST(0), .STAGE_LAST(0)
    ) u_hc0 (
        .mode_sel(mode_sel),
        .dist_in (bton_d), .idx_in (bton_i), .valid_in (bton_v),
        .dist_out(hc0_d),  .idx_out(hc0_i),  .valid_out(hc0_v)
    );
    wire [K*DW-1:0]    top_d = hc0_d[0 +: K*DW];
    wire [K*IDX_W-1:0] top_i = hc0_i[0 +: K*IDX_W];
    wire [K-1:0]       top_v = hc0_v[0 +: K];
    wire [K*DW-1:0]    p1_d;
    wire [K*IDX_W-1:0] p1_i;
    wire [K-1:0]       p1_v;
    bitonic_merge_net #(
        .N(K), .DW(DW), .IDX_W(IDX_W),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .STAGE_FIRST(0), .STAGE_LAST(CUT - 1)
    ) u_bm_lo (
        .mode_sel(mode_sel),
        .dist_in (top_d), .idx_in (top_i), .valid_in (top_v),
        .dist_out(p1_d),  .idx_out(p1_i),  .valid_out(p1_v)
    );
    reg [K*DW-1:0]    p1r_d;
    reg [K*IDX_W-1:0] p1r_i;
    reg [K-1:0]       p1r_v;
    reg               p1r_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p1r_d     <= {K*DW{1'b0}};
            p1r_i     <= {K*IDX_W{1'b0}};
            p1r_v     <= {K{1'b0}};
            p1r_valid <= 1'b0;
        end else begin
            p1r_valid <= valid_in;
            if (valid_in) begin
                p1r_d <= p1_d;
                p1r_i <= p1_i;
                p1r_v <= p1_v;
            end
        end
    end
    wire [K*DW-1:0]    p2_d;
    wire [K*IDX_W-1:0] p2_i;
    wire [K-1:0]       p2_v;
    bitonic_merge_net #(
        .N(K), .DW(DW), .IDX_W(IDX_W),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .STAGE_FIRST(CUT), .STAGE_LAST(LOG_K - 1)
    ) u_bm_hi (
        .mode_sel(mode_sel),
        .dist_in (p1r_d), .idx_in (p1r_i), .valid_in (p1r_v),
        .dist_out(p2_d),  .idx_out(p2_i),  .valid_out(p2_v)
    );
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_dist_bus  <= {K*DW{1'b0}};
            o_idx_bus   <= {K*IDX_W{1'b0}};
            o_valid_bus <= {K{1'b0}};
            valid_out   <= 1'b0;
        end else begin
            valid_out <= p1r_valid;
            if (p1r_valid) begin
                o_dist_bus  <= p2_d;
                o_idx_bus   <= p2_i;
                o_valid_bus <= p2_v;
            end
        end
    end
endmodule
module rq_select_finalize #(
    parameter integer DW                   = 32,
    parameter integer IDX_W                = 16,
    parameter integer TOPK_MAX             = 64,
    parameter [1:0]   MODE_BQ_TOPK         = 2'd0,
    parameter [1:0]   MODE_KNN_TOPK        = 2'd1,
    parameter [1:0]   MODE_FPS_MAX1        = 2'd2,
    parameter integer PAD_BQ_FIRST_VALID   = 1,
    parameter integer PAD_KNN_FIRST_VALID  = 0,
    parameter integer PAD_NO_HIT_WITH_NEG1 = 1,
    parameter [DW-1:0] POS_INF             = 32'h7f7fffff,
    parameter [DW-1:0] SENTINEL_MIN        = 32'h00000000
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           valid_in,
    input  wire [1:0]                     mode_sel,
    input  wire [$clog2(TOPK_MAX+1)-1:0]  req_k,
    input  wire [TOPK_MAX*DW-1:0]         dist_in_bus,
    input  wire [TOPK_MAX*IDX_W-1:0]      idx_in_bus,
    input  wire [TOPK_MAX-1:0]            valid_in_bus,
    output reg  [TOPK_MAX*DW-1:0]         dist_out_bus,
    output reg  [TOPK_MAX*IDX_W-1:0]      idx_out_bus,
    output reg  [$clog2(TOPK_MAX+1)-1:0]  out_count,
    output reg                            hit_out,
    output reg                            done
);
    localparam integer COUNT_W = $clog2(TOPK_MAX + 1);
    integer i;
    reg [COUNT_W-1:0] valid_count_c;
    reg [IDX_W-1:0]   first_idx_c;
    reg [DW-1:0]      first_dist_c;
    reg               do_first_valid_pad;
    reg [COUNT_W-1:0] eff_k_c;
    always @(*) begin
        valid_count_c = {COUNT_W{1'b0}};
        first_idx_c   = {IDX_W{1'b1}};
        first_dist_c  = (mode_sel == MODE_FPS_MAX1) ? SENTINEL_MIN : POS_INF;
        if (valid_in_bus[0]) begin
            first_idx_c  = idx_in_bus [0 +: IDX_W];
            first_dist_c = dist_in_bus[0 +: DW];
        end
        for (i = 0; i < TOPK_MAX; i = i + 1)
            if (valid_in_bus[i])
                valid_count_c = valid_count_c + 1'b1;
        do_first_valid_pad = (mode_sel == MODE_BQ_TOPK)
                             ? PAD_BQ_FIRST_VALID[0]
                             : PAD_KNN_FIRST_VALID[0];
        eff_k_c = (mode_sel == MODE_FPS_MAX1)
                  ? {{(COUNT_W-1){1'b0}}, 1'b1}
                  : req_k;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_out_bus <= {TOPK_MAX*DW{1'b0}};
            idx_out_bus  <= {TOPK_MAX*IDX_W{1'b0}};
            out_count    <= {COUNT_W{1'b0}};
            hit_out      <= 1'b0;
            done         <= 1'b0;
        end else begin
            done <= valid_in;
            if (valid_in) begin
                hit_out <= (valid_count_c != 0);
                if (mode_sel == MODE_FPS_MAX1) begin
                    out_count <= (valid_count_c != 0)
                                 ? {{(COUNT_W-1){1'b0}}, 1'b1}
                                 : {COUNT_W{1'b0}};
                    dist_out_bus[0 +: DW]    <= dist_in_bus[0 +: DW];
                    idx_out_bus [0 +: IDX_W] <= idx_in_bus [0 +: IDX_W];
                    for (i = 1; i < TOPK_MAX; i = i + 1) begin
                        dist_out_bus[i*DW +: DW]      <= SENTINEL_MIN;
                        idx_out_bus [i*IDX_W +: IDX_W] <= {IDX_W{1'b1}};
                    end
                end else begin
                    out_count <= (valid_count_c < eff_k_c)
                                 ? valid_count_c : eff_k_c;
                    if (valid_count_c == 0) begin
                        for (i = 0; i < TOPK_MAX; i = i + 1) begin
                            dist_out_bus[i*DW +: DW] <= POS_INF;
                            if (PAD_NO_HIT_WITH_NEG1)
                                idx_out_bus[i*IDX_W +: IDX_W] <= {IDX_W{1'b1}};
                            else
                                idx_out_bus[i*IDX_W +: IDX_W] <= {IDX_W{1'b0}};
                        end
                    end else begin
                        for (i = 0; i < TOPK_MAX; i = i + 1) begin
                            if (i < eff_k_c) begin
                                if (i < valid_count_c) begin
                                    dist_out_bus[i*DW +: DW]
                                        <= dist_in_bus[i*DW +: DW];
                                    idx_out_bus[i*IDX_W +: IDX_W]
                                        <= idx_in_bus[i*IDX_W +: IDX_W];
                                end else if (do_first_valid_pad) begin
                                    dist_out_bus[i*DW +: DW]      <= first_dist_c;
                                    idx_out_bus [i*IDX_W +: IDX_W] <= first_idx_c;
                                end else begin
                                    dist_out_bus[i*DW +: DW] <= POS_INF;
                                    if (PAD_NO_HIT_WITH_NEG1)
                                        idx_out_bus[i*IDX_W +: IDX_W]
                                            <= {IDX_W{1'b1}};
                                    else
                                        idx_out_bus[i*IDX_W +: IDX_W]
                                            <= {IDX_W{1'b0}};
                                end
                            end else begin
                                dist_out_bus[i*DW +: DW]      <= POS_INF;
                                idx_out_bus [i*IDX_W +: IDX_W] <= {IDX_W{1'b1}};
                            end
                        end
                    end
                end
            end
        end
    end
endmodule
module merge_batch_32 #(
    parameter integer DW                  = 32,
    parameter integer IDX_W               = 16,
    parameter integer K                   = 32,
    parameter [1:0]   MODE_BQ_TOPK        = 2'd0,
    parameter [1:0]   MODE_KNN_TOPK       = 2'd1,
    parameter [1:0]   MODE_FPS_MAX1       = 2'd2,
    parameter integer PAD_BQ_FIRST_VALID  = 1,
    parameter integer PAD_KNN_FIRST_VALID = 0,
    parameter integer PAD_NO_HIT_WITH_NEG1= 1,
    parameter [DW-1:0] POS_INF            = 32'h7f7fffff,
    parameter [DW-1:0] SENTINEL_MIN       = 32'h00000000
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          valid_in,
    input  wire [1:0]                    mode_sel,
    input  wire [$clog2(K+1)-1:0]        req_k,
    input  wire [K*DW-1:0]              prev_dist_bus,
    input  wire [K*IDX_W-1:0]           prev_idx_bus,
    input  wire [K-1:0]                 prev_valid_bus,
    input  wire [K*DW-1:0]              curr_dist_bus,
    input  wire [K*IDX_W-1:0]           curr_idx_bus,
    input  wire [K-1:0]                 curr_valid_bus,
    output wire [K*DW-1:0]              out_dist_bus,
    output wire [K*IDX_W-1:0]           out_idx_bus,
    output wire [$clog2(K+1)-1:0]        out_count,
    output wire                          hit_out,
    output wire                          done,
    output wire [K*DW-1:0]              raw_dist_bus,
    output wire [K*IDX_W-1:0]           raw_idx_bus,
    output wire [K-1:0]                 raw_valid_bus,
    output wire                          raw_valid_out
);
    wire [K*DW-1:0]    mg_dist;
    wire [K*IDX_W-1:0] mg_idx;
    wire [K-1:0]       mg_valid;
    wire               mg_valid_out;
    mergeK_raw #(
        .DW(DW), .IDX_W(IDX_W), .K(K),
        .MODE_BQ_TOPK(MODE_BQ_TOPK), .MODE_KNN_TOPK(MODE_KNN_TOPK),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .POS_INF(POS_INF), .SENTINEL_MIN(SENTINEL_MIN)
    ) u_merge (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .mode_sel(mode_sel),
        .a_dist_bus(prev_dist_bus), .a_idx_bus(prev_idx_bus),
        .a_valid_bus(prev_valid_bus),
        .b_dist_bus(curr_dist_bus), .b_idx_bus(curr_idx_bus),
        .b_valid_bus(curr_valid_bus),
        .o_dist_bus(mg_dist), .o_idx_bus(mg_idx),
        .o_valid_bus(mg_valid), .valid_out(mg_valid_out)
    );
    assign raw_dist_bus  = mg_dist;
    assign raw_idx_bus   = mg_idx;
    assign raw_valid_bus = mg_valid;
    assign raw_valid_out = mg_valid_out;
    rq_select_finalize #(
        .DW(DW), .IDX_W(IDX_W), .TOPK_MAX(K),
        .MODE_BQ_TOPK(MODE_BQ_TOPK), .MODE_KNN_TOPK(MODE_KNN_TOPK),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .PAD_BQ_FIRST_VALID(PAD_BQ_FIRST_VALID),
        .PAD_KNN_FIRST_VALID(PAD_KNN_FIRST_VALID),
        .PAD_NO_HIT_WITH_NEG1(PAD_NO_HIT_WITH_NEG1),
        .POS_INF(POS_INF), .SENTINEL_MIN(SENTINEL_MIN)
    ) u_fin (
        .clk(clk), .rst_n(rst_n),
        .valid_in(mg_valid_out), .mode_sel(mode_sel),
        .req_k(req_k),
        .dist_in_bus(mg_dist), .idx_in_bus(mg_idx),
        .valid_in_bus(mg_valid),
        .dist_out_bus(out_dist_bus), .idx_out_bus(out_idx_bus),
        .out_count(out_count), .hit_out(hit_out), .done(done)
    );
endmodule