// FILE: bq_64_top32.v
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
module sorted_merge #(
    parameter integer HALF_N        = 1,
    parameter integer DW            = 32,
    parameter integer IDX_W         = 16,
    parameter [1:0]   MODE_FPS_MAX1 = 2'd2
)(
    input  wire [1:0]                 mode_sel,
    input  wire [HALF_N*DW-1:0]      a_dist,
    input  wire [HALF_N*IDX_W-1:0]   a_idx,
    input  wire [HALF_N-1:0]         a_valid,
    input  wire [HALF_N*DW-1:0]      b_dist,
    input  wire [HALF_N*IDX_W-1:0]   b_idx,
    input  wire [HALF_N-1:0]         b_valid,
    output wire [2*HALF_N*DW-1:0]    o_dist,
    output wire [2*HALF_N*IDX_W-1:0] o_idx,
    output wire [2*HALF_N-1:0]       o_valid
);
    localparam integer N2     = 2 * HALF_N;
    localparam integer LOG_N2 = $clog2(N2);
    wire [N2*DW-1:0]    bton_d;
    wire [N2*IDX_W-1:0] bton_i;
    wire [N2-1:0]       bton_v;
    genvar g;
    generate
        for (g = 0; g < HALF_N; g = g + 1) begin : GEN_A
            assign bton_d[g*DW +: DW]       = a_dist [g*DW +: DW];
            assign bton_i[g*IDX_W +: IDX_W] = a_idx  [g*IDX_W +: IDX_W];
            assign bton_v[g]                 = a_valid[g];
        end
        for (g = 0; g < HALF_N; g = g + 1) begin : GEN_B_REV
            assign bton_d[(HALF_N+g)*DW +: DW]
                = b_dist [(HALF_N-1-g)*DW +: DW];
            assign bton_i[(HALF_N+g)*IDX_W +: IDX_W]
                = b_idx  [(HALF_N-1-g)*IDX_W +: IDX_W];
            assign bton_v[HALF_N+g]
                = b_valid[HALF_N-1-g];
        end
    endgenerate
    bitonic_merge_net #(
        .N(N2), .DW(DW), .IDX_W(IDX_W),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .STAGE_FIRST(0),
        .STAGE_LAST(LOG_N2 - 1)
    ) u_bm (
        .mode_sel (mode_sel),
        .dist_in  (bton_d),  .idx_in  (bton_i),  .valid_in  (bton_v),
        .dist_out (o_dist),  .idx_out (o_idx),    .valid_out (o_valid)
    );
endmodule
module rq_leaf_stage #(
    parameter integer NUM             = 64,
    parameter integer DW              = 32,
    parameter integer IDX_W           = 16,
    parameter [1:0]   MODE_BQ_TOPK    = 2'd0,
    parameter [1:0]   MODE_KNN_TOPK   = 2'd1,
    parameter [1:0]   MODE_FPS_MAX1   = 2'd2,
    parameter [DW-1:0] POS_INF        = 32'h7f7fffff,
    parameter [DW-1:0] SENTINEL_MIN   = 32'h00000000
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         valid_in,
    input  wire [1:0]                   mode_sel,
    input  wire [NUM*DW-1:0]            dist_in,
    input  wire [NUM*IDX_W-1:0]         idx_in,
    input  wire [DW-1:0]                radius,
    input  wire [NUM-1:0]               eligible_mask_in,
    output reg  [NUM-1:0]               eligible_mask_out,
    output reg  [NUM*DW-1:0]            dist_out,
    output reg  [NUM*IDX_W-1:0]         idx_out,
    output reg  [NUM-1:0]               valid_out_bus,
    output reg                          valid_out
);
    genvar i;
    wire [NUM-1:0]     eligible_w;
    wire [NUM*DW-1:0]  key_w;
    generate
        for (i = 0; i < NUM; i = i + 1) begin : GEN_LEAF
            wire [DW-1:0] dist_i;
            assign dist_i = dist_in[i*DW +: DW];
            assign eligible_w[i] =
                (mode_sel == MODE_BQ_TOPK)  ? (dist_i <= radius)      :
                (mode_sel == MODE_KNN_TOPK) ? eligible_mask_in[i]     :
                (mode_sel == MODE_FPS_MAX1) ? eligible_mask_in[i]     :
                                              1'b0;
            assign key_w[i*DW +: DW] =
                (mode_sel == MODE_FPS_MAX1)
                    ? (eligible_w[i] ? dist_i : SENTINEL_MIN)
                    : (eligible_w[i] ? dist_i : POS_INF);
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            eligible_mask_out <= {NUM{1'b0}};
            dist_out          <= {NUM*DW{1'b0}};
            idx_out           <= {NUM*IDX_W{1'b0}};
            valid_out_bus     <= {NUM{1'b0}};
            valid_out         <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                eligible_mask_out <= eligible_w;
                dist_out          <= key_w;
                idx_out           <= idx_in;
                valid_out_bus     <= eligible_w;
            end
        end
    end
endmodule
module rq_tree_stage #(
    parameter integer DW            = 32,
    parameter integer IDX_W         = 16,
    parameter integer NODE_IN       = 64,
    parameter integer LEN_IN        = 1,
    parameter integer LEN_OUT       = 2,
    parameter [1:0]   MODE_FPS_MAX1 = 2'd2
)(
    input  wire                                   clk,
    input  wire                                   rst_n,
    input  wire                                   valid_in,
    input  wire [1:0]                             mode_sel,
    input  wire [NODE_IN*LEN_IN*DW-1:0]           dist_in_bus,
    input  wire [NODE_IN*LEN_IN*IDX_W-1:0]        idx_in_bus,
    input  wire [NODE_IN*LEN_IN-1:0]              valid_in_bus,
    output reg  [(NODE_IN/2)*LEN_OUT*DW-1:0]      dist_out_bus,
    output reg  [(NODE_IN/2)*LEN_OUT*IDX_W-1:0]   idx_out_bus,
    output reg  [(NODE_IN/2)*LEN_OUT-1:0]         valid_out_bus,
    output reg                                    valid_out
);
    localparam integer NODE_OUT = NODE_IN / 2;
    wire [NODE_OUT*LEN_OUT*DW-1:0]    dist_m;
    wire [NODE_OUT*LEN_OUT*IDX_W-1:0] idx_m;
    wire [NODE_OUT*LEN_OUT-1:0]       valid_m;
    genvar g;
    generate
        for (g = 0; g < NODE_OUT; g = g + 1) begin : GEN_MERGE
            localparam integer AD = (2*g)   * LEN_IN * DW;
            localparam integer BD = (2*g+1) * LEN_IN * DW;
            localparam integer AI = (2*g)   * LEN_IN * IDX_W;
            localparam integer BI = (2*g+1) * LEN_IN * IDX_W;
            localparam integer AV = (2*g)   * LEN_IN;
            localparam integer BV = (2*g+1) * LEN_IN;
            localparam integer OD = g * LEN_OUT * DW;
            localparam integer OI = g * LEN_OUT * IDX_W;
            localparam integer OV = g * LEN_OUT;
            sorted_merge #(
                .HALF_N(LEN_IN),
                .DW(DW), .IDX_W(IDX_W),
                .MODE_FPS_MAX1(MODE_FPS_MAX1)
            ) u_sm (
                .mode_sel (mode_sel),
                .a_dist   (dist_in_bus [AD +: LEN_IN*DW]),
                .a_idx    (idx_in_bus  [AI +: LEN_IN*IDX_W]),
                .a_valid  (valid_in_bus[AV +: LEN_IN]),
                .b_dist   (dist_in_bus [BD +: LEN_IN*DW]),
                .b_idx    (idx_in_bus  [BI +: LEN_IN*IDX_W]),
                .b_valid  (valid_in_bus[BV +: LEN_IN]),
                .o_dist   (dist_m [OD +: LEN_OUT*DW]),
                .o_idx    (idx_m  [OI +: LEN_OUT*IDX_W]),
                .o_valid  (valid_m[OV +: LEN_OUT])
            );
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_out_bus  <= {(NODE_OUT*LEN_OUT*DW){1'b0}};
            idx_out_bus   <= {(NODE_OUT*LEN_OUT*IDX_W){1'b0}};
            valid_out_bus <= {(NODE_OUT*LEN_OUT){1'b0}};
            valid_out     <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                dist_out_bus  <= dist_m;
                idx_out_bus   <= idx_m;
                valid_out_bus <= valid_m;
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
module rq_block_select_32 #(
    parameter integer BLOCK_N             = 32,
    parameter integer DW                  = 32,
    parameter integer IDX_W               = 16,
    parameter integer TOPK_MAX            = 32,
    parameter [1:0]   MODE_BQ_TOPK        = 2'd0,
    parameter [1:0]   MODE_KNN_TOPK       = 2'd1,
    parameter [1:0]   MODE_FPS_MAX1       = 2'd2,
    parameter integer PAD_BQ_FIRST_VALID  = 1,
    parameter integer PAD_KNN_FIRST_VALID = 0,
    parameter integer PAD_NO_HIT_WITH_NEG1= 1,
    parameter [DW-1:0] POS_INF            = 32'h7f7fffff,
    parameter [DW-1:0] SENTINEL_MIN       = 32'h00000000
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           start,
    input  wire [1:0]                     mode_sel,
    input  wire [$clog2(TOPK_MAX+1)-1:0]  req_k,
    input  wire [BLOCK_N*DW-1:0]          dist_in,
    input  wire [BLOCK_N*IDX_W-1:0]       idx_in,
    input  wire [DW-1:0]                  radius,
    input  wire [BLOCK_N-1:0]             eligible_mask_in,
    output wire [TOPK_MAX*DW-1:0]         raw_dist_bus,
    output wire [TOPK_MAX*IDX_W-1:0]      raw_idx_bus,
    output wire [TOPK_MAX-1:0]            raw_valid_bus,
    output wire                           raw_valid_out,
    output wire [TOPK_MAX*DW-1:0]         final_dist_bus,
    output wire [TOPK_MAX*IDX_W-1:0]      final_idx_bus,
    output wire [$clog2(TOPK_MAX+1)-1:0]  out_count,
    output wire                           hit_out,
    output wire                           done
);
    wire [BLOCK_N-1:0]       leaf_eligible_unused;
    wire [BLOCK_N*DW-1:0]    s0_dist;
    wire [BLOCK_N*IDX_W-1:0] s0_idx;
    wire [BLOCK_N-1:0]       s0_valid_bus;
    wire                     s0_valid;
    rq_leaf_stage #(
        .NUM(BLOCK_N), .DW(DW), .IDX_W(IDX_W),
        .MODE_BQ_TOPK(MODE_BQ_TOPK),
        .MODE_KNN_TOPK(MODE_KNN_TOPK),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .POS_INF(POS_INF), .SENTINEL_MIN(SENTINEL_MIN)
    ) u_leaf (
        .clk(clk), .rst_n(rst_n),
        .valid_in(start), .mode_sel(mode_sel),
        .dist_in(dist_in), .idx_in(idx_in),
        .radius(radius), .eligible_mask_in(eligible_mask_in),
        .eligible_mask_out(leaf_eligible_unused),
        .dist_out(s0_dist), .idx_out(s0_idx),
        .valid_out_bus(s0_valid_bus), .valid_out(s0_valid)
    );
    wire [16*2*DW-1:0]    s1_dist;
    wire [16*2*IDX_W-1:0] s1_idx;
    wire [16*2-1:0]       s1_valid_bus;
    wire                  s1_valid;
    rq_tree_stage #(
        .DW(DW), .IDX_W(IDX_W),
        .NODE_IN(32), .LEN_IN(1), .LEN_OUT(2),
        .MODE_FPS_MAX1(MODE_FPS_MAX1)
    ) u_s1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(s0_valid), .mode_sel(mode_sel),
        .dist_in_bus(s0_dist), .idx_in_bus(s0_idx),
        .valid_in_bus(s0_valid_bus),
        .dist_out_bus(s1_dist), .idx_out_bus(s1_idx),
        .valid_out_bus(s1_valid_bus), .valid_out(s1_valid)
    );
    wire [8*4*DW-1:0]    s2_dist;
    wire [8*4*IDX_W-1:0] s2_idx;
    wire [8*4-1:0]       s2_valid_bus;
    wire                 s2_valid;
    rq_tree_stage #(
        .DW(DW), .IDX_W(IDX_W),
        .NODE_IN(16), .LEN_IN(2), .LEN_OUT(4),
        .MODE_FPS_MAX1(MODE_FPS_MAX1)
    ) u_s2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(s1_valid), .mode_sel(mode_sel),
        .dist_in_bus(s1_dist), .idx_in_bus(s1_idx),
        .valid_in_bus(s1_valid_bus),
        .dist_out_bus(s2_dist), .idx_out_bus(s2_idx),
        .valid_out_bus(s2_valid_bus), .valid_out(s2_valid)
    );
    wire [4*8*DW-1:0]    s3_dist;
    wire [4*8*IDX_W-1:0] s3_idx;
    wire [4*8-1:0]       s3_valid_bus;
    wire                 s3_valid;
    rq_tree_stage #(
        .DW(DW), .IDX_W(IDX_W),
        .NODE_IN(8), .LEN_IN(4), .LEN_OUT(8),
        .MODE_FPS_MAX1(MODE_FPS_MAX1)
    ) u_s3 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(s2_valid), .mode_sel(mode_sel),
        .dist_in_bus(s2_dist), .idx_in_bus(s2_idx),
        .valid_in_bus(s2_valid_bus),
        .dist_out_bus(s3_dist), .idx_out_bus(s3_idx),
        .valid_out_bus(s3_valid_bus), .valid_out(s3_valid)
    );
    wire [2*16*DW-1:0]    s4_dist;
    wire [2*16*IDX_W-1:0] s4_idx;
    wire [2*16-1:0]       s4_valid_bus;
    wire                  s4_valid;
    rq_tree_stage #(
        .DW(DW), .IDX_W(IDX_W),
        .NODE_IN(4), .LEN_IN(8), .LEN_OUT(16),
        .MODE_FPS_MAX1(MODE_FPS_MAX1)
    ) u_s4 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(s3_valid), .mode_sel(mode_sel),
        .dist_in_bus(s3_dist), .idx_in_bus(s3_idx),
        .valid_in_bus(s3_valid_bus),
        .dist_out_bus(s4_dist), .idx_out_bus(s4_idx),
        .valid_out_bus(s4_valid_bus), .valid_out(s4_valid)
    );
    wire [1*32*DW-1:0]    s5_dist;
    wire [1*32*IDX_W-1:0] s5_idx;
    wire [1*32-1:0]       s5_valid_bus;
    wire                  s5_valid;
    rq_tree_stage #(
        .DW(DW), .IDX_W(IDX_W),
        .NODE_IN(2), .LEN_IN(16), .LEN_OUT(32),
        .MODE_FPS_MAX1(MODE_FPS_MAX1)
    ) u_s5 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(s4_valid), .mode_sel(mode_sel),
        .dist_in_bus(s4_dist), .idx_in_bus(s4_idx),
        .valid_in_bus(s4_valid_bus),
        .dist_out_bus(s5_dist), .idx_out_bus(s5_idx),
        .valid_out_bus(s5_valid_bus), .valid_out(s5_valid)
    );
    assign raw_dist_bus  = s5_dist;
    assign raw_idx_bus   = s5_idx;
    assign raw_valid_bus = s5_valid_bus;
    assign raw_valid_out = s5_valid;
    rq_select_finalize #(
        .DW(DW), .IDX_W(IDX_W), .TOPK_MAX(TOPK_MAX),
        .MODE_BQ_TOPK(MODE_BQ_TOPK),
        .MODE_KNN_TOPK(MODE_KNN_TOPK),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .PAD_BQ_FIRST_VALID(PAD_BQ_FIRST_VALID),
        .PAD_KNN_FIRST_VALID(PAD_KNN_FIRST_VALID),
        .PAD_NO_HIT_WITH_NEG1(PAD_NO_HIT_WITH_NEG1),
        .POS_INF(POS_INF), .SENTINEL_MIN(SENTINEL_MIN)
    ) u_final (
        .clk(clk), .rst_n(rst_n),
        .valid_in(s5_valid), .mode_sel(mode_sel),
        .req_k(req_k),
        .dist_in_bus(s5_dist), .idx_in_bus(s5_idx),
        .valid_in_bus(s5_valid_bus),
        .dist_out_bus(final_dist_bus), .idx_out_bus(final_idx_bus),
        .out_count(out_count), .hit_out(hit_out), .done(done)
    );
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
module bq_64_top32 #(
    parameter integer DW                  = 32,
    parameter integer IDX_W               = 16,
    parameter integer HALF_N              = 32,
    parameter integer TOPK_MAX            = 32,
    parameter integer K_MERGE             = 32,
    parameter [1:0]   MODE_BQ_TOPK        = 2'd0,
    parameter [1:0]   MODE_KNN_TOPK       = 2'd1,
    parameter [1:0]   MODE_FPS_MAX1       = 2'd2,
    parameter integer PAD_BQ_FIRST_VALID  = 1,
    parameter integer PAD_KNN_FIRST_VALID = 0,
    parameter integer PAD_NO_HIT_WITH_NEG1= 1,
    parameter [DW-1:0] POS_INF            = 32'h7f7fffff,
    parameter [DW-1:0] SENTINEL_MIN       = 32'h00000000
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           start,
    input  wire [1:0]                     mode_sel,
    input  wire [$clog2(TOPK_MAX+1)-1:0]  req_k,
    input  wire [(2*HALF_N)*DW-1:0]       dist_in,
    input  wire [(2*HALF_N)*IDX_W-1:0]    idx_in,
    input  wire [DW-1:0]                  radius,
    input  wire [(2*HALF_N)-1:0]          eligible_mask_in,
    output wire [TOPK_MAX*DW-1:0]         final_dist_bus,
    output wire [TOPK_MAX*IDX_W-1:0]      final_idx_bus,
    output wire [$clog2(TOPK_MAX+1)-1:0]  out_count,
    output wire                           hit_out,
    output wire                           done,
    output wire                           busy_o,
    output wire [2:0]                     sched_phase_o
);
    localparam integer COUNT_W = $clog2(TOPK_MAX + 1);
    localparam [COUNT_W-1:0] REQ_K_BLK = TOPK_MAX;
    localparam [3:0] S_IDLE = 4'd0,
                     S_B0P  = 4'd1,
                     S_B0W  = 4'd2,
                     S_B1P  = 4'd3,
                     S_B1W  = 4'd4,
                     S_MG   = 4'd5,
                     S_MGW  = 4'd6,
                     S_FIN1 = 4'd7,
                     S_FIN2 = 4'd8;
    reg [3:0] state, state_nxt;
    reg                       blk_start;
    reg                       half_sel;
    reg                       merge_valid_in;
    reg                       fin_valid_in;
    reg [TOPK_MAX*DW-1:0]    raw0_dist,  raw1_dist;
    reg [TOPK_MAX*IDX_W-1:0] raw0_idx,   raw1_idx;
    reg [TOPK_MAX-1:0]       raw0_valid, raw1_valid;
    reg                       done_r;
    wire [HALF_N*DW-1:0]    blk_dist_w;
    wire [HALF_N*IDX_W-1:0] blk_idx_w;
    wire [HALF_N-1:0]       blk_mask_w;
    genvar gi;
    generate
        for (gi = 0; gi < HALF_N; gi = gi + 1) begin : GEN_MUX
            assign blk_dist_w[gi*DW +: DW] =
                half_sel ? dist_in[(HALF_N+gi)*DW +: DW]
                         : dist_in[gi*DW +: DW];
            assign blk_idx_w[gi*IDX_W +: IDX_W] =
                half_sel ? idx_in[(HALF_N+gi)*IDX_W +: IDX_W]
                         : idx_in[gi*IDX_W +: IDX_W];
            assign blk_mask_w[gi] =
                half_sel ? eligible_mask_in[HALF_N+gi]
                         : eligible_mask_in[gi];
        end
    endgenerate
    wire [TOPK_MAX*DW-1:0]    blk_raw_dist;
    wire [TOPK_MAX*IDX_W-1:0] blk_raw_idx;
    wire [TOPK_MAX-1:0]       blk_raw_valid;
    wire                      blk_raw_v;
    wire                      blk_done_unused;
    rq_block_select_32 #(
        .BLOCK_N(HALF_N), .DW(DW), .IDX_W(IDX_W), .TOPK_MAX(TOPK_MAX),
        .MODE_BQ_TOPK(MODE_BQ_TOPK), .MODE_KNN_TOPK(MODE_KNN_TOPK),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .PAD_BQ_FIRST_VALID(PAD_BQ_FIRST_VALID),
        .PAD_KNN_FIRST_VALID(PAD_KNN_FIRST_VALID),
        .PAD_NO_HIT_WITH_NEG1(PAD_NO_HIT_WITH_NEG1),
        .POS_INF(POS_INF), .SENTINEL_MIN(SENTINEL_MIN)
    ) u_blk (
        .clk(clk), .rst_n(rst_n),
        .start(blk_start), .mode_sel(mode_sel),
        .req_k(REQ_K_BLK),
        .dist_in(blk_dist_w), .idx_in(blk_idx_w),
        .radius(radius), .eligible_mask_in(blk_mask_w),
        .raw_dist_bus(blk_raw_dist), .raw_idx_bus(blk_raw_idx),
        .raw_valid_bus(blk_raw_valid), .raw_valid_out(blk_raw_v),
        .final_dist_bus(), .final_idx_bus(),
        .out_count(), .hit_out(), .done(blk_done_unused)
    );
    wire [K_MERGE*DW-1:0]    mg_o_dist;
    wire [K_MERGE*IDX_W-1:0] mg_o_idx;
    wire [K_MERGE-1:0]       mg_o_valid;
    wire                     mg_valid_out;
    mergeK_raw #(
        .DW(DW), .IDX_W(IDX_W), .K(K_MERGE),
        .MODE_BQ_TOPK(MODE_BQ_TOPK), .MODE_KNN_TOPK(MODE_KNN_TOPK),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .POS_INF(POS_INF), .SENTINEL_MIN(SENTINEL_MIN)
    ) u_merge (
        .clk(clk), .rst_n(rst_n),
        .valid_in(merge_valid_in), .mode_sel(mode_sel),
        .a_dist_bus(raw0_dist), .a_idx_bus(raw0_idx), .a_valid_bus(raw0_valid),
        .b_dist_bus(raw1_dist), .b_idx_bus(raw1_idx), .b_valid_bus(raw1_valid),
        .o_dist_bus(mg_o_dist), .o_idx_bus(mg_o_idx), .o_valid_bus(mg_o_valid),
        .valid_out(mg_valid_out)
    );
    wire [TOPK_MAX*DW-1:0]    fn_dist;
    wire [TOPK_MAX*IDX_W-1:0] fn_idx;
    wire [COUNT_W-1:0]        fn_count;
    wire                      fn_hit;
    wire                      fn_done;
    rq_select_finalize #(
        .DW(DW), .IDX_W(IDX_W), .TOPK_MAX(TOPK_MAX),
        .MODE_BQ_TOPK(MODE_BQ_TOPK), .MODE_KNN_TOPK(MODE_KNN_TOPK),
        .MODE_FPS_MAX1(MODE_FPS_MAX1),
        .PAD_BQ_FIRST_VALID(PAD_BQ_FIRST_VALID),
        .PAD_KNN_FIRST_VALID(PAD_KNN_FIRST_VALID),
        .PAD_NO_HIT_WITH_NEG1(PAD_NO_HIT_WITH_NEG1),
        .POS_INF(POS_INF), .SENTINEL_MIN(SENTINEL_MIN)
    ) u_fin (
        .clk(clk), .rst_n(rst_n),
        .valid_in(fin_valid_in), .mode_sel(mode_sel),
        .req_k(req_k),
        .dist_in_bus(mg_o_dist), .idx_in_bus(mg_o_idx),
        .valid_in_bus(mg_o_valid),
        .dist_out_bus(fn_dist), .idx_out_bus(fn_idx),
        .out_count(fn_count), .hit_out(fn_hit), .done(fn_done)
    );
    assign final_dist_bus = fn_dist;
    assign final_idx_bus  = fn_idx;
    assign out_count      = fn_count;
    assign hit_out        = fn_hit;
    assign done           = done_r;
    assign busy_o         = (state != S_IDLE);
    assign sched_phase_o  =
        (state == S_B0P || state == S_B0W) ? 3'd1 :
        (state == S_B1P || state == S_B1W) ? 3'd2 :
        (state == S_MG  || state == S_MGW) ? 3'd3 :
        (state == S_FIN1|| state == S_FIN2) ? 3'd4 : 3'd0;
    always @(*) begin
        state_nxt = state;
        case (state)
            S_IDLE: if (start)       state_nxt = S_B0P;
            S_B0P:                   state_nxt = S_B0W;
            S_B0W: if (blk_raw_v)   state_nxt = S_B1P;
            S_B1P:                   state_nxt = S_B1W;
            S_B1W: if (blk_raw_v)   state_nxt = S_MG;
            S_MG:                    state_nxt = S_MGW;
            S_MGW: if (mg_valid_out) state_nxt = S_FIN1;
            S_FIN1:                  state_nxt = S_FIN2;
            S_FIN2:                  state_nxt = S_IDLE;
            default:                 state_nxt = S_IDLE;
        endcase
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= state_nxt;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            blk_start      <= 1'b0;
            half_sel       <= 1'b0;
            merge_valid_in <= 1'b0;
            fin_valid_in   <= 1'b0;
            raw0_dist      <= {TOPK_MAX*DW{1'b0}};
            raw0_idx       <= {TOPK_MAX*IDX_W{1'b0}};
            raw0_valid     <= {TOPK_MAX{1'b0}};
            raw1_dist      <= {TOPK_MAX*DW{1'b0}};
            raw1_idx       <= {TOPK_MAX*IDX_W{1'b0}};
            raw1_valid     <= {TOPK_MAX{1'b0}};
            done_r         <= 1'b0;
        end else begin
            blk_start      <= 1'b0;
            merge_valid_in <= 1'b0;
            fin_valid_in   <= 1'b0;
            done_r         <= 1'b0;
            case (state)
                S_IDLE: if (start) half_sel <= 1'b0;
                S_B0P: blk_start <= 1'b1;
                S_B0W: begin
                    if (blk_raw_v) begin
                        raw0_dist  <= blk_raw_dist;
                        raw0_idx   <= blk_raw_idx;
                        raw0_valid <= blk_raw_valid;
                        half_sel   <= 1'b1;
                    end
                end
                S_B1P: blk_start <= 1'b1;
                S_B1W: begin
                    if (blk_raw_v) begin
                        raw1_dist  <= blk_raw_dist;
                        raw1_idx   <= blk_raw_idx;
                        raw1_valid <= blk_raw_valid;
                    end
                end
                S_MG:   merge_valid_in <= 1'b1;
                S_FIN1: fin_valid_in   <= 1'b1;
                S_FIN2: done_r         <= 1'b1;
                default: ;
            endcase
        end
    end
endmodule