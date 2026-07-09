module bucketdist_skip_top #(
    parameter RID_W = 10
)(
    input  wire               clk,
    input  wire               rst_n,

    // ---- request: one (q, bucket) implicit test per beat ----
    input  wire               req_valid,
    input  wire [RID_W-1:0]   req_rid,        // bucket under test
    input  wire [31:0]        qx, qy, qz,     // latest sampled point s_k (fp32)
    input  wire               first_sample,   // |S|==1 -> never skip

    // ---- shared read address for the external stores ----
    output wire [RID_W-1:0]   rd_rid,         // = req_rid (combinational)

    // ---- bucket AABB read-back (external store) ----
    input  wire [31:0]        box_minx, box_miny, box_minz,
    input  wire [31:0]        box_maxx, box_maxy, box_maxz,

    // ---- far-point-distance read-back (external): dist(farPoint, S_{k-1}) ----
    input  wire [31:0]        treg_max,       // fp32; init +inf (32'h7F800000)

    // ---- decision (registered, pipelined) ----
    output wire               res_valid,
    output wire [RID_W-1:0]   res_rid,
    output wire [31:0]        res_dlb,
    output wire               res_skip,       // 1 => implicit: skip this bucket
    output wire               res_fetch       // 1 => bucket needs processing
);
    assign rd_rid = req_rid;

    wire              pv, ps;
    wire [RID_W-1:0]  pr;
    wire [31:0]       pd;

    fp32_bucketdist_prune_pipe #(.RID_W(RID_W)) u_prune (
        .clk(clk), .rst_n(rst_n),
        .in_valid (req_valid),
        .in_rid   (req_rid),
        .qx(qx), .qy(qy), .qz(qz),
        .minx(box_minx), .miny(box_miny), .minz(box_minz),
        .maxx(box_maxx), .maxy(box_maxy), .maxz(box_maxz),
        .t_region (treg_max),
        .no_skip  (first_sample),
        .out_valid(pv),
        .out_rid  (pr),
        .out_dlb  (pd),
        .out_skip (ps)
    );

    assign res_valid = pv;
    assign res_rid   = pr;
    assign res_dlb   = pd;
    assign res_skip  = ps;
    assign res_fetch = pv & ~ps;
endmodule
