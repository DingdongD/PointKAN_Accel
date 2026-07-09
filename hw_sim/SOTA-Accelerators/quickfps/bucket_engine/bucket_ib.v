module bucket_ib (
    input  wire        valid,
    input  wire        first_iter,
    input  wire [31:0] fdist,     // farPoint.dist = dist(farPoint, S_{k-1})
    input  wire [31:0] dlb,       // bucketDist(s_k, box)
    input  wire [31:0] d,         // dist(farPoint, s_k)

    output wire        do_issue,
    output wire        do_defer,
    output wire        do_skip
);
    wire merge_ok    = (fdist[30:0] < d[30:0]);
    wire implicit_ok = (fdist[30:0] < dlb[30:0]);

    assign do_issue = valid & ( first_iter | ~merge_ok );
    assign do_skip  = valid & ~first_iter &  merge_ok &  implicit_ok;
    assign do_defer = valid & ~first_iter &  merge_ok & ~implicit_ok;
endmodule
