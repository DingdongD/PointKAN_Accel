module bucket_cd #(
    parameter META_W = 116
)(
    input  wire               clk,
    input  wire               rst_n,

    input  wire               in_valid,
    input  wire [META_W-1:0]  meta_in,      // opaque: delayed 3 cycles

    input  wire [31:0]        qx, qy, qz,    // latest sampled point s_k
    input  wire [31:0]        minx, miny, minz,   // bucket boundary (AABB)
    input  wire [31:0]        maxx, maxy, maxz,
    input  wire [31:0]        fpx, fpy, fpz, // bucket far-point coordinates

    output reg                out_valid,
    output reg  [META_W-1:0]  out_meta,
    output reg  [31:0]        out_dlb,       // bucketDist(s_k, box)
    output reg  [31:0]        out_d          // dist(farPoint, s_k)
);
    // ---------- Stage 1 ----------
    wire [31:0] gx, gy, gz;                  // box gaps
    fp32_box_gap u_gx (.q(qx), .bmin(minx), .bmax(maxx), .gap(gx));
    fp32_box_gap u_gy (.q(qy), .bmin(miny), .bmax(maxy), .gap(gy));
    fp32_box_gap u_gz (.q(qz), .bmin(minz), .bmax(maxz), .gap(gz));
    wire [31:0] ex, ey, ez;                  // merge diffs (s_k - farPoint)
    fp32_sub u_ex (.a(qx), .b(fpx), .result(ex));
    fp32_sub u_ey (.a(qy), .b(fpy), .result(ey));
    fp32_sub u_ez (.a(qz), .b(fpz), .result(ez));

    reg [31:0] gx_q, gy_q, gz_q, ex_q, ey_q, ez_q;
    reg [META_W-1:0] m1;
    reg s1_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) s1_valid <= 1'b0;
        else begin
            s1_valid <= in_valid;  m1 <= meta_in;
            gx_q<=gx; gy_q<=gy; gz_q<=gz;
            ex_q<=ex; ey_q<=ey; ez_q<=ez;
        end
    end

    // ---------- Stage 2 ----------
    wire [31:0] sbx, sby, sbz, smx, smy, smz;
    fp32_mul u_sbx (.a(gx_q), .b(gx_q), .result(sbx));
    fp32_mul u_sby (.a(gy_q), .b(gy_q), .result(sby));
    fp32_mul u_sbz (.a(gz_q), .b(gz_q), .result(sbz));
    fp32_mul u_smx (.a(ex_q), .b(ex_q), .result(smx));
    fp32_mul u_smy (.a(ey_q), .b(ey_q), .result(smy));
    fp32_mul u_smz (.a(ez_q), .b(ez_q), .result(smz));

    reg [31:0] sbx_q, sby_q, sbz_q, smx_q, smy_q, smz_q;
    reg [META_W-1:0] m2;
    reg s2_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) s2_valid <= 1'b0;
        else begin
            s2_valid <= s1_valid;  m2 <= m1;
            sbx_q<=sbx; sby_q<=sby; sbz_q<=sbz;
            smx_q<=smx; smy_q<=smy; smz_q<=smz;
        end
    end

    // ---------- Stage 3 : sum ----------
    wire [31:0] sbxy, dlb_c, smxy, d_c;
    fp32_add u_ba1 (.a(sbx_q), .b(sby_q), .result(sbxy));
    fp32_add u_ba2 (.a(sbxy),  .b(sbz_q), .result(dlb_c));
    fp32_add u_ma1 (.a(smx_q), .b(smy_q), .result(smxy));
    fp32_add u_ma2 (.a(smxy),  .b(smz_q), .result(d_c));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) out_valid <= 1'b0;
        else begin
            out_valid <= s2_valid;
            out_meta  <= m2;
            out_dlb   <= dlb_c;
            out_d     <= d_c;
        end
    end
endmodule
