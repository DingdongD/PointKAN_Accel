module point_engine #(
    parameter L      = 4,    // PEs per row = merge points per pass
    parameter LIDX_W = 11    // bucket-local point index width
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clr,             // new-bucket reset for the max-tree

    // ---- per-column merge-point load (shared across rows) ----
    input  wire [L-1:0]       m_load,
    input  wire [L*32-1:0]    m_x,
    input  wire [L*32-1:0]    m_y,
    input  wire [L*32-1:0]    m_z,

    // ---- per-row bucket-point stream (R = 4 rows) ----
    input  wire [4-1:0]       in_valid,
    input  wire [4*LIDX_W-1:0] in_idx,
    input  wire [4*32-1:0]    in_x,
    input  wire [4*32-1:0]    in_y,
    input  wire [4*32-1:0]    in_z,
    input  wire [4*32-1:0]    in_dist,

    // ---- bucket far point (to FarPoint FIFO) ----
    output wire [LIDX_W-1:0]  far_idx,
    output wire [31:0]        far_dist,
    output wire               far_valid,

    // ---- per-row streamed outputs (updated distances, for write-back / recirc) ----
    // Exposed for the point-engine wrapper: o_row_dist[r] is the point's updated
    // running min (min over the resident merge points), tagged by o_row_idx[r].
    output wire [4-1:0]        o_row_valid,
    output wire [4*LIDX_W-1:0] o_row_idx,
    output wire [4*32-1:0]     o_row_dist
);
    localparam R = 4;        // rows; fixed to the 4-way max-tree

    // Per-row outputs collected for the max-tree.
    wire               row_valid [0:R-1];
    wire [LIDX_W-1:0]  row_idx   [0:R-1];
    wire [31:0]        row_dist  [0:R-1];

    genvar r;
    generate for (r = 0; r < R; r = r + 1) begin : g_row
        pe_row #(.L(L), .LIDX_W(LIDX_W)) u_row (
            .clk      (clk),
            .rst_n    (rst_n),
            .m_load   (m_load),
            .m_x      (m_x),
            .m_y      (m_y),
            .m_z      (m_z),
            .in_valid (in_valid[r]),
            .in_idx   (in_idx [r*LIDX_W +: LIDX_W]),
            .in_x     (in_x   [r*32     +: 32]),
            .in_y     (in_y   [r*32     +: 32]),
            .in_z     (in_z   [r*32     +: 32]),
            .in_dist  (in_dist[r*32     +: 32]),
            .out_valid(row_valid[r]),
            .out_idx  (row_idx[r]),
            .out_x    (),                 // coords not needed downstream (idx selects the point)
            .out_y    (),
            .out_z    (),
            .out_dist (row_dist[r])
        );
    end endgenerate

    // expose per-row outputs for the wrapper (distance write-back / recirculation)
    genvar gr;
    generate for (gr = 0; gr < R; gr = gr + 1) begin : g_expose
        assign o_row_valid[gr]                = row_valid[gr];
        assign o_row_idx[gr*LIDX_W +: LIDX_W] = row_idx[gr];
        assign o_row_dist[gr*32     +: 32]    = row_dist[gr];
    end endgenerate

    // 4-way argmax over the rows -> bucket far point.
    max_tree4 #(.LIDX_W(LIDX_W)) u_max_tree (
        .clk        (clk),
        .rst_n      (rst_n),
        .clr        (clr),
        .leaf_valid0(row_valid[0]), .leaf_idx0(row_idx[0]), .leaf_dist0(row_dist[0]),
        .leaf_valid1(row_valid[1]), .leaf_idx1(row_idx[1]), .leaf_dist1(row_dist[1]),
        .leaf_valid2(row_valid[2]), .leaf_idx2(row_idx[2]), .leaf_dist2(row_dist[2]),
        .leaf_valid3(row_valid[3]), .leaf_idx3(row_idx[3]), .leaf_dist3(row_dist[3]),
        .far_idx    (far_idx),
        .far_dist   (far_dist),
        .far_valid  (far_valid)
    );
endmodule
