module pe_row #(
    parameter L      = 4,
    parameter LIDX_W = 11
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire [L-1:0]       m_load,          // per-column merge-point load strobe
    input  wire [L*32-1:0]    m_x,             // per-column merge x (one per column)
    input  wire [L*32-1:0]    m_y,
    input  wire [L*32-1:0]    m_z,
    input  wire               in_valid,
    input  wire [LIDX_W-1:0]  in_idx,
    input  wire [31:0]        in_x,
    input  wire [31:0]        in_y,
    input  wire [31:0]        in_z,
    input  wire [31:0]        in_dist,
    output wire               out_valid,
    output wire [LIDX_W-1:0]  out_idx,
    output wire [31:0]        out_x,
    output wire [31:0]        out_y,
    output wire [31:0]        out_z,
    output wire [31:0]        out_dist
);
    // Inter-PE bundle nets: index [0]=row input, [L]=row output.
    wire              valid_c [0:L];
    wire [LIDX_W-1:0] idx_c   [0:L];
    wire [31:0]       x_c     [0:L];
    wire [31:0]       y_c     [0:L];
    wire [31:0]       z_c     [0:L];
    wire [31:0]       dist_c  [0:L];

    assign valid_c[0] = in_valid;
    assign idx_c[0]   = in_idx;
    assign x_c[0]     = in_x;
    assign y_c[0]     = in_y;
    assign z_c[0]     = in_z;
    assign dist_c[0]  = in_dist;

    genvar j;
    generate for (j = 0; j < L; j = j + 1) begin : g_col
        pe #(.LIDX_W(LIDX_W)) u_pe (
            .clk      (clk),
            .rst_n    (rst_n),
            .m_load   (m_load[j]),
            .m_x_in   (m_x[j*32 +: 32]),
            .m_y_in   (m_y[j*32 +: 32]),
            .m_z_in   (m_z[j*32 +: 32]),
            .in_valid (valid_c[j]),
            .in_idx   (idx_c[j]),
            .in_x     (x_c[j]),
            .in_y     (y_c[j]),
            .in_z     (z_c[j]),
            .in_dist  (dist_c[j]),
            .out_valid(valid_c[j+1]),
            .out_idx  (idx_c[j+1]),
            .out_x    (x_c[j+1]),
            .out_y    (y_c[j+1]),
            .out_z    (z_c[j+1]),
            .out_dist (dist_c[j+1])
        );
    end endgenerate

    assign out_valid = valid_c[L];
    assign out_idx   = idx_c[L];
    assign out_x     = x_c[L];
    assign out_y     = y_c[L];
    assign out_z     = z_c[L];
    assign out_dist  = dist_c[L];
endmodule
