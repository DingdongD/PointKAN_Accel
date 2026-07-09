module pe #(
    parameter LIDX_W = 11,   // bucket-local point index width = clog2(MAX_BUCKET_PTS)
    parameter PIPE   = 3     // datapath register depth (= side-channel depth)
)(
    input  wire               clk,
    input  wire               rst_n,

    // merge-point load (PE-resident): one fp32 xyz
    input  wire               m_load,
    input  wire [31:0]        m_x_in,
    input  wire [31:0]        m_y_in,
    input  wire [31:0]        m_z_in,

    // bundle in (from left neighbour)
    input  wire               in_valid,
    input  wire [LIDX_W-1:0]  in_idx,
    input  wire [31:0]        in_x,
    input  wire [31:0]        in_y,
    input  wire [31:0]        in_z,
    input  wire [31:0]        in_dist,    // running min so far for this point

    // bundle out (to right neighbour)
    output reg                out_valid,
    output reg  [LIDX_W-1:0]  out_idx,
    output reg  [31:0]        out_x,
    output reg  [31:0]        out_y,
    output reg  [31:0]        out_z,
    output reg  [31:0]        out_dist
);
    // ---- PE-resident merge point (payload; not reset, gated by m_load) ----
    reg [31:0] mx_q, my_q, mz_q;
    always @(posedge clk) begin
        if (m_load) begin
            mx_q <= m_x_in;
            my_q <= m_y_in;
            mz_q <= m_z_in;
        end
    end

    // ---- stage 1: sub (q - m), registered [depth 1] ----
    wire [31:0] dx, dy, dz;
    fp32_sub u_sub_x (.a(in_x), .b(mx_q), .result(dx));
    fp32_sub u_sub_y (.a(in_y), .b(my_q), .result(dy));
    fp32_sub u_sub_z (.a(in_z), .b(mz_q), .result(dz));
    reg [31:0] dx_q, dy_q, dz_q;
    always @(posedge clk) begin
        dx_q <= dx;
        dy_q <= dy;
        dz_q <= dz;
    end

    // ---- stage 2: square (mul), registered [depth 2] ----
    wire [31:0] sx, sy, sz;
    fp32_mul u_mul_x (.a(dx_q), .b(dx_q), .result(sx));
    fp32_mul u_mul_y (.a(dy_q), .b(dy_q), .result(sy));
    fp32_mul u_mul_z (.a(dz_q), .b(dz_q), .result(sz));
    reg [31:0] sx_q, sy_q, sz_q;
    always @(posedge clk) begin
        sx_q <= sx;
        sy_q <= sy;
        sz_q <= sz;
    end

    // ---- stage 3: add-tree (sx+sy)+sz, REGISTERED into temp_q [depth 3] ----
    // The two back-to-back fp32_add are the PE critical path at 1 GHz; temp_q
    // both isolates that path into its own stage AND keeps the squared distance
    // at the same depth (3) as the side channel. (See header: do not drop it.)
    wire [31:0] sxy, temp;
    fp32_add u_add_xy (.a(sx_q), .b(sy_q), .result(sxy));
    fp32_add u_add_z  (.a(sxy),  .b(sz_q), .result(temp));
    reg [31:0] temp_q;
    always @(posedge clk) temp_q <= temp;

    // ---- side channel: delay bundle by PIPE=3 to align with temp_q ----
    // *_q[k] is the input delayed by (k+1) cycles; *_q[PIPE-1] is depth 3.
    reg [31:0]       x_q    [0:PIPE-1];
    reg [31:0]       y_q    [0:PIPE-1];
    reg [31:0]       z_q    [0:PIPE-1];
    reg [31:0]       dist_q [0:PIPE-1];
    reg [LIDX_W-1:0] idx_q  [0:PIPE-1];
    reg [PIPE-1:0]   valid_q;
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_q <= {PIPE{1'b0}};   // reset only the qualifying valid bits
        end else begin
            x_q[0]     <= in_x;
            y_q[0]     <= in_y;
            z_q[0]     <= in_z;
            dist_q[0]  <= in_dist;
            idx_q[0]   <= in_idx;
            valid_q[0] <= in_valid;
            for (i = 1; i < PIPE; i = i + 1) begin
                x_q[i]     <= x_q[i-1];
                y_q[i]     <= y_q[i-1];
                z_q[i]     <= z_q[i-1];
                dist_q[i]  <= dist_q[i-1];
                idx_q[i]   <= idx_q[i-1];
                valid_q[i] <= valid_q[i-1];
            end
        end
    end

    // ---- output stage: running-min + registered output bundle [depth 4] ----
    // temp_q and dist_q[PIPE-1] are both depth-3 => same cohort. Squared
    // distances are >= 0 (sign bit 0), so compare the low 31 bits as unsigned.
    wire [31:0] dist_aligned = dist_q[PIPE-1];
    wire        temp_smaller = (temp_q[30:0] < dist_aligned[30:0]);
    wire [31:0] dist_min     = temp_smaller ? temp_q : dist_aligned;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
        end else begin
            out_valid <= valid_q[PIPE-1];
            out_idx   <= idx_q[PIPE-1];
            out_x     <= x_q[PIPE-1];
            out_y     <= y_q[PIPE-1];
            out_z     <= z_q[PIPE-1];
            out_dist  <= dist_min;
        end
    end
endmodule
