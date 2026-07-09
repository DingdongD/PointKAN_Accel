module max_tree4 #(
    parameter LIDX_W = 11
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clr,           // new-bucket reset

    // ---- per-row leaf ports (row 0..3) ----
    input  wire               leaf_valid0,
    input  wire [LIDX_W-1:0]  leaf_idx0,
    input  wire [31:0]        leaf_dist0,

    input  wire               leaf_valid1,
    input  wire [LIDX_W-1:0]  leaf_idx1,
    input  wire [31:0]        leaf_dist1,

    input  wire               leaf_valid2,
    input  wire [LIDX_W-1:0]  leaf_idx2,
    input  wire [31:0]        leaf_dist2,

    input  wire               leaf_valid3,
    input  wire [LIDX_W-1:0]  leaf_idx3,
    input  wire [31:0]        leaf_dist3,

    output reg  [LIDX_W-1:0]  far_idx,
    output reg  [31:0]        far_dist,
    output reg                far_valid
);
    // low-31-bit magnitude of each leaf (squared dist has sign bit 0)
    wire [30:0] mag0 = leaf_dist0[30:0];
    wire [30:0] mag1 = leaf_dist1[30:0];
    wire [30:0] mag2 = leaf_dist2[30:0];
    wire [30:0] mag3 = leaf_dist3[30:0];

    // ---- Stage 1: pairwise-tree argmax over 4 valid-gated leaves ----
    // pair A: leaf0 vs leaf1
    wire              a_pick1 = (leaf_valid1 && (!leaf_valid0 || (mag1 > mag0)));
    wire              a_vld   = leaf_valid0 | leaf_valid1;
    wire [30:0]       a_mag   = a_pick1 ? mag1 : mag0;
    wire [LIDX_W-1:0] a_idx   = a_pick1 ? leaf_idx1 : leaf_idx0;

    // pair B: leaf2 vs leaf3
    wire              b_pick3 = (leaf_valid3 && (!leaf_valid2 || (mag3 > mag2)));
    wire              b_vld   = leaf_valid2 | leaf_valid3;
    wire [30:0]       b_mag   = b_pick3 ? mag3 : mag2;
    wire [LIDX_W-1:0] b_idx   = b_pick3 ? leaf_idx3 : leaf_idx2;

    // final: A vs B
    wire              pick_b  = (b_vld && (!a_vld || (b_mag > a_mag)));
    wire              best_vld = a_vld | b_vld;
    wire [30:0]       best_mag = pick_b ? b_mag : a_mag;
    wire [LIDX_W-1:0] best_idx = pick_b ? b_idx : a_idx;

    // ---- Stage 2: running max across beats / point-blocks ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            far_dist  <= 32'h0;
            far_idx   <= {LIDX_W{1'b0}};
            far_valid <= 1'b0;
        end else if (clr) begin
            far_dist  <= 32'h0;
            far_idx   <= {LIDX_W{1'b0}};
            far_valid <= 1'b0;
        end else if (best_vld && (!far_valid || (best_mag > far_dist[30:0]))) begin
            far_dist  <= {1'b0, best_mag};   // sign bit 0: squared distance >= 0
            far_idx   <= best_idx;
            far_valid <= 1'b1;
        end
    end
endmodule
