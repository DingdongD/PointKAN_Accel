module apu_bscu #(
    parameter W     = 8,                  // coordinate bit-width (SPEC [D-W])
    parameter NLANE = 256                 // bit-lines = per-cycle parallelism ([D-LANE])
)(
    input  wire              clk,
    input  wire              rst_n,

    // ---- broadcast micro-op (scalar; same to every lane) ----
    input  wire [1:0]        c_init_src,
    input  wire              c_we,
    input  wire              a_xor_t,
    input  wire [1:0]        wb_sel,
    input  wire              wb_en,
    input  wire              wb_pred,
    input  wire              t_we,
    input  wire              t_src,

    // ---- per-lane data buses ----
    input  wire [NLANE-1:0]  sa_and,
    input  wire [NLANE-1:0]  sa_nor,
    input  wire [NLANE-1:0]  din,
    output wire [NLANE-1:0]  wb_data,
    output wire [NLANE-1:0]  wb_we,

    // ---- observation (testbench only) ----
    output wire [NLANE-1:0]  c_obs,
    output wire [NLANE-1:0]  t_obs
);
    genvar g;
    generate
        for (g = 0; g < NLANE; g = g + 1) begin : g_lane
            bscu_lane #(.W(W)) u_lane (
                .clk        (clk),
                .rst_n      (rst_n),
                .c_init_src (c_init_src),
                .c_we       (c_we),
                .a_xor_t    (a_xor_t),
                .wb_sel     (wb_sel),
                .wb_en      (wb_en),
                .wb_pred    (wb_pred),
                .t_we       (t_we),
                .t_src      (t_src),
                .sa_and     (sa_and[g]),
                .sa_nor     (sa_nor[g]),
                .din        (din[g]),
                .wb_data    (wb_data[g]),
                .wb_we      (wb_we[g]),
                .c_obs      (c_obs[g]),
                .t_obs      (t_obs[g])
            );
        end
    endgenerate
endmodule