module bscu_lane #(
    parameter W = 8                       // coordinate bit-width (unused here; kept
)(                                        // for a uniform lane/top parameterization)
    input  wire        clk,
    input  wire        rst_n,             // async, active-low; resets C and T only

    // ---- broadcast micro-op (identical across all lanes; SPEC Sec.6) ----
    input  wire [1:0]  c_init_src,        // 0 NONE(use C) / 1 ZERO / 2 ONE / 3 TAG
    input  wire        c_we,              // 1: latch Cout into C (gated by predicate)
    input  wire        a_xor_t,           // 1: XOR the A^B operand with T (abs negate)
    input  wire [1:0]  wb_sel,            // 0 SUM / 1 COUT / 2 DIN / 3 TAG
    input  wire        wb_en,             // 1: attempt a write-back this cycle
    input  wire        wb_pred,           // 1: gate write (and carry) with T
    input  wire        t_we,              // 1: load T this cycle
    input  wire        t_src,             // 0 DIN / 1 SUM

    // ---- per-lane data in ----
    input  wire        sa_and,            // BL  rail = A & B
    input  wire        sa_nor,            // BLB rail = ~(A|B)
    input  wire        din,               // single-row read (copy/hold/mult/sign/mask)

    // ---- per-lane data out ----
    output wire        wb_data,           // bit to write back to the addressed row
    output wire        wb_we,             // effective write enable = wb_en & predicate

    // ---- observation (testbench only) ----
    output wire        c_obs,             // carry flip-flop
    output wire        t_obs              // tag flip-flop
);
    // ---- micro-op field encodings (mirror SPEC Sec.6) ----
    localparam [1:0] CI_NONE = 2'd0, CI_ZERO = 2'd1, CI_ONE = 2'd2, CI_TAG = 2'd3;
    localparam [1:0] WB_SUM  = 2'd0, WB_COUT = 2'd1, WB_DIN = 2'd2, WB_TAG = 2'd3;
    localparam       T_DIN   = 1'b0, T_SUM   = 1'b1;

    // ---- carry / tag state (two independent posedge-clk DFFs) ----
    reg c_q, t_q;
    assign c_obs = c_q;
    assign t_obs = t_q;

    // ---- combinational boolean core + full-adder slice ----
    // INVARIANT (microcode contract): a_xor_t=1 is only a correct conditional-
    // negate ADD when the paired sensed row is constant-0 -- then sa_and=A&0=0
    // and cout=(A^T)&cin = MAJ3(A^T,0,cin). With a non-zero second operand,
    // sa_and uses the UN-negated A and the carry would be wrong. P1b always
    // senses {d-row, R_ZERO} (SPEC Sec.6 conditional-negate recipe).
    wire sa_xor = ~(sa_nor | sa_and);              // A ^ B (Neural Cache: via a NOR gate)
    wire a_op   = sa_xor ^ (t_q & a_xor_t);         // conditional-negate operand
    wire cin    = (c_init_src == CI_NONE) ? c_q    :
                  (c_init_src == CI_ZERO) ? 1'b0   :
                  (c_init_src == CI_ONE)  ? 1'b1   :
                                            t_q;    // CI_TAG
    wire sum    = a_op ^ cin;                        // A^B^Cin
    wire cout   = sa_and | (a_op & cin);            // MAJ3(A,B,Cin)

    // ---- predicated write-back (tag gates the bit-line write driver) ----
    wire pred   = wb_pred ? t_q : 1'b1;
    assign wb_we   = wb_en & pred;
    assign wb_data = (wb_sel == WB_SUM)  ? sum  :
                     (wb_sel == WB_COUT) ? cout :
                     (wb_sel == WB_DIN)  ? din  :
                                           t_q;   // WB_TAG

    // ---- state update: carry write is ALSO predicate-gated so a skipped
    //      predicated add (T=0 lane) is a full no-op (product AND carry hold) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_q <= 1'b0;
            t_q <= 1'b0;
        end else begin
            if (c_we & pred) c_q <= cout;
            if (t_we)        t_q <= (t_src == T_SUM) ? sum : din;
        end
    end
endmodule