module batch_topk32 #(
    parameter DW = 32
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire [32*DW-1:0]   data_in,
    input  wire                valid_in,
    output wire [32*DW-1:0]   data_out,
    output wire                valid_out
);

    localparam N       = 32;
    localparam LOG2_N  = 5;
    localparam NUM_LVL = LOG2_N * (LOG2_N + 1) / 2;

    reg [N*DW-1:0] lvl   [0:NUM_LVL];
    reg            vpipe  [0:NUM_LVL];

    always @(posedge clk) begin
        lvl[0] <= data_in;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) vpipe[0] <= 1'b0;
        else        vpipe[0] <= valid_in;
    end

    genvar gp, gs, gi;
    generate
        for (gp = 0; gp < LOG2_N; gp = gp + 1) begin : phase
            for (gs = 0; gs <= gp; gs = gs + 1) begin : step

                localparam LVL    = gp * (gp + 1) / 2 + gs;
                localparam STRIDE = 1 << (gp - gs);

                wire [N*DW-1:0] comb;

                for (gi = 0; gi < N; gi = gi + 1) begin : elem
                    localparam PARTNER  = gi ^ STRIDE;
                    localparam IS_LOWER = (gi < PARTNER) ? 1 : 0;
                    localparam DIR_ASC  = (((gi >> (gp + 1)) & 1) == 0) ? 1 : 0;
                    localparam WANT_MIN = (DIR_ASC == IS_LOWER) ? 1 : 0;

                    wire [DW-1:0] self_v = lvl[LVL][gi*DW      +: DW];
                    wire [DW-1:0] peer_v = lvl[LVL][PARTNER*DW +: DW];
                    wire          gt     = (self_v > peer_v);

                    assign comb[gi*DW +: DW] =
                        (WANT_MIN ^ gt) ? self_v : peer_v;
                end

                always @(posedge clk) begin
                    lvl[LVL + 1] <= comb;
                end

                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) vpipe[LVL + 1] <= 1'b0;
                    else        vpipe[LVL + 1] <= vpipe[LVL];
                end

            end
        end
    endgenerate

    assign data_out  = lvl[NUM_LVL];
    assign valid_out = vpipe[NUM_LVL];

endmodule


module bitonic_merge32 #(
    parameter DW = 32,
    parameter K  = 32
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire [K*DW-1:0]   sorted_a,
    input  wire [K*DW-1:0]   sorted_b,
    input  wire               valid_in,
    output wire [K*DW-1:0]   merged_out,
    output wire               valid_out
);

    localparam LOG2_K  = $clog2(K);
    localparam NUM_LVL = 1 + LOG2_K;

    reg [K*DW-1:0] lvl   [0:LOG2_K];
    reg            vpipe  [0:LOG2_K];

    wire [K*DW-1:0] cross_out;

    genvar gi;
    generate
        for (gi = 0; gi < K; gi = gi + 1) begin : cross
            wire [DW-1:0] a_v    = sorted_a[gi*DW +: DW];
            wire [DW-1:0] b_v    = sorted_b[(K - 1 - gi)*DW +: DW];
            wire          take_b = (a_v > b_v);
            assign cross_out[gi*DW +: DW] = take_b ? b_v : a_v;
        end
    endgenerate

    always @(posedge clk) begin
        lvl[0] <= cross_out;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) vpipe[0] <= 1'b0;
        else        vpipe[0] <= valid_in;
    end

    genvar gl, hi;
    generate
        for (gl = 0; gl < LOG2_K; gl = gl + 1) begin : hclean
            localparam STRIDE = K >> (gl + 1);

            wire [K*DW-1:0] hc_out;

            for (hi = 0; hi < K; hi = hi + 1) begin : elem
                if ((hi & STRIDE) == 0) begin : lo
                    localparam P = hi + STRIDE;

                    wire [DW-1:0] self_v = lvl[gl][hi*DW +: DW];
                    wire [DW-1:0] peer_v = lvl[gl][P*DW  +: DW];
                    wire          sw     = (self_v > peer_v);

                    assign hc_out[hi*DW +: DW] = sw ? peer_v : self_v;
                    assign hc_out[P*DW  +: DW] = sw ? self_v : peer_v;
                end
            end

            always @(posedge clk) begin
                lvl[gl + 1] <= hc_out;
            end
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) vpipe[gl + 1] <= 1'b0;
                else        vpipe[gl + 1] <= vpipe[gl];
            end
        end
    endgenerate

    assign merged_out = lvl[LOG2_K];
    assign valid_out  = vpipe[LOG2_K];

endmodule


module batch_topk32 #(
    parameter DW = 32
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire [64*DW-1:0]   data_in,
    input  wire                valid_in,
    output wire [32*DW-1:0]   topk_out,
    output wire                valid_out
);

    localparam K = 32;
    localparam N = 64;

    wire [K*DW-1:0] group_a = data_in[0      +: K*DW];
    wire [K*DW-1:0] group_b = data_in[K*DW   +: K*DW];

    wire [K*DW-1:0] sorted_a, sorted_b;
    wire             va_out,   vb_out;

    bitonic_sort32 #(.DW(DW)) u_sort_a (
        .clk      (clk),
        .rst_n    (rst_n),
        .data_in  (group_a),
        .valid_in (valid_in),
        .data_out (sorted_a),
        .valid_out(va_out)
    );

    bitonic_sort32 #(.DW(DW)) u_sort_b (
        .clk      (clk),
        .rst_n    (rst_n),
        .data_in  (group_b),
        .valid_in (valid_in),
        .data_out (sorted_b),
        .valid_out(vb_out)
    );

    bitonic_merge32 #(.DW(DW), .K(K)) u_merge (
        .clk       (clk),
        .rst_n     (rst_n),
        .sorted_a  (sorted_a),
        .sorted_b  (sorted_b),
        .valid_in  (va_out),
        .merged_out(topk_out),
        .valid_out (valid_out)
    );

endmodule