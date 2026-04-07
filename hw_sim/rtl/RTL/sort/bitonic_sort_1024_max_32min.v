module bitonic_sort_1024_max_32min #(
    parameter DATA_WIDTH = 32,
    parameter COMP_NUM   = 1024
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             valid_in,
    input  wire [COMP_NUM*DATA_WIDTH-1:0]   data_in,

    output reg                              valid_out,
    output reg  [32*DATA_WIDTH-1:0]         data_out,
    output reg                              max_valid_out,
    output reg  [DATA_WIDTH-1:0]            max_out,
    output reg                              ready
);

    localparam W1024 = 1024 * DATA_WIDTH;
    localparam W512  =  512 * DATA_WIDTH;
    localparam W256  =  256 * DATA_WIDTH;
    localparam W128  =  128 * DATA_WIDTH;
    localparam W64   =   64 * DATA_WIDTH;
    localparam W32   =   32 * DATA_WIDTH;

    reg [18:0] valid_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipe <= 19'b0;
        else
            valid_pipe <= {valid_pipe[17:0], valid_in};
    end

    wire [W1024-1:0] p1_w0, p1_comb;
    reg  [W1024-1:0] p1_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(2),  .STRIDE(1)) u_p1_bm2_s0  (.in(data_in), .out(p1_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(4),  .STRIDE(2)) u_p1_bm4_s0  (.in(p1_w0),   .out(p1_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p1_reg <= {W1024{1'b0}};
        else        p1_reg <= p1_comb;
    end

    wire [W1024-1:0] p2_w0, p2_comb;
    reg  [W1024-1:0] p2_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(4),  .STRIDE(1)) u_p2_bm4_s1  (.in(p1_reg), .out(p2_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(8),  .STRIDE(4)) u_p2_bm8_s0  (.in(p2_w0),  .out(p2_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p2_reg <= {W1024{1'b0}};
        else        p2_reg <= p2_comb;
    end

    wire [W1024-1:0] p3_w0, p3_comb;
    reg  [W1024-1:0] p3_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(8),  .STRIDE(2)) u_p3_bm8_s1  (.in(p2_reg), .out(p3_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(8),  .STRIDE(1)) u_p3_bm8_s2  (.in(p3_w0),  .out(p3_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p3_reg <= {W1024{1'b0}};
        else        p3_reg <= p3_comb;
    end

    wire [W1024-1:0] p4_w0, p4_comb;
    reg  [W1024-1:0] p4_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(16), .STRIDE(8)) u_p4_bm16_s0 (.in(p3_reg), .out(p4_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(16), .STRIDE(4)) u_p4_bm16_s1 (.in(p4_w0),  .out(p4_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p4_reg <= {W1024{1'b0}};
        else        p4_reg <= p4_comb;
    end

    wire [W1024-1:0] p5_w0, p5_comb;
    reg  [W1024-1:0] p5_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(16), .STRIDE(2)) u_p5_bm16_s2 (.in(p4_reg), .out(p5_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(16), .STRIDE(1)) u_p5_bm16_s3 (.in(p5_w0),  .out(p5_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p5_reg <= {W1024{1'b0}};
        else        p5_reg <= p5_comb;
    end

    wire [W1024-1:0] p6_w0, p6_comb;
    reg  [W1024-1:0] p6_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(32), .STRIDE(16)) u_p6_bm32_s0 (.in(p5_reg), .out(p6_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(32), .STRIDE(8))  u_p6_bm32_s1 (.in(p6_w0),  .out(p6_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p6_reg <= {W1024{1'b0}};
        else        p6_reg <= p6_comb;
    end

    wire [W1024-1:0] p7_w0, p7_comb;
    reg  [W1024-1:0] p7_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(32), .STRIDE(4)) u_p7_bm32_s2 (.in(p6_reg), .out(p7_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(32), .STRIDE(2)) u_p7_bm32_s3 (.in(p7_w0),  .out(p7_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p7_reg <= {W1024{1'b0}};
        else        p7_reg <= p7_comb;
    end

    wire [W1024-1:0] p8_comb;
    reg  [W1024-1:0] p8_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .MERGE_NUM(32), .STRIDE(1)) u_p8_bm32_s4 (.in(p7_reg), .out(p8_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p8_reg <= {W1024{1'b0}};
        else        p8_reg <= p8_comb;
    end

    wire [W32-1:0] group_max_candidates;
    genvar gk;
    generate
        for (gk = 0; gk < 16; gk = gk + 1) begin : gen_group_max
            assign group_max_candidates[(2*gk)*DATA_WIDTH +: DATA_WIDTH]
                = p8_reg[(64*gk + 31)*DATA_WIDTH +: DATA_WIDTH];
            assign group_max_candidates[(2*gk+1)*DATA_WIDTH +: DATA_WIDTH]
                = p8_reg[(64*gk + 32)*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    wire [16*DATA_WIDTH-1:0] max_p9_w0;
    wire [ 8*DATA_WIDTH-1:0] max_p9_w1;
    wire [ 4*DATA_WIDTH-1:0] max_p9_comb;
    reg  [ 4*DATA_WIDTH-1:0] max_p9_reg;

    max_reduce #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(32)) u_max_p9_s0 (.in(group_max_candidates), .out(max_p9_w0));
    max_reduce #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(16)) u_max_p9_s1 (.in(max_p9_w0),            .out(max_p9_w1));
    max_reduce #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(8))  u_max_p9_s2 (.in(max_p9_w1),            .out(max_p9_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) max_p9_reg <= {(4*DATA_WIDTH){1'b0}};
        else        max_p9_reg <= max_p9_comb;
    end

    wire [2*DATA_WIDTH-1:0] max_p10_w0;
    wire [  DATA_WIDTH-1:0] max_p10_comb;

    max_reduce #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(4)) u_max_p10_s0 (.in(max_p9_reg), .out(max_p10_w0));
    max_reduce #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(2)) u_max_p10_s1 (.in(max_p10_w0), .out(max_p10_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_valid_out <= 1'b0;
            max_out       <= {DATA_WIDTH{1'b0}};
        end else begin
            max_valid_out <= valid_pipe[8];
            if (valid_pipe[8])
                max_out <= max_p10_comb;
        end
    end

    wire [W512-1:0] p9_sub1_out, p9_w0, p9_comb;
    reg  [W512-1:0] p9_reg;

    select_sub1 #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(1024), .GROUP_SIZE(32)) u_p9_sub1 (
        .in(p8_reg), .out_min(p9_sub1_out)
    );
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(512), .MERGE_NUM(32), .STRIDE(16)) u_p9_sort_s0 (.in(p9_sub1_out), .out(p9_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(512), .MERGE_NUM(32), .STRIDE(8))  u_p9_sort_s1 (.in(p9_w0),       .out(p9_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p9_reg <= {W512{1'b0}};
        else        p9_reg <= p9_comb;
    end

    wire [W512-1:0] p10_w0, p10_w1, p10_comb;
    reg  [W512-1:0] p10_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(512), .MERGE_NUM(32), .STRIDE(4)) u_p10_sort_s2 (.in(p9_reg),  .out(p10_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(512), .MERGE_NUM(32), .STRIDE(2)) u_p10_sort_s3 (.in(p10_w0),  .out(p10_w1));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(512), .MERGE_NUM(32), .STRIDE(1)) u_p10_sort_s4 (.in(p10_w1),  .out(p10_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p10_reg <= {W512{1'b0}};
        else        p10_reg <= p10_comb;
    end

    wire [W256-1:0] p11_sub1_out, p11_w0, p11_comb;
    reg  [W256-1:0] p11_reg;

    select_sub1 #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(512), .GROUP_SIZE(32)) u_p11_sub1 (
        .in(p10_reg), .out_min(p11_sub1_out)
    );
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(256), .MERGE_NUM(32), .STRIDE(16)) u_p11_sort_s0 (.in(p11_sub1_out), .out(p11_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(256), .MERGE_NUM(32), .STRIDE(8))  u_p11_sort_s1 (.in(p11_w0),       .out(p11_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p11_reg <= {W256{1'b0}};
        else        p11_reg <= p11_comb;
    end

    wire [W256-1:0] p12_w0, p12_w1, p12_comb;
    reg  [W256-1:0] p12_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(256), .MERGE_NUM(32), .STRIDE(4)) u_p12_sort_s2 (.in(p11_reg), .out(p12_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(256), .MERGE_NUM(32), .STRIDE(2)) u_p12_sort_s3 (.in(p12_w0),  .out(p12_w1));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(256), .MERGE_NUM(32), .STRIDE(1)) u_p12_sort_s4 (.in(p12_w1),  .out(p12_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p12_reg <= {W256{1'b0}};
        else        p12_reg <= p12_comb;
    end

    wire [W128-1:0] p13_sub1_out, p13_w0, p13_comb;
    reg  [W128-1:0] p13_reg;

    select_sub1 #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(256), .GROUP_SIZE(32)) u_p13_sub1 (
        .in(p12_reg), .out_min(p13_sub1_out)
    );
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(128), .MERGE_NUM(32), .STRIDE(16)) u_p13_sort_s0 (.in(p13_sub1_out), .out(p13_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(128), .MERGE_NUM(32), .STRIDE(8))  u_p13_sort_s1 (.in(p13_w0),       .out(p13_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p13_reg <= {W128{1'b0}};
        else        p13_reg <= p13_comb;
    end

    wire [W128-1:0] p14_w0, p14_w1, p14_comb;
    reg  [W128-1:0] p14_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(128), .MERGE_NUM(32), .STRIDE(4)) u_p14_sort_s2 (.in(p13_reg), .out(p14_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(128), .MERGE_NUM(32), .STRIDE(2)) u_p14_sort_s3 (.in(p14_w0),  .out(p14_w1));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(128), .MERGE_NUM(32), .STRIDE(1)) u_p14_sort_s4 (.in(p14_w1),  .out(p14_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p14_reg <= {W128{1'b0}};
        else        p14_reg <= p14_comb;
    end

    wire [W64-1:0] p15_sub1_out, p15_w0, p15_comb;
    reg  [W64-1:0] p15_reg;

    select_sub1 #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(128), .GROUP_SIZE(32)) u_p15_sub1 (
        .in(p14_reg), .out_min(p15_sub1_out)
    );
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(64), .MERGE_NUM(32), .STRIDE(16)) u_p15_sort_s0 (.in(p15_sub1_out), .out(p15_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(64), .MERGE_NUM(32), .STRIDE(8))  u_p15_sort_s1 (.in(p15_w0),       .out(p15_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p15_reg <= {W64{1'b0}};
        else        p15_reg <= p15_comb;
    end

    wire [W64-1:0] p16_w0, p16_w1, p16_comb;
    reg  [W64-1:0] p16_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(64), .MERGE_NUM(32), .STRIDE(4)) u_p16_sort_s2 (.in(p15_reg), .out(p16_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(64), .MERGE_NUM(32), .STRIDE(2)) u_p16_sort_s3 (.in(p16_w0),  .out(p16_w1));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(64), .MERGE_NUM(32), .STRIDE(1)) u_p16_sort_s4 (.in(p16_w1),  .out(p16_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p16_reg <= {W64{1'b0}};
        else        p16_reg <= p16_comb;
    end

    wire [W32-1:0] p17_comb;
    reg  [W32-1:0] p17_reg;

    select_sub1 #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(64), .GROUP_SIZE(32)) u_p17_sub1 (
        .in(p16_reg), .out_min(p17_comb)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p17_reg <= {W32{1'b0}};
        else        p17_reg <= p17_comb;
    end

    wire [W32-1:0] p18_w0, p18_w1, p18_comb;
    reg  [W32-1:0] p18_reg;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(32), .MERGE_NUM(32), .STRIDE(16)) u_p18_sort_s0 (.in(p17_reg), .out(p18_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(32), .MERGE_NUM(32), .STRIDE(8))  u_p18_sort_s1 (.in(p18_w0),  .out(p18_w1));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(32), .MERGE_NUM(32), .STRIDE(4))  u_p18_sort_s2 (.in(p18_w1),  .out(p18_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) p18_reg <= {W32{1'b0}};
        else        p18_reg <= p18_comb;
    end

    wire [W32-1:0] p19_w0, p19_comb;

    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(32), .MERGE_NUM(32), .STRIDE(2)) u_p19_sort_s3 (.in(p18_reg), .out(p19_w0));
    bm_substage #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(32), .MERGE_NUM(32), .STRIDE(1)) u_p19_sort_s4 (.in(p19_w0),  .out(p19_comb));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out  <= {W32{1'b0}};
        end else begin
            valid_out <= valid_pipe[17];
            if (valid_pipe[17])
                data_out <= p19_comb;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            ready <= 1'b1;
        else
            ready <= 1'b1;
    end

endmodule


module cae_unit #(
    parameter DATA_WIDTH = 28,
    parameter TREND = 1
)(
    input  wire [DATA_WIDTH-1:0] a,
    input  wire [DATA_WIDTH-1:0] b,
    output wire [DATA_WIDTH-1:0] out_a,
    output wire [DATA_WIDTH-1:0] out_b
);
    wire compare = a >= b;
    wire [DATA_WIDTH-1:0] great = compare ? a : b;
    wire [DATA_WIDTH-1:0] low   = compare ? b : a;
    assign out_a = TREND ? low   : great;
    assign out_b = TREND ? great : low;
endmodule


module bm_substage #(
    parameter DATA_WIDTH = 28,
    parameter COMP_NUM   = 1024,
    parameter MERGE_NUM  = 2,
    parameter STRIDE     = 1
)(
    input  wire [COMP_NUM*DATA_WIDTH-1:0] in,
    output wire [COMP_NUM*DATA_WIDTH-1:0] out
);
    genvar i;
    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_elem
            if ((i & STRIDE) == 0) begin : gen_cae
                if ((i & MERGE_NUM) == 0) begin : asc
                    cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(1)) u_cae (
                        .a    (in [i         *DATA_WIDTH +: DATA_WIDTH]),
                        .b    (in [(i|STRIDE)*DATA_WIDTH +: DATA_WIDTH]),
                        .out_a(out[i         *DATA_WIDTH +: DATA_WIDTH]),
                        .out_b(out[(i|STRIDE)*DATA_WIDTH +: DATA_WIDTH])
                    );
                end else begin : desc
                    cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(0)) u_cae (
                        .a    (in [i         *DATA_WIDTH +: DATA_WIDTH]),
                        .b    (in [(i|STRIDE)*DATA_WIDTH +: DATA_WIDTH]),
                        .out_a(out[i         *DATA_WIDTH +: DATA_WIDTH]),
                        .out_b(out[(i|STRIDE)*DATA_WIDTH +: DATA_WIDTH])
                    );
                end
            end
        end
    endgenerate
endmodule


module bm_N #(
    parameter DATA_WIDTH = 28,
    parameter COMP_NUM   = 1024,
    parameter MERGE_NUM  = 2
)(
    input  wire [COMP_NUM*DATA_WIDTH-1:0] in,
    output wire [COMP_NUM*DATA_WIDTH-1:0] out
);
    localparam NUM_STAGES = $clog2(MERGE_NUM);
    wire [DATA_WIDTH-1:0] stage [0:NUM_STAGES][0:COMP_NUM-1];

    genvar stage_idx, i;

    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_inport
            assign stage[0][i] = in[i*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    generate
        for (stage_idx = 0; stage_idx < NUM_STAGES; stage_idx = stage_idx + 1) begin : gen_stages
            localparam integer substage = MERGE_NUM >> (stage_idx + 1);

            for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_elem
                if ((i & substage) == 0) begin : gen_cae
                    if ((i & MERGE_NUM) == 0) begin : asc
                        cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(1)) u_cae (
                            .a(stage[stage_idx][i]),
                            .b(stage[stage_idx][i | substage]),
                            .out_a(stage[stage_idx+1][i]),
                            .out_b(stage[stage_idx+1][i | substage])
                        );
                    end else begin : desc
                        cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(0)) u_cae (
                            .a(stage[stage_idx][i]),
                            .b(stage[stage_idx][i | substage]),
                            .out_a(stage[stage_idx+1][i]),
                            .out_b(stage[stage_idx+1][i | substage])
                        );
                    end
                end
            end
        end
    endgenerate

    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_outport
            assign out[i*DATA_WIDTH +: DATA_WIDTH] = stage[NUM_STAGES][i];
        end
    endgenerate
endmodule


module max_reduce #(
    parameter DATA_WIDTH = 28,
    parameter COMP_NUM   = 1024
)(
    input  wire [COMP_NUM*DATA_WIDTH-1:0]       in,
    output wire [(COMP_NUM/2)*DATA_WIDTH-1:0]   out
);
    localparam HALF = COMP_NUM / 2;

    genvar i;
    generate
        for (i = 0; i < HALF; i = i + 1) begin : gen_max
            wire [DATA_WIDTH-1:0] a, b;
            assign a = in[(2*i)  *DATA_WIDTH +: DATA_WIDTH];
            assign b = in[(2*i+1)*DATA_WIDTH +: DATA_WIDTH];
            assign out[i*DATA_WIDTH +: DATA_WIDTH] = (a >= b) ? a : b;
        end
    endgenerate
endmodule


module select_sub1 #(
    parameter DATA_WIDTH  = 28,
    parameter COMP_NUM    = 1024,
    parameter GROUP_SIZE  = 32
)(
    input  wire [COMP_NUM*DATA_WIDTH-1:0]       in,
    output wire [(COMP_NUM/2)*DATA_WIDTH-1:0]   out_min
);
    localparam NUM_PAIRS = COMP_NUM / (2 * GROUP_SIZE);
    localparam PAIR_SIZE = 2 * GROUP_SIZE;

    genvar p, i;
    generate
        for (p = 0; p < NUM_PAIRS; p = p + 1) begin : gen_pair
            for (i = 0; i < GROUP_SIZE; i = i + 1) begin : gen_cae
                wire [DATA_WIDTH-1:0] a, b, min_val, max_val;
                assign a = in[(p*PAIR_SIZE + i)              * DATA_WIDTH +: DATA_WIDTH];
                assign b = in[(p*PAIR_SIZE + i + GROUP_SIZE) * DATA_WIDTH +: DATA_WIDTH];

                cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(1)) u_cae (
                    .a(a),
                    .b(b),
                    .out_a(min_val),
                    .out_b(max_val)
                );

                assign out_min[(p * GROUP_SIZE + i) * DATA_WIDTH +: DATA_WIDTH] = min_val;
            end
        end
    endgenerate
endmodule