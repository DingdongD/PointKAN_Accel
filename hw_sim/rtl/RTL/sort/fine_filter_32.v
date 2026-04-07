module fine_filter_32 #(
    parameter DATA_WIDTH = 28,
    parameter COMP_NUM = 32
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             valid_in,
    input  wire                             mode,
    input  wire [$clog2(COMP_NUM)-1:0]      comp_in_num,
    input  wire [COMP_NUM*DATA_WIDTH-1:0]   data_in,

    output reg                              valid_out,
    output reg  [24*DATA_WIDTH-1:0]         data_out,
    output reg                              ready
);

    wire [DATA_WIDTH-1:0] data_padded [0:COMP_NUM-1];

    genvar pad_i;
    generate
        for (pad_i = 0; pad_i < COMP_NUM; pad_i = pad_i + 1) begin : gen_inport_pad
            assign data_padded[pad_i] =
                (pad_i < comp_in_num) ?
                    data_in[pad_i*DATA_WIDTH +: DATA_WIDTH] :
                    (mode ? {DATA_WIDTH{1'b1}} : {DATA_WIDTH{1'b0}});
        end
    endgenerate

    wire [COMP_NUM*DATA_WIDTH-1:0] data_padded_packed;
    generate
        for (pad_i = 0; pad_i < COMP_NUM; pad_i = pad_i + 1) begin : gen_inport_pack
            assign data_padded_packed[pad_i*DATA_WIDTH +: DATA_WIDTH] = data_padded[pad_i];
        end
    endgenerate

    wire [COMP_NUM*DATA_WIDTH-1:0] bm2_out;
    wire [COMP_NUM*DATA_WIDTH-1:0] bm4_out;
    wire [COMP_NUM*DATA_WIDTH-1:0] bm8_out;
    wire [COMP_NUM*DATA_WIDTH-1:0] bm16_out;
    wire [16*DATA_WIDTH-1:0] max_16_packed;
    wire [16*DATA_WIDTH-1:0] min_16_packed;

    bm_N #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(COMP_NUM), .MERGE_NUM(2)) u_bm_2(
        .in(data_padded_packed),
        .out(bm2_out)
    );

    bm_N #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(COMP_NUM), .MERGE_NUM(4)) u_bm_4(
        .in(bm2_out),
        .out(bm4_out)
    );

    bm_N #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(COMP_NUM), .MERGE_NUM(8)) u_bm_8(
        .in(bm4_out),
        .out(bm8_out)
    );

    bm_N #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(COMP_NUM), .MERGE_NUM(16)) u_bm_16(
        .in(bm8_out),
        .out(bm16_out)
    );

    bm_32_sub1 #(.DATA_WIDTH(DATA_WIDTH), .COMP_NUM(COMP_NUM), .MERGE_NUM(32)) u_bm32_sub1(
        .in(bm16_out),
        .out_max(max_16_packed),
        .out_min(min_16_packed)
    );

    wire [DATA_WIDTH-1:0] max_16 [0:15];
    wire [DATA_WIDTH-1:0] min_16 [0:15];

    generate
        for (pad_i = 0; pad_i < 16; pad_i = pad_i + 1) begin : gen_pack_stage1
            assign max_16[pad_i] = max_16_packed[pad_i*DATA_WIDTH +: DATA_WIDTH];
            assign min_16[pad_i] = min_16_packed[pad_i*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    reg [DATA_WIDTH-1:0]             max_16_reg [0:15];
    reg [DATA_WIDTH-1:0]             min_16_reg [0:15];
    reg                              valid_stage1;
    reg                              mode_reg;
    reg [$clog2(COMP_NUM)-1:0]      comp_in_num_reg;

    integer reg_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (reg_i = 0; reg_i < 16; reg_i = reg_i + 1) begin
                max_16_reg[reg_i] <= {DATA_WIDTH{1'b0}};
                min_16_reg[reg_i] <= {DATA_WIDTH{1'b0}};
            end
            valid_stage1 <= 1'b0;
            mode_reg <= 1'b0;
            comp_in_num_reg <= {$clog2(COMP_NUM){1'b0}};
        end else begin
            if (valid_in) begin
                for (reg_i = 0; reg_i < 16; reg_i = reg_i + 1) begin
                    max_16_reg[reg_i] <= max_16[reg_i];
                    min_16_reg[reg_i] <= min_16[reg_i];
                end
                valid_stage1 <= 1'b1;
                mode_reg <= mode;
                comp_in_num_reg <= comp_in_num;
            end else begin
                valid_stage1 <= 1'b0;
            end
        end
    end

    wire max_path_enable;
    wire min_path_enable;
    assign max_path_enable = valid_stage1 && (mode_reg == 1'b0);
    assign min_path_enable = valid_stage1 && (mode_reg == 1'b1);

    wire [DATA_WIDTH-1:0] max_level1 [0:7];
    wire [DATA_WIDTH-1:0] max_level2 [0:3];
    wire [DATA_WIDTH-1:0] max_level3 [0:1];
    wire [DATA_WIDTH-1:0] max_final;

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_max1_level1
            assign max_level1[i] = max_path_enable ?
                                   ((max_16_reg[i*2] > max_16_reg[i*2+1]) ?
                                    max_16_reg[i*2] : max_16_reg[i*2+1]) : {DATA_WIDTH{1'b0}};
        end
        for (i = 0; i < 4; i = i + 1) begin : gen_max1_level2
            assign max_level2[i] = max_path_enable ?
                                   ((max_level1[i*2] > max_level1[i*2+1]) ?
                                    max_level1[i*2] : max_level1[i*2+1]) : {DATA_WIDTH{1'b0}};
        end
        for (i = 0; i < 2; i = i + 1) begin : gen_max1_level3
            assign max_level3[i] = max_path_enable ?
                                   ((max_level2[i*2] > max_level2[i*2+1]) ?
                                    max_level2[i*2] : max_level2[i*2+1]) : {DATA_WIDTH{1'b0}};
        end
    endgenerate
    assign max_final = max_path_enable ?
                       ((max_level3[0] > max_level3[1]) ? max_level3[0] : max_level3[1]) :
                       {DATA_WIDTH{1'b0}};

    wire [16*DATA_WIDTH-1:0] max_16_reg_packed;
    generate
        for (i = 0; i < 16; i = i + 1) begin : gen_pack_max16
            assign max_16_reg_packed[i*DATA_WIDTH +: DATA_WIDTH] = max_16_reg[i];
        end
    endgenerate

    wire [8*DATA_WIDTH-1:0] max16_min8_packed;
    wire [8*DATA_WIDTH-1:0] max16_max8_packed;

    bm_16_sub1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .COMP_NUM(16),
        .MERGE_NUM(16)
    ) u_bm16_sub1 (
        .in(max_16_reg_packed),
        .out_min(max16_min8_packed),
        .out_max(max16_max8_packed)
    );

    wire [DATA_WIDTH-1:0] max16_min8 [0:7];
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_unpack_min8
            assign max16_min8[i] = min_path_enable ?
                                   max16_min8_packed[i*DATA_WIDTH +: DATA_WIDTH] :
                                   {DATA_WIDTH{1'b0}};
        end
    endgenerate

    localparam PADDING_VALUE = {DATA_WIDTH{1'b1}};

    wire [DATA_WIDTH-1:0] combined_24 [0:23];
    generate
        for (i = 0; i < 24; i = i + 1) begin : gen_min24_combined
            if (i < 16)
                assign combined_24[i] = min_path_enable ?
                                        (min_16_reg[i] == PADDING_VALUE ? {DATA_WIDTH{1'b0}} : min_16_reg[i]) :
                                        {DATA_WIDTH{1'b0}};
            else
                assign combined_24[i] = min_path_enable ?
                                        (max16_min8[i-16] == PADDING_VALUE ? {DATA_WIDTH{1'b0}} : max16_min8[i-16]) :
                                        {DATA_WIDTH{1'b0}};
        end
    endgenerate

    integer out_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out <= {(24*DATA_WIDTH){1'b0}};
        end else begin
            valid_out <= valid_stage1;

            if (valid_stage1) begin
                if (mode_reg == 1'b0) begin
                    data_out[0*DATA_WIDTH +: DATA_WIDTH] <= max_final;
                    for (out_i = 1; out_i < 24; out_i = out_i + 1) begin
                        data_out[out_i*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                    end
                end else begin
                    for (out_i = 0; out_i < 24; out_i = out_i + 1) begin
                        data_out[out_i*DATA_WIDTH +: DATA_WIDTH] <= combined_24[out_i];
                    end
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready <= 1'b1;
        end else begin
            ready <= !valid_stage1;
        end
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
    wire [DATA_WIDTH-1:0] low = compare ? b : a;
    assign out_a = TREND ? low : great;
    assign out_b = TREND ? great : low;
endmodule

module bm_N #(
    parameter DATA_WIDTH = 28,
    parameter COMP_NUM = 32,
    parameter MERGE_NUM = 2
)(
    input  wire [COMP_NUM*DATA_WIDTH-1:0] in,
    output wire [COMP_NUM*DATA_WIDTH-1:0] out
);
    localparam NUM_STAGES = $clog2(MERGE_NUM);
    wire [DATA_WIDTH-1:0] stage [0:NUM_STAGES][0:COMP_NUM-1];

    genvar stage_idx, i;

    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_bminport
            assign stage[0][i] = in[i*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    generate
        for (stage_idx = 0; stage_idx < NUM_STAGES; stage_idx = stage_idx + 1) begin : gen_stages
            localparam integer substage = MERGE_NUM >> (stage_idx + 1);

            for (i = 0; i < COMP_NUM; i = i + 1) begin : gen
                if ((i & substage) == 0) begin : inst
                    if ((i & MERGE_NUM) == 0) begin : asc
                        cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(1)) cae (
                            .a(stage[stage_idx][i]),
                            .b(stage[stage_idx][i | substage]),
                            .out_a(stage[stage_idx+1][i]),
                            .out_b(stage[stage_idx+1][i | substage])
                        );
                    end else begin : desc
                        cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(0)) cae (
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
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_bmoutport
            assign out[i*DATA_WIDTH +: DATA_WIDTH] = stage[NUM_STAGES][i];
        end
    endgenerate
endmodule


module bm_32_sub1 #(
    parameter DATA_WIDTH = 28,
    parameter COMP_NUM = 32,
    parameter MERGE_NUM = 32
)(
    input  wire [COMP_NUM*DATA_WIDTH-1:0] in,
    output wire [COMP_NUM/2*DATA_WIDTH-1:0] out_max,
    output wire [COMP_NUM/2*DATA_WIDTH-1:0] out_min
);
    localparam substage = MERGE_NUM >> 1;

    wire [DATA_WIDTH-1:0] in_arr [0:COMP_NUM-1];
    wire [DATA_WIDTH-1:0] temp_out [0:COMP_NUM-1];

    genvar i;

    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_bm32inport
            assign in_arr[i] = in[i*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen
            if ((i & substage) == 0) begin : inst
                if ((i & MERGE_NUM) == 0) begin : asc
                    cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(1)) cae (
                        .a(in_arr[i]),
                        .b(in_arr[i | substage]),
                        .out_a(temp_out[i]),
                        .out_b(temp_out[i | substage])
                    );
                end else begin : desc
                    cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(0)) cae (
                        .a(in_arr[i]),
                        .b(in_arr[i | substage]),
                        .out_a(temp_out[i]),
                        .out_b(temp_out[i | substage])
                    );
                end
            end
        end

        for (i = 0; i < 16; i = i + 1) begin : gen_bm32outport
            assign out_min[i*DATA_WIDTH +: DATA_WIDTH] = temp_out[i];
            assign out_max[i*DATA_WIDTH +: DATA_WIDTH] = temp_out[i+16];
        end
    endgenerate
endmodule


module bm_16_sub1 #(
    parameter DATA_WIDTH = 28,
    parameter COMP_NUM = 16,
    parameter MERGE_NUM = 16
)(
    input  wire [COMP_NUM*DATA_WIDTH-1:0] in,
    output wire [COMP_NUM/2*DATA_WIDTH-1:0] out_min,
    output wire [COMP_NUM/2*DATA_WIDTH-1:0] out_max
);
    localparam substage = MERGE_NUM >> 1;

    wire [DATA_WIDTH-1:0] in_arr [0:COMP_NUM-1];
    wire [DATA_WIDTH-1:0] temp_out [0:COMP_NUM-1];

    genvar i;

    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_bm16inport
            assign in_arr[i] = in[i*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen
            if ((i & substage) == 0) begin : inst
                if ((i & MERGE_NUM) == 0) begin : asc
                    cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(1)) cae (
                        .a(in_arr[i]),
                        .b(in_arr[i | substage]),
                        .out_a(temp_out[i]),
                        .out_b(temp_out[i | substage])
                    );
                end else begin : desc
                    cae_unit #(.DATA_WIDTH(DATA_WIDTH), .TREND(0)) cae (
                        .a(in_arr[i]),
                        .b(in_arr[i | substage]),
                        .out_a(temp_out[i]),
                        .out_b(temp_out[i | substage])
                    );
                end
            end
        end

        for (i = 0; i < 8; i = i + 1) begin : gen_bm16outport
            assign out_min[i*DATA_WIDTH +: DATA_WIDTH] = temp_out[i];
            assign out_max[i*DATA_WIDTH +: DATA_WIDTH] = temp_out[i+8];
        end
    endgenerate
endmodule