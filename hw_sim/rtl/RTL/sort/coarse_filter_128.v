module coarse_filter #(
    parameter DATA_WIDTH = 28,
    parameter COMP_NUM = 128
)(
    input   wire                                clk,
    input   wire                                rst_n,
    input   wire                                coarse_valid,
    input   wire                                fps_mode,
    input   wire                                layer_mode,
    input   wire    [DATA_WIDTH*COMP_NUM-1:0]   data_in,
    input   wire    [DATA_WIDTH-1:0]            fps_specul_thres,
    input   wire    [DATA_WIDTH-1:0]            knn_dyna_thres_0,
    input   wire    [DATA_WIDTH-1:0]            knn_dyna_thres_1,
    output  reg     [$clog2(COMP_NUM)-1:0]      knn_sort_num,
    output  reg                                 ready,
    output  reg     [DATA_WIDTH*COMP_NUM-1:0]   fps_sel_out,
    output  reg     [DATA_WIDTH*COMP_NUM-1:0]   knn_sel_out
);

    reg [DATA_WIDTH-1:0] data_reg [0:COMP_NUM-1];
    reg comparing_flag;
    integer i;

    wire [DATA_WIDTH-1:0] knn_out_data [0:COMP_NUM-1];
    wire knn_valid_sel [0:COMP_NUM-1];
    reg [$clog2(COMP_NUM)+1:0] knn_valid_count;

    reg [DATA_WIDTH-1:0] fps_in_a [0:COMP_NUM-1];
    reg [DATA_WIDTH-1:0] fps_in_b [0:COMP_NUM-1];
    reg [DATA_WIDTH-1:0] fps_in_thres [0:COMP_NUM-1];

    wire [DATA_WIDTH-1:0] fps_out_data [0:COMP_NUM-1];
    wire fps_select_valid [0:COMP_NUM-1];

    wire [DATA_WIDTH-1:0] fps_stage1_out [0:63];

    reg [$clog2(COMP_NUM)+1:0] fps_knn_valid_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < COMP_NUM; i = i + 1) begin
                data_reg[i] <= {DATA_WIDTH{1'b0}};
            end
        end else if (coarse_valid) begin
            for (i = 0; i < COMP_NUM; i = i + 1) begin
                data_reg[i] <= data_in[DATA_WIDTH*i +: DATA_WIDTH];
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            comparing_flag <= 1'b0;
        end else if (coarse_valid) begin
            comparing_flag <= 1'b1;
        end else if (ready) begin
            comparing_flag <= 1'b0;
        end
    end

    always @(*) begin
        for (i = 0; i < COMP_NUM; i = i + 1) begin
            fps_in_a[i] = data_reg[i];
            fps_in_b[i] = {DATA_WIDTH{1'b0}};
            fps_in_thres[i] = {DATA_WIDTH{1'b0}};
        end

        if (layer_mode) begin
            if (fps_mode) begin
                for (i = 0; i < 64; i = i + 1) begin
                    fps_in_a[i] = data_reg[i];
                    fps_in_b[i] = data_reg[i + 64];
                    fps_in_thres[i] = {DATA_WIDTH{1'b0}};
                end
                for (i = 64; i < 96; i = i + 1) begin
                    fps_in_a[i] = fps_stage1_out[2*(i-64)];
                    fps_in_b[i] = fps_stage1_out[2*(i-64)+1];
                    fps_in_thres[i] = {DATA_WIDTH{1'b0}};
                end
                for (i = 96; i < COMP_NUM; i = i + 1) begin
                    fps_in_a[i] = {DATA_WIDTH{1'b0}};
                    fps_in_b[i] = {DATA_WIDTH{1'b0}};
                    fps_in_thres[i] = {DATA_WIDTH{1'b0}};
                end
            end else begin
                for (i = 0; i < COMP_NUM; i = i + 1) begin
                    fps_in_a[i] = data_reg[i];
                    fps_in_b[i] = {DATA_WIDTH{1'b0}};
                    fps_in_thres[i] = fps_specul_thres;
                end
            end
        end else begin
            for (i = 0; i < COMP_NUM; i = i + 1) begin
                fps_in_a[i] = data_reg[i];
                fps_in_b[i] = {DATA_WIDTH{1'b0}};
                fps_in_thres[i] = knn_dyna_thres_1;
            end
        end
    end

    genvar j;
    generate
        for (j = 0; j < COMP_NUM; j = j + 1) begin : knn_comp_gen
            knn_comp_unit #(
                .DATA_WIDTH(DATA_WIDTH)
            ) u_knn_comp (
                .in_a(data_reg[j]),
                .in_thres(knn_dyna_thres_0),
                .out_data(knn_out_data[j]),
                .valid_sel(knn_valid_sel[j])
            );
        end
    endgenerate

    always @(*) begin
        knn_valid_count = 0;
        for (i = 0; i < COMP_NUM; i = i + 1) begin
            knn_valid_count = knn_valid_count + knn_valid_sel[i];
        end
    end

    generate
        for (j = 0; j < COMP_NUM; j = j + 1) begin : fps_comp_gen
            fps_comp_unit #(
                .DATA_WIDTH(DATA_WIDTH)
            ) u_fps_comp (
                .in_a(fps_in_a[j]),
                .in_b(fps_in_b[j]),
                .in_thres(fps_in_thres[j]),
                .fps_mode(fps_mode),
                .layer_mode(layer_mode),
                .out_data(fps_out_data[j]),
                .valid_sel(fps_select_valid[j])
            );
        end
    endgenerate

    generate
        for (j = 0; j < 64; j = j + 1) begin : fps_stage1_assign
            assign fps_stage1_out[j] = fps_out_data[j];
        end
    endgenerate

    always @(*) begin
        fps_knn_valid_count = 0;
        if (!layer_mode) begin
            for (i = 0; i < COMP_NUM; i = i + 1) begin
                fps_knn_valid_count = fps_knn_valid_count + fps_select_valid[i];
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fps_sel_out <= {DATA_WIDTH*COMP_NUM{1'b0}};
        end else if (comparing_flag) begin
            if (layer_mode) begin
                if (fps_mode) begin
                    for (i = 0; i < 32; i = i + 1) begin
                        fps_sel_out[DATA_WIDTH*i +: DATA_WIDTH] <= fps_out_data[64 + i];
                    end
                    for (i = 32; i < COMP_NUM; i = i + 1) begin
                        fps_sel_out[DATA_WIDTH*i +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                    end
                end else begin
                    for (i = 0; i < COMP_NUM; i = i + 1) begin
                        fps_sel_out[DATA_WIDTH*i +: DATA_WIDTH] <= fps_out_data[i];
                    end
                end
            end else begin
                for (i = 0; i < COMP_NUM; i = i + 1) begin
                    fps_sel_out[DATA_WIDTH*i +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready <= 1'b0;
            knn_sort_num <= {$clog2(COMP_NUM){1'b0}};
            knn_sel_out <= {DATA_WIDTH*COMP_NUM{1'b0}};
        end else begin
            if (comparing_flag) begin
                if (knn_valid_count >= 24 && knn_valid_count <= 32) begin
                    ready <= 1'b1;
                    knn_sort_num <= knn_valid_count[$clog2(COMP_NUM)-1:0];
                    for (i = 0; i < COMP_NUM; i = i + 1) begin
                        knn_sel_out[DATA_WIDTH*i +: DATA_WIDTH] <= knn_out_data[i];
                    end
                end
                else if (!layer_mode && fps_knn_valid_count >= 24 && fps_knn_valid_count <= 32) begin
                    ready <= 1'b1;
                    knn_sort_num <= fps_knn_valid_count[$clog2(COMP_NUM)-1:0];
                    for (i = 0; i < COMP_NUM; i = i + 1) begin
                        knn_sel_out[DATA_WIDTH*i +: DATA_WIDTH] <= fps_out_data[i];
                    end
                end else begin
                    ready <= 1'b0;
                end
            end else begin
                ready <= 1'b0;
            end
        end
    end

endmodule


module fps_comp_unit #(
    parameter   DATA_WIDTH = 28
)(
    input   wire    [DATA_WIDTH-1:0]    in_a,
    input   wire    [DATA_WIDTH-1:0]    in_b,
    input   wire    [DATA_WIDTH-1:0]    in_thres,
    input   wire                        fps_mode,
    input   wire                        layer_mode,

    output  wire    [DATA_WIDTH-1:0]    out_data,
    output  wire                        valid_sel
);

assign out_data     =   layer_mode ?
                        (fps_mode ?
                        (in_a > in_b ? in_a : in_b)
                        : in_a > in_thres ? in_a : 0)
                        : in_a < in_thres ? in_a : 0;

assign select_valid =   layer_mode ?
                        (fps_mode ?
                        0
                        : in_a > in_thres ? 1 : 0)
                        : in_a < in_thres ? 1 : 0;
endmodule

module knn_comp_unit #(
    parameter   DATA_WIDTH = 28
)(
    input   wire    [DATA_WIDTH-1:0]    in_a,
    input   wire    [DATA_WIDTH-1:0]    in_thres,

    output  wire    [DATA_WIDTH-1:0]    out_data,
    output  wire                        valid_sel
);

assign out_data     =   in_a < in_thres ? in_a : 0;

assign select_valid =   in_a < in_thres ? 1 : 0;
endmodule