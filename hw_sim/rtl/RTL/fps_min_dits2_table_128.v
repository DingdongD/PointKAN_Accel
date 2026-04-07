module fps_min_dits2_table #(
    parameter COMP_NUM = 128,
    parameter DATA_WIDTH = 28
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           update_en,
    input  wire [COMP_NUM*DATA_WIDTH-1:0] dist2_in,
    output wire  [COMP_NUM*DATA_WIDTH-1:0] min_dist2_table
);

wire [DATA_WIDTH-1:0] pack_dist2 [0:COMP_NUM-1];
reg  [DATA_WIDTH-1:0] min_dist [0:COMP_NUM-1];
wire [DATA_WIDTH-1:0] min_dist_next [0:COMP_NUM-1];

genvar i;

generate
    for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_input_pack
        assign pack_dist2[i] = dist2_in[i*DATA_WIDTH +: DATA_WIDTH];
    end
endgenerate

generate
    for (i = 0; i < COMP_NUM; i = i + 1) begin : min_comp
        assign min_dist_next[i] = update_en ?
                                  ((min_dist[i] <= pack_dist2[i]) ?
                                   min_dist[i]
                                   : pack_dist2[i])
                                  : min_dist[i];
    end
endgenerate

generate
    for (i = 0; i < COMP_NUM; i = i + 1) begin : dist_reg
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                min_dist[i] <= {DATA_WIDTH{1'b1}};
            else
                min_dist[i] <= min_dist_next[i];
        end
    end
endgenerate

generate
    for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_output_pack
        assign min_dist2_table[i*DATA_WIDTH +: DATA_WIDTH] = min_dist[i];
    end
endgenerate

endmodule
