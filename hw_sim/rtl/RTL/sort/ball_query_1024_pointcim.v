module ball_query_1024 #(
    parameter DATA_WIDTH = 32,
    parameter COMP_NUM   = 1024
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              valid_in,
    input  wire [COMP_NUM*DATA_WIDTH-1:0]    dist_in,
    input  wire [DATA_WIDTH-1:0]             radius,
    output reg  [COMP_NUM*DATA_WIDTH-1:0]    dist_out,
    output reg  [COMP_NUM-1:0]              mask_out,
    output reg                               valid_out
);

    wire [DATA_WIDTH-1:0] dist_w [0:COMP_NUM-1];
    wire [COMP_NUM-1:0]   cmp_mask;
    wire [COMP_NUM*DATA_WIDTH-1:0] result_packed;

    genvar i;
    generate
        for (i = 0; i < COMP_NUM; i = i + 1) begin : gen_cmp
            assign dist_w[i] = dist_in[i*DATA_WIDTH +: DATA_WIDTH];
            assign cmp_mask[i] = (dist_w[i] <= radius);
            assign result_packed[i*DATA_WIDTH +: DATA_WIDTH] =
                cmp_mask[i] ? dist_w[i] : {DATA_WIDTH{1'b0}};
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_out  <= {COMP_NUM*DATA_WIDTH{1'b0}};
            mask_out  <= {COMP_NUM{1'b0}};
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            dist_out  <= result_packed;
            mask_out  <= cmp_mask;
        end
    end

endmodule