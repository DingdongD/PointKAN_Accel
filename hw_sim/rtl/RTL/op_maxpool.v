module fp32_fmax (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] out
);

    wire a_neg = a[31];
    wire b_neg = b[31];

    reg a_greater;

    always @(*) begin
        if (a_neg && !b_neg)
            a_greater = 1'b0;
        else if (!a_neg && b_neg)
            a_greater = 1'b1;
        else if (!a_neg)
            a_greater = (a[30:0] > b[30:0]);
        else
            a_greater = (a[30:0] < b[30:0]);
    end

    assign out = a_greater ? a : b;

endmodule

module op_maxpool #(
    parameter P  = 32,
    parameter DW = 32
)(
    input  wire              clk,
    input  wire              rst_n,

    input  wire [1:0]        k_sel,

    input  wire              valid_in,
    input  wire [P*DW-1:0]  data_in,

    output reg               valid_out,
    output reg  [DW-1:0]     data_out
);

    reg [2:0] chunks_max;

    always @(*) begin
        case (k_sel)
            2'd0:    chunks_max = 3'd0;
            2'd1:    chunks_max = 3'd1;
            2'd2:    chunks_max = 3'd3;
            default: chunks_max = 3'd0;
        endcase
    end

    wire [16*DW-1:0] tree_L1;
    wire [ 8*DW-1:0] tree_L2;
    wire [ 4*DW-1:0] tree_L3;
    wire [ 2*DW-1:0] tree_L4;
    wire [   DW-1:0] tree_L5;

    genvar gi;

    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : gen_L1
            fp32_fmax u_fmax (
                .a  (data_in[(2*gi  )*DW +: DW]),
                .b  (data_in[(2*gi+1)*DW +: DW]),
                .out(tree_L1[gi*DW +: DW])
            );
        end
    endgenerate

    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : gen_L2
            fp32_fmax u_fmax (
                .a  (tree_L1[(2*gi  )*DW +: DW]),
                .b  (tree_L1[(2*gi+1)*DW +: DW]),
                .out(tree_L2[gi*DW +: DW])
            );
        end
    endgenerate

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : gen_L3
            fp32_fmax u_fmax (
                .a  (tree_L2[(2*gi  )*DW +: DW]),
                .b  (tree_L2[(2*gi+1)*DW +: DW]),
                .out(tree_L3[gi*DW +: DW])
            );
        end
    endgenerate

    generate
        for (gi = 0; gi < 2; gi = gi + 1) begin : gen_L4
            fp32_fmax u_fmax (
                .a  (tree_L3[(2*gi  )*DW +: DW]),
                .b  (tree_L3[(2*gi+1)*DW +: DW]),
                .out(tree_L4[gi*DW +: DW])
            );
        end
    endgenerate

    fp32_fmax u_tree_L5 (
        .a  (tree_L4[0*DW +: DW]),
        .b  (tree_L4[1*DW +: DW]),
        .out(tree_L5)
    );

    wire [DW-1:0] chunk_max = tree_L5;

    reg  [DW-1:0] running_max;
    wire [DW-1:0] updated_max;

    fp32_fmax u_running (
        .a  (running_max),
        .b  (chunk_max),
        .out(updated_max)
    );

    reg [2:0] chunk_cnt;
    wire      is_first = (chunk_cnt == 3'd0);
    wire      is_last  = (chunk_cnt == chunks_max);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running_max <= {DW{1'b0}};
            chunk_cnt   <= 3'd0;
            valid_out   <= 1'b0;
            data_out    <= {DW{1'b0}};
        end else begin
            valid_out <= 1'b0;

            if (valid_in) begin
                if (is_first) begin
                    running_max <= chunk_max;
                end else begin
                    running_max <= updated_max;
                end

                if (is_last) begin
                    data_out  <= is_first ? chunk_max : updated_max;
                    valid_out <= 1'b1;
                    chunk_cnt <= 3'd0;
                end else begin
                    chunk_cnt <= chunk_cnt + 3'd1;
                end
            end
        end
    end

endmodule