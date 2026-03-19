

module op_relu (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] x_in,
    input  wire        valid_in,
    output wire [31:0] y_out,
    output wire        valid_out
);

reg [31:0]  y_out_reg;
reg         valid_out_reg;
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        y_out_reg       <= 32'b0;
        valid_out_reg   <= 32'b0;
    end
    else begin
        y_out_reg       <= x_in[31] ? 32'b0 : x_in;
        valid_out_reg   <= valid_in;
    end
end

assign y_out = y_out_reg;
assign valid_out = valid_out_reg;

endmodule
