module sync_fifo #(
    parameter W     = 32,
    parameter DEPTH = 8,                 // power of two
    parameter AW    = $clog2(DEPTH)
)(
    input  wire         clk,
    input  wire         rst_n,
    // write side
    input  wire         wen,
    input  wire [W-1:0] wdata,
    output wire         full,
    // read side (FWFT)
    input  wire         ren,
    output wire [W-1:0] rdata,
    output wire         empty
);
    reg [W-1:0] mem [0:DEPTH-1];
    reg [AW:0]  wptr, rptr;              // one extra MSB to tell full from empty

    wire [AW-1:0] wa = wptr[AW-1:0];
    wire [AW-1:0] ra = rptr[AW-1:0];

    assign empty = (wptr == rptr);
    assign full  = (wa == ra) && (wptr[AW] != rptr[AW]);
    assign rdata = mem[ra];             // show-ahead

    wire do_wr = wen && !full;
    wire do_rd = ren && !empty;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr <= {(AW+1){1'b0}};
            rptr <= {(AW+1){1'b0}};
        end else begin
            if (do_wr) begin mem[wa] <= wdata; wptr <= wptr + 1'b1; end
            if (do_rd) rptr <= rptr + 1'b1;
        end
    end
endmodule
