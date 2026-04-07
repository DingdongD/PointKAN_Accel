module cim_peri_shift_add_2b (
    input   wire            clk,
    input   wire            rst_n,
    input   wire    [11:0]  x_in_bit0,
    input   wire    [11:0]  y_in_bit0,
    input   wire    [11:0]  z_in_bit0,
    input   wire    [11:0]  x_in_bit1,
    input   wire    [11:0]  y_in_bit1,
    input   wire    [11:0]  z_in_bit1,
    input   wire    [25:0]  key_self2,
    input   wire    [25:0]  query_self2,
    input   wire            valid,
    output  reg     [27:0]  dist2,
    output  reg             ready     
);

reg [2:0]   counts;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        counts      <= 3'b0;
        ready       <= 1'b0;
    end
    else if(valid) begin
        if(counts < 6)
            counts <= counts + 1;
        else if(counts == 6)
            ready <= 1'b1;
    end
    else if(!valid) begin
        counts      <= 3'b0;
        ready       <= 1'b0;
    end
end

wire    [11:0]  x_comple0;
wire    [11:0]  y_comple0;
wire    [11:0]  z_comple0;

assign  x_comple0 =  valid ? 
                    x_in_bit0 ? 
                    ~x_in_bit0 + 1 
                    : 12'b0
                    : 12'b0;
assign  y_comple0 =  valid ? 
                    y_in_bit0 ? 
                    ~y_in_bit0 + 1 
                    : 12'b0
                    : 12'b0;
assign  z_comple0 =  valid ?
                    z_in_bit0 ? 
                    ~z_in_bit0 + 1 
                    : 12'b0
                    : 12'b0;

wire    [11:0]  x_comple1;
wire    [11:0]  y_comple1;
wire    [11:0]  z_comple1;

assign  x_comple1 =  valid ? 
                    x_in_bit1 ? 
                    ~x_in_bit1 + 1 
                    : 12'b0
                    : 12'b0;
assign  y_comple1 =  valid ? 
                    y_in_bit1 ? 
                    ~y_in_bit1 + 1 
                    : 12'b0
                    : 12'b0;
assign  z_comple1 =  valid ?
                    z_in_bit1 ? 
                    ~z_in_bit1 + 1 
                    : 12'b0
                    : 12'b0;

wire    [23:0]  x_shift0;
wire    [23:0]  y_shift0;
wire    [23:0]  z_shift0;


assign  x_shift0 =   valid ? 
                    x_in_bit0 ? 
                    ({{12{x_in_bit0[11] ? 1'b1 : 1'b0}}, x_in_bit0} << 2*counts)
                    : 24'b0 
                    : 24'b0;
assign  y_shift0 =   valid ? 
                    y_in_bit0 ? 
                    ({{12{y_in_bit0[11] ? 1'b1 : 1'b0}}, y_in_bit0} << 2*counts) 
                    : 24'b0 
                    : 24'b0;
assign  z_shift0 =   valid ? 
                    z_in_bit0 ? 
                    ({{12{z_in_bit0[11] ? 1'b1 : 1'b0}}, z_in_bit0} << 2*counts) 
                    : 24'b0 
                    : 24'b0;

wire    [23:0]  x_shift1;
wire    [23:0]  y_shift1;
wire    [23:0]  z_shift1;


assign  x_shift1 =   valid ? 
                    x_in_bit1 ? 
                    counts == 5 ? 
                    ({x_comple1[11], x_comple1[11:0]} << (2*counts+1))
                    : ({{12{x_in_bit1[11] ? 1'b1 : 1'b0}}, x_in_bit1} << (2*counts+1))
                    : 24'b0 
                    : 24'b0;
assign  y_shift1 =   valid ? 
                    y_in_bit1 ? 
                    counts == 5 ?
                    ({y_comple1[11], y_comple1[11:0]} << (2*counts+1))
                    : ({{12{y_in_bit1[11] ? 1'b1 : 1'b0}}, y_in_bit1} << (2*counts+1))
                    : 24'b0 
                    : 24'b0;
assign  z_shift1 =   valid ? 
                    z_in_bit1 ? 
                    counts == 5 ?
                    ({z_comple1[11], z_comple1[11:0]} << (2*counts+1))
                    : ({{12{z_in_bit1[11] ? 1'b1 : 1'b0}}, z_in_bit1} << (2*counts+1))
                    : 24'b0 
                    : 24'b0;

reg [23:0]  temp_x;
reg [23:0]  temp_y;
reg [23:0]  temp_z;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        temp_x <= 24'b0;
        temp_y <= 24'b0;
        temp_z <= 24'b0;
    end
    else if(!valid) begin
        temp_x <= 24'b0;
        temp_y <= 24'b0;
        temp_z <= 24'b0;
    end
    else if(valid && counts <= 5) begin
        temp_x <= temp_x + x_shift0 + x_shift1;
        temp_y <= temp_y + y_shift0 + y_shift1;
        temp_z <= temp_z + z_shift0 + z_shift1;
    end
end

always @(posedge clk or negedge rst_n) begin 
    if(!rst_n)
        dist2   <= 28'b0;
    else if(!valid)
        dist2   <= 28'b0;
    else if(counts==6)
        dist2   <= ((~(temp_x+temp_y+temp_z)+1) << 1) + key_self2 + query_self2;
end

endmodule
