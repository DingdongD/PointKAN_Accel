`include "adder.v"

module BitDist2_LB #(
	parameter D = 3,
    parameter B = 8,
    parameter F_WIDTH = 10,
	parameter DIST_WIDTH = 19,
	parameter LB_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [2*D-1:0] q_r_3dflat,
	input wire init_en,
	input wire [DIST_WIDTH-1:0] kth_dist2,
	
    output wire [DIST_WIDTH-1:0] dist2_out,
	output wire early_terminate_out
);
	localparam SUB_01 = 2'b01;
	localparam SUB_10 = 2'b10;

    reg [4:0] m;
	wire [4:0] B_m;
	assign B_m = B - m - 1;
	
	wire q_r2_D [0:D-1];
	wire [D-1:0] q_r2_sum;
	assign q_r2_D[0] = (q_r_3dflat[1:0] == SUB_01 || q_r_3dflat[1:0] == SUB_10) ? 1 : 0;
	assign q_r2_D[1] = (q_r_3dflat[3:2] == SUB_01 || q_r_3dflat[3:2] == SUB_10) ? 1 : 0;
	assign q_r2_D[2] = (q_r_3dflat[5:4] == SUB_01 || q_r_3dflat[5:4] == SUB_10) ? 1 : 0;
	assign q_r2_sum = q_r2_D[0] + q_r2_D[1] + q_r2_D[2];
    wire [DIST_WIDTH-1:0] q_r2_sum_ex;
	assign q_r2_sum_ex = {{(F_WIDTH-D-1){q_r2_sum[D-1]}}, q_r2_sum};
	
    wire signed [F_WIDTH-1:0] cross_term [0:D-1];
    wire signed [F_WIDTH-1:0] cross_term_sum;
	reg signed [F_WIDTH-1:0] f [0:D-1];
    wire signed [F_WIDTH-1:0] f_next [0:D-1];
	wire [LB_WIDTH-1:0] f_next_2[D-1:0];
	assign cross_term[0] = q_r_3dflat[1:0] == SUB_01 ? f[0] : (q_r_3dflat[1:0] == SUB_10 ? -f[0] : 0);
	assign cross_term[1] = q_r_3dflat[3:2] == SUB_01 ? f[1] : (q_r_3dflat[3:2] == SUB_10 ? -f[1] : 0);
	assign cross_term[2] = q_r_3dflat[5:4] == SUB_01 ? f[2] : (q_r_3dflat[5:4] == SUB_10 ? -f[2] : 0);			   
    assign cross_term_sum = cross_term[0] + cross_term[1] + cross_term[2];
	
	assign f_next[0] = q_r_3dflat[1:0] == SUB_01 ? (f[0] << 1) + 1 :(q_r_3dflat[1:0] == SUB_10 ? (f[0] << 1) - 1 : f[0] << 1);
	assign f_next[1] = q_r_3dflat[3:2] == SUB_01 ? (f[1] << 1) + 1 :(q_r_3dflat[3:2] == SUB_10 ? (f[1] << 1) - 1 : f[1] << 1);
	assign f_next[2] = q_r_3dflat[5:4] == SUB_01 ? (f[2] << 1) + 1 :(q_r_3dflat[5:4] == SUB_10 ? (f[2] << 1) - 1 : f[2] << 1);
	
	reg signed [DIST_WIDTH-1:0] dist2;
	wire [DIST_WIDTH-1:0] dist2_shift;
	assign dist2_shift = dist2 << 2;
	assign dist2_out = dist2;

	wire [F_WIDTH-1:0] psum_csa, pcarry_csa;
	wire [F_WIDTH-1:0] cross_term_sum_shift = cross_term_sum << 2;
    genvar i;
    generate
        for (i = 0; i < F_WIDTH; i = i + 1) begin : pcsa_gen
            assign psum_csa[i] = q_r2_sum_ex[i] ^ cross_term_sum_shift[i];
            assign pcarry_csa[i] = q_r2_sum_ex[i] & cross_term_sum_shift[i];
        end
    endgenerate

    wire [F_WIDTH-1:0] p_sum;
	reg [DIST_WIDTH-1:0] psum_delay;
    csa_10b partial_adder(
        .a(psum_csa),
        .b({pcarry_csa[F_WIDTH-2:0], 1'b0}),
        .cin(1'b0),
        .sum(p_sum),
        .cout()
    );
	
	wire [DIST_WIDTH-1:0] sum_csa, carry_csa;
	genvar n;
    generate
        for (n = 0; n < DIST_WIDTH; n = n + 1) begin : csa_gen
            assign sum_csa[n] = dist2_shift[n] ^ psum_delay[n];
            assign carry_csa[n] = dist2_shift[n] & psum_delay[n];
        end
    endgenerate

    wire [DIST_WIDTH-1:0] final_sum;
    csa_19b final_adder(
        .a(sum_csa),
        .b({carry_csa[DIST_WIDTH-2:0], 1'b0}),
        .cin(1'b0),
        .sum(final_sum),
        .cout()
    );

	wire signed [F_WIDTH-1:0] h [0:D-1];
	wire signed [F_WIDTH-1:0] h_sum;
	wire signed [LB_WIDTH-1:0] LB [0:D-1];
	wire signed [LB_WIDTH-1:0] LB_sum;
	assign h[0] = (f_next[0]>=0) ? 2*f_next[0] - 1 : 2*f_next[0] + 1;
	assign h[1] = (f_next[1]>=0) ? 2*f_next[1] - 1 : 2*f_next[1] + 1;
	assign h[2] = (f_next[2]>=0) ? 2*f_next[2] - 1 : 2*f_next[2] + 1;
	assign h_sum = h[0] + h[1] + h[2];
	assign LB_sum = m<B-1 ? (final_sum - h_sum) <<2*(B_m) : 0;
		
    reg clear_next; 
	reg early_terminate;
	assign early_terminate_out = early_terminate;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist2 <= 0;
            m <= 0;
            clear_next <= 0;
			early_terminate <= 0;
			f[0] <= 0;
			f[1] <= 0;
			f[2] <= 0;
			psum_delay <= 0;
        end else if (clear_next) begin
            dist2 <= 0;
            m <= 0;
            clear_next <= 0;
			early_terminate <= 0;
			f[0] <= 0;
			f[1] <= 0;
			f[2] <= 0;
			psum_delay <= 0;
        end else if (enable && !clear_next) begin
            if ( m < B) begin
				psum_delay <= {{(DIST_WIDTH-F_WIDTH-1){p_sum[F_WIDTH-1]}}, p_sum} << 2;
				dist2 <= final_sum;
				
                m <= m + 1;
                f[0] <= f_next[0];
				f[1] <= f_next[1];
				f[2] <= f_next[2];
                
				if (!init_en && (LB_sum > kth_dist2))begin
					early_terminate <= 1;
					clear_next <= 1;
				end
				else if (m == B-1)begin
					clear_next <= 1;
				end else begin
					early_terminate <= 0;
					clear_next <= 0;
				end
            end else
                clear_next <= 1;
        end
    end
endmodule
