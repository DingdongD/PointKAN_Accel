/* verilator lint_off MODDUP */
module csa (a,b,ci,s,c);	// carry save adders
	input a,b,ci;		// input a,b,carry_in
	output s,c;		// output sum,carry_out
	assign c = a&b | b&ci | ci&a;
	assign s = (a^b)^ci;
endmodule