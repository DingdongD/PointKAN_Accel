`timescale 1ns/1ps

module controller_os_v2 #(
    parameter M          = 8,
    parameter K          = 8,
    parameter N          = 8,
    parameter ARR_R      = 8,
    parameter ARR_C      = 8,
    parameter WORD_SIZE  = 16,
    parameter ACC_WIDTH  = 64,
    parameter CNT_WIDTH  = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,

    input  wire [M*K*WORD_SIZE-1:0] A_flat,
    input  wire [K*N*WORD_SIZE-1:0] B_flat,

    output reg  en,
    output reg  psum_clr,
    output reg  [ARR_R*WORD_SIZE-1:0] a_left_bus,
    output reg  [ARR_R-1:0]           a_valid_bus,
    output reg  [ARR_C*ACC_WIDTH-1:0] vert_top_bus,
    output reg  [ARR_C-1:0]           vert_top_valid_bus,

    output wire [ARR_R*ARR_C*WORD_SIZE-1:0] w_local_bus,
    output wire [ARR_R*ARR_C-1:0]           w_local_valid_bus,

    output reg  done
);

    localparam S_IDLE  = 2'd0;
    localparam S_CLEAR = 2'd1;
    localparam S_RUN   = 2'd2;
    localparam S_DONE  = 2'd3;

    reg [1:0] state;
    reg [CNT_WIDTH-1:0] cycle_cnt;

    function signed [WORD_SIZE-1:0] get_A;
        input integer row;
        input integer col;
        begin
            get_A = A_flat[(row*K + col)*WORD_SIZE +: WORD_SIZE];
        end
    endfunction

    function signed [WORD_SIZE-1:0] get_B;
        input integer row;
        input integer col;
        begin
            get_B = B_flat[(row*N + col)*WORD_SIZE +: WORD_SIZE];
        end
    endfunction

    assign w_local_bus       = {ARR_R*ARR_C*WORD_SIZE{1'b0}};
    assign w_local_valid_bus = {ARR_R*ARR_C{1'b0}};

    integer r, c, kk;
    reg signed [WORD_SIZE-1:0] a_val, b_val;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            cycle_cnt <= 0;
            en <= 0;
            psum_clr <= 0;
            a_left_bus <= 0;
            a_valid_bus <= 0;
            vert_top_bus <= 0;
            vert_top_valid_bus <= 0;
            done <= 0;
        end else begin
            a_left_bus <= 0;
            a_valid_bus <= 0;
            vert_top_bus <= 0;
            vert_top_valid_bus <= 0;
            done <= 0;

            case (state)
                S_IDLE: begin
                    en <= 0;
                    psum_clr <= 0;
                    cycle_cnt <= 0;
                    if (start)
                        state <= S_CLEAR;
                end

                S_CLEAR: begin
                    en <= 1;
                    psum_clr <= 1;
                    cycle_cnt <= 0;
                    state <= S_RUN;
                end

                S_RUN: begin
                    en <= 1;
                    psum_clr <= 0;

                    for (r = 0; r < ARR_R; r = r + 1) begin
                        kk = cycle_cnt - r;
                        if (r < M && kk >= 0 && kk < K) begin
                            a_val = get_A(r, kk);
                            a_left_bus[r*WORD_SIZE +: WORD_SIZE] <= a_val;
                            a_valid_bus[r] <= 1'b1;
                        end
                    end

                    for (c = 0; c < ARR_C; c = c + 1) begin
                        kk = cycle_cnt - c;
                        if (c < N && kk >= 0 && kk < K) begin
                            b_val = get_B(kk, c);
                            vert_top_bus[c*ACC_WIDTH +: ACC_WIDTH] <= {{(ACC_WIDTH-WORD_SIZE){b_val[WORD_SIZE-1]}}, b_val};
                            vert_top_valid_bus[c] <= 1'b1;
                        end
                    end

                    cycle_cnt <= cycle_cnt + 1'b1;
                    if (cycle_cnt >= (K + ARR_R + ARR_C + 3))
                        state <= S_DONE;
                end

                S_DONE: begin
                    en <= 0;
                    psum_clr <= 0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule