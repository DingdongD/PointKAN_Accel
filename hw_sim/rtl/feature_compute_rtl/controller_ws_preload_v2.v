`timescale 1ns/1ps

module controller_ws_preload_v2 #(
    parameter M          = 8,
    parameter K          = 8,
    parameter N          = 8,
    parameter ARR_R      = 4,
    parameter ARR_C      = 4,
    parameter WORD_SIZE  = 16,
    parameter ACC_WIDTH  = 64,
    parameter CNT_WIDTH  = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,

    input  wire [M*K*WORD_SIZE-1:0] A_flat,
    input  wire [K*N*WORD_SIZE-1:0] B_flat,

    input  wire [ARR_C*ACC_WIDTH-1:0] vert_bottom_bus,
    input  wire [ARR_C-1:0]           vert_bottom_valid_bus,

    output reg  en,
    output reg  psum_clr,
    output reg  [ARR_R*WORD_SIZE-1:0] a_left_bus,
    output reg  [ARR_R-1:0]           a_valid_bus,
    output reg  [ARR_C*ACC_WIDTH-1:0] vert_top_bus,
    output reg  [ARR_C-1:0]           vert_top_valid_bus,

    output reg  [ARR_R*ARR_C*WORD_SIZE-1:0] w_local_bus,
    output reg  [ARR_R*ARR_C-1:0]           w_local_valid_bus,

    output reg  [M*N*ACC_WIDTH-1:0] C_out_flat,
    output reg  done
);

    localparam S_IDLE         = 4'd0;
    localparam S_PRELOAD_DATA = 4'd1;
    localparam S_PRELOAD_FIRE = 4'd2;
    localparam S_RUN          = 4'd3;
    localparam S_CAPTURE_LAST = 4'd4;
    localparam S_GAP          = 4'd5;
    localparam S_STORE        = 4'd6;
    localparam S_DONE         = 4'd7;

    reg [3:0] state;

    integer m_idx, n0, k0;
    reg [CNT_WIDTH-1:0] step_cnt;
    reg signed [ACC_WIDTH-1:0] partial [0:ARR_C-1];

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

    integer rr, cc;
    reg signed [WORD_SIZE-1:0] a_val, w_val;
    reg signed [ACC_WIDTH-1:0] p_val;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            en <= 0;
            psum_clr <= 0;
            a_left_bus <= 0;
            a_valid_bus <= 0;
            vert_top_bus <= 0;
            vert_top_valid_bus <= 0;
            w_local_bus <= 0;
            w_local_valid_bus <= 0;
            C_out_flat <= 0;
            done <= 0;
            m_idx <= 0;
            n0 <= 0;
            k0 <= 0;
            step_cnt <= 0;
            for (cc = 0; cc < ARR_C; cc = cc + 1)
                partial[cc] <= 0;
        end else begin
            a_left_bus <= 0;
            a_valid_bus <= 0;
            vert_top_bus <= 0;
            vert_top_valid_bus <= 0;
            w_local_valid_bus <= 0;
            done <= 0;
            psum_clr <= 0;

            case (state)
                S_IDLE: begin
                    en <= 0;
                    if (start) begin
                        C_out_flat <= 0;
                        m_idx <= 0;
                        n0 <= 0;
                        k0 <= 0;
                        step_cnt <= 0;
                        for (cc = 0; cc < ARR_C; cc = cc + 1)
                            partial[cc] <= 0;
                        state <= S_PRELOAD_DATA;
                    end
                end

                S_PRELOAD_DATA: begin
                    en <= 0;
                    for (rr = 0; rr < ARR_R; rr = rr + 1) begin
                        for (cc = 0; cc < ARR_C; cc = cc + 1) begin
                            if ((k0 + rr) < K && (n0 + cc) < N)
                                w_val = get_B(k0 + rr, n0 + cc);
                            else
                                w_val = 0;
                            w_local_bus[(rr*ARR_C + cc)*WORD_SIZE +: WORD_SIZE] <= w_val;
                        end
                    end
                    state <= S_PRELOAD_FIRE;
                end

                S_PRELOAD_FIRE: begin
                    en <= 0;
                    w_local_valid_bus <= {ARR_R*ARR_C{1'b1}};
                    step_cnt <= 0;
                    state <= S_RUN;
                end

                S_RUN: begin
                    en <= 1;

                    for (rr = 0; rr < ARR_R; rr = rr + 1) begin
                        if (step_cnt == rr && (k0 + rr) < K) begin
                            a_val = get_A(m_idx, k0 + rr);
                            a_left_bus[rr*WORD_SIZE +: WORD_SIZE] <= a_val;
                            a_valid_bus[rr] <= 1'b1;
                        end
                    end

                    for (cc = 0; cc < ARR_C; cc = cc + 1) begin
                        if (step_cnt == cc && (n0 + cc) < N) begin
                            p_val = partial[cc];
                            vert_top_bus[cc*ACC_WIDTH +: ACC_WIDTH] <= p_val;
                            vert_top_valid_bus[cc] <= 1'b1;
                        end
                    end

                    for (cc = 0; cc < ARR_C; cc = cc + 1) begin
                        if (vert_bottom_valid_bus[cc] && ((n0 + cc) < N))
                            partial[cc] <= $signed(vert_bottom_bus[cc*ACC_WIDTH +: ACC_WIDTH]);
                    end

                    step_cnt <= step_cnt + 1'b1;
                    if (step_cnt >= (ARR_R + ARR_C - 1))
                        state <= S_CAPTURE_LAST;
                end

                S_CAPTURE_LAST: begin
                    en <= 0;
                    for (cc = 0; cc < ARR_C; cc = cc + 1) begin
                        if (vert_bottom_valid_bus[cc] && ((n0 + cc) < N))
                            partial[cc] <= $signed(vert_bottom_bus[cc*ACC_WIDTH +: ACC_WIDTH]);
                    end
                    state <= S_GAP;
                end

                S_GAP: begin
                    en <= 0;
                    state <= S_STORE;
                end

                S_STORE: begin
                    if ((k0 + ARR_R) < K) begin
                        k0 <= k0 + ARR_R;
                        state <= S_PRELOAD_DATA;
                    end else begin
                        for (cc = 0; cc < ARR_C; cc = cc + 1)
                            if ((n0 + cc) < N)
                                C_out_flat[(m_idx*N + (n0 + cc))*ACC_WIDTH +: ACC_WIDTH] <= partial[cc];

                        if ((n0 + ARR_C) < N) begin
                            n0 <= n0 + ARR_C;
                            k0 <= 0;
                            for (cc = 0; cc < ARR_C; cc = cc + 1) partial[cc] <= 0;
                            state <= S_PRELOAD_DATA;
                        end else if ((m_idx + 1) < M) begin
                            m_idx <= m_idx + 1;
                            n0 <= 0;
                            k0 <= 0;
                            for (cc = 0; cc < ARR_C; cc = cc + 1) partial[cc] <= 0;
                            state <= S_PRELOAD_DATA;
                        end else begin
                            state <= S_DONE;
                        end
                    end
                end

                S_DONE: begin
                    en <= 0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule