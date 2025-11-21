`timescale 100ps / 100ps

module ALU(
    input [15:0] A,
    input [15:0] B,
    input Cin,
    input [3:0] OP,
    output reg Cout,
    output reg [15:0] C
    );
    
    parameter [3:0] OP_ADD = 4'b0000,
                    OP_SUB = 4'b0001,
                    OP_ID = 4'b0010,
                    OP_NAND  = 4'b0011,
                    OP_NOR = 4'b0100,
                    OP_XNOR = 4'b0101,
                    OP_NOT= 4'b0110,
                    OP_AND = 4'b0111,
                    OP_OR= 4'b1000,
                    OP_XOR  = 4'b1001,
                    OP_LRS = 4'b1010,
                    OP_ARS = 4'b1011,
                    OP_RR = 4'b1100,
                    OP_LLS = 4'b1101,
                    OP_ALS  = 4'b1110,
                    OP_RL  = 4'b1111;

    always @(*) begin
        {Cout, C} = 17'b0;
        case (OP)
            OP_ADD: {Cout, C} = A + B + Cin;
            OP_SUB: {Cout, C} = A - B - Cin;
            OP_ID: C = A;
            OP_NAND: C = ~(A & B);
            OP_NOR: C = ~(A | B);
            OP_XNOR: C = ~(A ^ B);
            OP_NOT: C = ~A;
            OP_AND: C = A & B;
            OP_OR: C = A | B;
            OP_XOR: C = A ^ B;
            OP_LRS: C = A >> 1;
            OP_ARS: C = {A[15],A[15:1]};
            OP_RR: C = {A[0], A[15:1]}; 
            OP_LLS: C = A << 1;
            OP_ALS: C = A <<< 1;
            OP_RL: C = {A[14:0], A[15]};
            default: {Cout, C} = 17'b0;
        endcase
    end

endmodule