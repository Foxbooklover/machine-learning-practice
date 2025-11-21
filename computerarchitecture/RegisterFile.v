`timescale 100ps / 100ps

module RF(
    input [1:0] addr1,
    input [1:0] addr2,
    input [1:0] addr3,
    input [15:0] data3,
    input write,
    input clk,
    input reset,
    output reg [15:0] data1,
    output reg [15:0] data2
    );
    
    reg [15:0] registers [0:3];


    always @(posedge clk) begin
       if(reset) begin
           registers[0] <= 16'b0;
           registers[1] <= 16'b0;
           registers[2] <= 16'b0;
           registers[3] <= 16'b0;
       end else if(write) begin
           registers[addr3] <= data3;
       end
    end
    

    always @(*) begin
        data1 = registers[addr1];
        data2 = registers[addr2];
    end
endmodule