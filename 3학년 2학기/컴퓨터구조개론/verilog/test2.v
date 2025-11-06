module wave_test;
  reg clk = 0, rst = 1;

  always #5 clk = ~clk; // 10ns period

  initial begin
    $dumpfile("wave.vcd");
    $dumpvars(0, wave_test);
    #12 rst = 0;
    #80 $finish;
  end
endmodule
