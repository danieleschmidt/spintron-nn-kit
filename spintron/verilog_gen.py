"""
Verilog generator for SOT crossbar arrays.

Generates synthesizable RTL that models the digital-domain equivalent of the
analog SOT crossbar. Each weight is stored as a 1-bit register (0 = parallel,
1 = antiparallel). The VMM is approximated as an integer accumulate.

Output module interface:
    - clk, rst_n: clock & active-low reset
    - input [DATA_W-1:0] in_vec [0:ROWS-1]: input activations
    - output reg [ACC_W-1:0] out_vec [0:COLS-1]: output accumulations
    - weight_val [ROWS-1:0][COLS-1:0]: flattened 1-bit weight matrix (P=1, AP=0)
"""

from __future__ import annotations

import os
from typing import Optional

from .crossbar import CrossbarArray


class VerilogGenerator:
    """
    Generate synthesizable Verilog RTL for a programmed SOT crossbar.

    Args:
        crossbar (CrossbarArray): Programmed crossbar instance.
        data_width (int): Bit-width of each input element (default 8).
    """

    def __init__(self, crossbar: CrossbarArray, data_width: int = 8):
        self.crossbar = crossbar
        self.data_width = data_width
        self.acc_width = data_width + int(crossbar.rows).bit_length() + 1

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_module(self, module_name: str = "sot_crossbar") -> str:
        """
        Generate a synthesizable Verilog module string.

        Args:
            module_name: Verilog module identifier.

        Returns:
            Multi-line Verilog source string.
        """
        rows = self.crossbar.rows
        cols = self.crossbar.cols
        dw = self.data_width
        aw = self.acc_width

        # Encode weight matrix as Verilog parameter init
        weight_rows = []
        for r in range(rows):
            row_bits = []
            for c in range(cols):
                dev = self.crossbar.device(r, c)
                # parallel → 1 (high conductance / positive weight)
                # antiparallel → 0 (low conductance / zero weight)
                bit = 1 if dev.is_parallel else 0
                row_bits.append(str(bit))
            weight_rows.append(", ".join(row_bits))

        lines = []
        lines.append(f"// Auto-generated SOT crossbar RTL — {rows}x{cols}")
        lines.append(f"// Rows (inputs): {rows}, Cols (outputs): {cols}")
        lines.append(f"// Data width: {dw} bits, Accumulator width: {aw} bits")
        lines.append("")
        lines.append(f"module {module_name} #(")
        lines.append(f"    parameter ROWS     = {rows},")
        lines.append(f"    parameter COLS     = {cols},")
        lines.append(f"    parameter DATA_W   = {dw},")
        lines.append(f"    parameter ACC_W    = {aw}")
        lines.append(") (")
        lines.append("    input  wire                      clk,")
        lines.append("    input  wire                      rst_n,")
        lines.append(f"    input  wire [DATA_W-1:0]         in_vec  [0:ROWS-1],")
        lines.append(f"    output reg  [ACC_W-1:0]          out_vec [0:COLS-1]")
        lines.append(");")
        lines.append("")
        lines.append("    // Weight memory: 1=parallel (high-G), 0=antiparallel (low-G)")
        lines.append(f"    reg weight [0:ROWS-1][0:COLS-1];")
        lines.append("")
        lines.append("    integer r, c;")
        lines.append("    reg [ACC_W-1:0] acc [0:COLS-1];")
        lines.append("")
        lines.append("    // Weight initialisation (synthesis: use memory init file)")
        lines.append("    initial begin")
        for r in range(rows):
            for c in range(cols):
                dev = self.crossbar.device(r, c)
                bit = 1 if dev.is_parallel else 0
                lines.append(f"        weight[{r}][{c}] = 1'b{bit};")
        lines.append("    end")
        lines.append("")
        lines.append("    // Combinational VMM: accumulate over rows")
        lines.append("    always @(posedge clk or negedge rst_n) begin")
        lines.append("        if (!rst_n) begin")
        lines.append("            for (c = 0; c < COLS; c = c + 1)")
        lines.append("                out_vec[c] <= {ACC_W{1'b0}};")
        lines.append("        end else begin")
        lines.append("            for (c = 0; c < COLS; c = c + 1) begin")
        lines.append("                acc[c] = {ACC_W{1'b0}};")
        lines.append("                for (r = 0; r < ROWS; r = r + 1) begin")
        lines.append("                    if (weight[r][c])")
        lines.append("                        acc[c] = acc[c] + {{(ACC_W-DATA_W){1'b0}}, in_vec[r]};")
        lines.append("                end")
        lines.append("                out_vec[c] <= acc[c];")
        lines.append("            end")
        lines.append("        end")
        lines.append("    end")
        lines.append("")
        lines.append(f"endmodule  // {module_name}")

        return "\n".join(lines) + "\n"

    def save(self, filepath: str) -> None:
        """
        Write the generated Verilog to a file.

        Args:
            filepath: Destination file path (created/overwritten).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        verilog = self.generate_module(
            module_name=os.path.splitext(os.path.basename(filepath))[0]
        )
        with open(filepath, "w") as f:
            f.write(verilog)

    def __repr__(self) -> str:
        return (
            f"VerilogGenerator(crossbar={self.crossbar!r}, "
            f"data_width={self.data_width})"
        )
