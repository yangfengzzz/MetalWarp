import ast
import unittest
from pathlib import Path

from mycompiler.codegen_metal import MetalCodeGenerator


class SphKernelContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("mycompiler/sph_simulation.py").read_text()
        cls.tree = ast.parse(cls.source)
        cls.fn_map = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
        }

    def test_expected_kernel_functions_exist(self):
        expected = {
            "count_particles_per_cell",
            "prefix_sum_cell_counts",
            "scatter_particles_by_cell",
            "compute_density",
            "update_particles",
        }
        self.assertTrue(expected.issubset(self.fn_map.keys()))

    def test_mass_is_float_scalar_in_generated_metal(self):
        node = self.fn_map["compute_density"]
        code = MetalCodeGenerator().generate(ast.Module(body=[node], type_ignores=[]))
        self.assertIn("constant float& mass", code)
        self.assertIn("[[thread_position_in_grid]]", code)


if __name__ == "__main__":
    unittest.main()
