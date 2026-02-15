import subprocess
import sys
import unittest


class EntryCliTests(unittest.TestCase):
    def test_emit_ast_demo(self):
        proc = subprocess.run(
            [sys.executable, "-m", "mycompiler.entry", "--emit", "ast", "--demo"],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("Module(", proc.stdout)

    def test_emit_run_demo_outputs_fibonacci_prefix(self):
        proc = subprocess.run(
            [sys.executable, "-m", "mycompiler.entry", "--emit", "run", "--demo"],
            text=True,
            capture_output=True,
            check=True,
        )
        lines = [x.strip() for x in proc.stdout.strip().splitlines() if x.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertEqual(lines[0], "0")
        self.assertEqual(lines[1], "1")


if __name__ == "__main__":
    unittest.main()
