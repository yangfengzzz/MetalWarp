import unittest
from pathlib import Path


class RuntimeHeaderTests(unittest.TestCase):
    def test_metal_device_header_has_buffer_api(self):
        text = Path("metal_runtime/metal_device.h").read_text()
        self.assertIn("create_buffer", text)
        self.assertIn("run_kernel_with_buffers", text)
        self.assertIn("raw_buffer", text)

    def test_bindings_expose_renderer_buffer_path(self):
        text = Path("metal_runtime/bindings.mm").read_text()
        self.assertIn("render_frame_from_buffers", text)
        self.assertIn("py::class_<MetalRenderer>", text)


if __name__ == "__main__":
    unittest.main()
