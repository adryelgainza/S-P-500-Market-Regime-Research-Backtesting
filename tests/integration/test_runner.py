"""Exercise real private kernels, normal execution and injected nested code."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nbformat

from regime_research.runner import execute_cells


PROJECT = Path(__file__).resolve().parents[2]


class FreshKernelTests(unittest.TestCase):
    def setUp(self):
        base = PROJECT/"tmp/kernel_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)

    def run_cells(self, sources, **kwargs):
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source) for source in sources])
        path = self.project/"test.ipynb"
        nbformat.write(notebook, path)
        return execute_cells(self.project, path, self.project/"run", timeout=60, **kwargs)

    def test_fresh_kernel_has_exact_execution_counts_and_receipt(self):
        receipt = self.run_cells(["value = 7", "print(value * 2)"])
        self.assertEqual([row["Execution_Count"] for row in receipt["Cells"]], [1, 2])
        executed = nbformat.read(self.project/"run/executed_notebook.ipynb", as_version=4)
        self.assertEqual(executed.cells[1].outputs[0].text.strip(), "14")

    def test_nested_untracked_execution_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_cells(['get_ipython().run_cell("unexpected = 1")'])
        self.assertFalse((self.project/"run/execution_receipt.json").exists())
        events = [json.loads(line) for line in (self.project/"run/kernel_audit.jsonl").read_text().splitlines()]
        self.assertIn("unexpected_nested_execution", [event["Type"] for event in events])

    def test_file_mutation_during_execution_is_rejected(self):
        marker = self.project/"marker.py"
        marker.write_text("initial")
        def unchanged():
            if marker.read_text() != "initial":
                raise ValueError("tracked source mutated")
        with self.assertRaisesRegex(ValueError, "tracked source mutated"):
            self.run_cells(['from pathlib import Path\nPath("marker.py").write_text("changed")'], source_check=unchanged)
        self.assertFalse((self.project/"run/execution_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
