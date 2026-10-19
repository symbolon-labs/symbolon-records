import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import final_v2


class FinalGuards(unittest.TestCase):
    def test_exclusive_manifest_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            final_v2.exclusive_json(path, {"initial": True})
            with self.assertRaises(FileExistsError):
                final_v2.exclusive_json(path, {"initial": False})
            self.assertEqual(json.loads(path.read_text()), {"initial": True})

    def test_unpassed_mechanical_gate_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run_tiny").mkdir()
            (root / "run_tiny/semantic_gates.json").write_text(json.dumps({"gate0": True, "gate1": False, "gate2": False}))
            with patch.object(final_v2, "ROOT", root):
                with self.assertRaises(AssertionError):
                    final_v2.freeze()
            self.assertFalse((root / "run_full/FINAL_FREEZE.json").exists())
            self.assertFalse((root / "data/final").exists())

    def test_final_already_materialized_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data/final").mkdir(parents=True)
            with patch.object(final_v2, "ROOT", root):
                with self.assertRaises(AssertionError):
                    final_v2.freeze()

    def test_code_change_invalidates_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run_full").mkdir()
            (root / "run_full/FINAL_FREEZE.json").write_text(json.dumps({"components": {"old": "hash"}}))
            with patch.object(final_v2, "ROOT", root), patch.object(final_v2, "components", return_value={"new": "hash"}):
                with self.assertRaises(AssertionError):
                    final_v2.read_freeze()


if __name__ == "__main__":
    unittest.main()
