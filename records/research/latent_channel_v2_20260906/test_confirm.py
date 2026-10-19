import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import confirm


class ConfirmationQueue(unittest.TestCase):
    def test_successful_terminal_development_requires_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_full_completion.json").write_text('{"status":"completed"}')
            terminal = {"status": "complete", "stages": [{"exit_code": 0, "status": "complete"}]}
            with patch.object(confirm, "ROOT", root), patch.object(confirm, "development_state", return_value=terminal):
                confirm.wait_for_development({}, root)

    def test_failed_development_cannot_advance(self):
        with patch.object(confirm, "development_state", return_value={"status": "failed"}):
            with self.assertRaises(AssertionError):
                confirm.wait_for_development({}, Path("unused-test-path"))

    def test_wrong_process_identity_cannot_advance(self):
        with patch.object(confirm, "development_state", return_value={"status": "running", "pid": 123}), \
             patch.object(Path, "read_bytes", return_value=b"some-other-process"):
            with self.assertRaises(AssertionError):
                confirm.wait_for_development({}, Path("unused-test-path"))

    def test_failing_selected_checkpoint_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ["run_full/eval_0050", "dev_reproduction", "confirmation_job"]:
                (root / folder).mkdir(parents=True)
            (root / "run_full/decision.json").write_text('{"best_step":50}')
            (root / "dev_reproduction/verification.json").write_text(json.dumps({"selected_step":50,
                "all_conditions_exact":True, "endpoint_frozen":True}))
            metric = {"correct": {"accuracy": .80, "per_class": [1,1,1,.2]},
                      "receiver_only": {"accuracy": .25}, "zero": {"accuracy": .25}}
            for control in ["shuffled", "random_71001", "random_71002"]:
                metric[control] = {"accuracy": .25}
            (root / "run_full/eval_0050/metrics.json").write_text(json.dumps({"conditions":metric}))
            with patch.object(confirm, "ROOT", root):
                with self.assertRaises(AssertionError):
                    confirm.preflight()
            self.assertFalse((root / "data/final").exists())
            result = json.loads((root / "confirmation_job/dev_preflight.json").read_text())
            self.assertFalse(result["checks"]["correct_ge90"])


if __name__ == "__main__":
    unittest.main()
