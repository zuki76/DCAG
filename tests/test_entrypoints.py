"""Validate launch configuration and complete evaluation coverage."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import training_command
from evaluate import validate_predictions


class EntrypointTests(unittest.TestCase):
    def test_five_stage_commands_match_training_arguments(self):
        module = ast.parse((ROOT / "llava/train/train.py").read_text())
        fields = {
            statement.target.id
            for node in module.body
            if isinstance(node, ast.ClassDef)
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
        }
        for task in range(1, 6):
            command, inputs, config = training_command(
                task, "configs/train", "configs/model.json", "0,1", 29500
            )
            for flag in command:
                if flag.startswith("--dcag_") or flag == "--domain_name":
                    self.assertIn(flag[2:], fields)
            self.assertEqual(
                config["gpu_num"] * config["batch_size"] * config["grad_acc"], 16
            )
            self.assertEqual("--previous_task_model_path" in command, task > 1)
            self.assertTrue(inputs)

    def test_gpu_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            training_command(1, "configs/train", "configs/model.json", "0", 29500)

    def test_incomplete_or_duplicate_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "test.json"
            result = Path(directory) / "predictions.jsonl"
            annotation.write_text(
                json.dumps([{"question_id": "a"}, {"question_id": "b"}])
            )
            result.write_text('{"question_id":"a"}\n')
            with self.assertRaises(ValueError):
                validate_predictions(annotation, result)
            result.write_text('{"question_id":"a"}\n{"question_id":"a"}\n')
            with self.assertRaises(ValueError):
                validate_predictions(annotation, result)
            result.write_text('{"question_id":"a"}\n{"question_id":"b"}\n')
            validate_predictions(annotation, result)


if __name__ == "__main__":
    unittest.main()
