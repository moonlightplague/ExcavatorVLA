import ast
import re
import unittest
from pathlib import Path
from typing import Any

import numpy as np


class PolicyClientCliContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.client_path = (
            self.repo_root
            / "scripts"
            / "bridge_test"
            / "smolvla_policy_client.py"
        )
        self.tree = ast.parse(
            self.client_path.read_text(encoding="utf-8")
        )
        self.option_names = set()
        self.destinations = set()
        for node in ast.walk(self.tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "parser"
                and node.func.attr == "add_argument"
            ):
                continue
            long_options = [
                argument.value
                for argument in node.args
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.startswith("--")
                )
            ]
            self.option_names.update(long_options)
            explicit_dest = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if (
                        keyword.arg == "dest"
                        and isinstance(keyword.value, ast.Constant)
                    )
                ),
                None,
            )
            if explicit_dest:
                self.destinations.add(explicit_dest)
            elif long_options:
                self.destinations.add(
                    long_options[0][2:].replace("-", "_")
                )

    def test_every_args_attribute_has_a_parser_destination(self):
        referenced = {
            node.attr
            for node in ast.walk(self.tree)
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "args"
            )
        }
        self.assertEqual(sorted(referenced - self.destinations), [])

    def test_policy_flags_in_launchers_are_declared(self):
        used_options = set()
        for launcher in self.repo_root.glob("run_isaacsim*.sh"):
            lines = launcher.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
            for start, line in enumerate(lines):
                if "$CLIENT" not in line or not line.rstrip().endswith("\\"):
                    continue
                end = start + 1
                while (
                    end < len(lines)
                    and lines[end].rstrip().endswith("\\")
                ):
                    end += 1
                command = "\n".join(lines[start:end + 1])
                used_options.update(
                    re.findall(r"--[a-z][a-z0-9-]*", command)
                )
        self.assertEqual(
            sorted(used_options - self.option_names),
            [],
        )

    def test_temporal_ensemble_helper_is_defined_and_blends_chunks(self):
        helper = next(
            (
                node
                for node in self.tree.body
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "temporal_ensemble_action"
                )
            ),
            None,
        )
        self.assertIsNotNone(helper)
        namespace = {
            "Any": Any,
            "EXPECTED_ACTION_DIM": 4,
            "np": np,
        }
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]),
                str(self.client_path),
                "exec",
            ),
            namespace,
        )
        ensemble, metadata = namespace["temporal_ensemble_action"](
            [
                (0, np.full((3, 4), 1.0, dtype=np.float32)),
                (1, np.full((2, 4), 3.0, dtype=np.float32)),
            ],
            target_step=1,
            width=2,
            decay=0.0,
        )
        np.testing.assert_allclose(ensemble, np.full(4, 2.0))
        self.assertEqual(metadata["num_predictions"], 2)
        self.assertEqual(metadata["source_chunk_origins"], [1, 0])

    def test_execution_constraints_assign_executed_action(self):
        source = self.client_path.read_text(encoding="utf-8")
        self.assertIn(
            "executed_action, execution_constraints = (\n"
            "                    apply_excavation_sequence_supervisor(",
            source,
        )
        self.assertIn(
            "executed_action, execution_constraints = (\n"
            "                    apply_optional_execution_constraints(",
            source,
        )

    def test_overview_recording_is_separate_from_model_cameras(self):
        source = self.client_path.read_text(encoding="utf-8")
        self.assertIn("--record-overview-camera", self.option_names)
        self.assertIn(
            'CAMERA_IDS + ("overview",)',
            source,
        )
        self.assertIn(
            'video_frames["overview"] = decode_rgb(',
            source,
        )
        self.assertIn(
            '"record_overview_camera": record_overview_camera',
            source,
        )


if __name__ == "__main__":
    unittest.main()
