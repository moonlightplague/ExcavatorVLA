import ast
import re
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
