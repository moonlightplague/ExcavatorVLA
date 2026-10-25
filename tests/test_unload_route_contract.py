import ast
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "scripts" / "excavator_app" / "excavator_runtime.py"


def function_source(name):
    tree = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"function not found: {name}")


class UnloadRouteContractTests(unittest.TestCase):
    def test_full_plan_unload_waypoints_stay_in_one_loaded_motion_group(self):
        namespace = {"np": np, "LOADED_ROUTE_RUNTIME_FAST_EXEC": True}
        for name in ("clearance_route_stage_label", "loaded_route_continuous_group"):
            exec(function_source(name), namespace)
        route_label = namespace["clearance_route_stage_label"]
        q = np.zeros(4)
        seq = [("lift_carry", q, 1.), (route_label("unload_to_bin", 0), q, 1.),
               (route_label("unload_to_bin", 1), q, 1.), ("unload_to_bin", q, 1.),
               ("pre_dig", q, 1.)]
        group, next_index = namespace["loaded_route_continuous_group"](seq, 0)
        self.assertEqual([row[1] for row in group], [row[0] for row in seq[:4]])
        self.assertEqual(next_index, 4)
        self.assertEqual(route_label("pull_mid_cut", 0), "clearance_route_1")

    def test_loaded_route_generates_coordinated_arc_before_tuck_swing_fallback(self):
        source = function_source("find_clearance_route")
        self.assertIn("loaded_coordinated_arc", source)
        self.assertIn("LOADED_ROUTE_ARC_SWING_FRACTIONS", source)
        self.assertIn("boom_arm_swing_coordinated", source)
        arc_index = source.index("loaded_coordinated_arc")
        self.assertGreater(source.find("joint_tuck_swing", arc_index), arc_index)

    def test_loaded_route_prunes_waypoints_with_full_validator(self):
        source = function_source("find_clearance_route")
        self.assertIn("if pre_dig_route or carry_locked_route:", source)
        self.assertIn("minimize_valid_clearance_waypoints", source)
        self.assertIn("route_segments_ok", function_source("minimize_valid_clearance_waypoints"))

    def test_slow_loaded_template_is_compared_with_rrt(self):
        source = function_source("find_clearance_route")
        self.assertIn("LOADED_ROUTE_RRT_TIME_RATIO_TRIGGER", source)
        self.assertIn("(pre_dig_route or carry_locked_route)", source)
        self.assertIn("try_joint_rrt_route()", source)

    def test_unload_execution_keeps_one_continuous_motion_group(self):
        source = function_source("execute_unload_route_continuous_group")
        self.assertIn("path_waypoints=route_waypoints", source)
        self.assertIn("arrival_gate=unload_only", source)
        self.assertEqual(source.count("await move_to_profile("), 1)


if __name__ == "__main__":
    unittest.main()
