"""Evaluate ACT using the shared offline benchmark evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_vlm.evaluation.offline import main as evaluate
from excavator_vlm.policies.act.dataset import make_act_dataset
from excavator_vlm.policies.act.inference import load_act


def main():
    evaluate(load_policy=load_act, make_dataset=make_act_dataset, policy_type='act')


if __name__ == '__main__':
    main()
