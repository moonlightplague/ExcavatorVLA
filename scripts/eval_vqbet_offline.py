"""Evaluate VQ-BeT using the shared offline benchmark evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_vlm.evaluation.offline import main as evaluate
from excavator_vlm.policies.vqbet.dataset import make_vqbet_dataset
from excavator_vlm.policies.vqbet.inference import load_vqbet


def main():
    evaluate(load_policy=load_vqbet, make_dataset=make_vqbet_dataset, policy_type='vqbet')


if __name__ == '__main__':
    main()
