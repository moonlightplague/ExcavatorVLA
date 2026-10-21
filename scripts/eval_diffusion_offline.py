"""Evaluate Diffusion Policy using the shared offline benchmark evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_vlm.evaluation.offline import main as evaluate
from excavator_vlm.policies.diffusion.dataset import make_diffusion_dataset
from excavator_vlm.policies.diffusion.inference import load_diffusion


def main():
    evaluate(load_policy=load_diffusion, make_dataset=make_diffusion_dataset, policy_type='diffusion')


if __name__ == '__main__':
    main()
