"""Run ACT through the parent benchmark's shared Isaac evaluator and metrics."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_vlm.evaluation.isaac import main as evaluate


def main():
    evaluate(policy_type='act')

if __name__ == '__main__':
    main()
