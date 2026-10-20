"""Evaluate PI05 through the shared physical-unit offline metrics."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator_vlm.evaluation.offline import main
from excavator_vlm.policies.pi05.dataset import make_pi05_dataset
from excavator_vlm.policies.pi05.inference import load_pi05

if __name__ == '__main__':
    main(load_pi05, make_pi05_dataset, 'pi05')
