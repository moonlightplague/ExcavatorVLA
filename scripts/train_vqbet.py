"""Compatibility CLI for the VQ-BeT training implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_vlm.policies.vqbet.train import main

if __name__ == '__main__':
    main()
