"""Run the isolated fixed-input PI05 numerical diagnostic."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator_vlm.policies.pi05.diagnostics import main

if __name__ == '__main__':
    main()
