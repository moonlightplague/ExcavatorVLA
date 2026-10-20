# Vendored LeRobot policy source

This directory contains the ACT and pi0.5 policy implementation copied from
Hugging Face LeRobot tag `v0.4.4` at commit
`8fff0fde7c79f23a93d845d1a50e985de01f8b8a`.

Upstream repository: <https://github.com/huggingface/lerobot>

Included source:

- `src/lerobot/policies/act/`
- `src/lerobot/policies/pi05/`
- the shared policy factory, pretrained base class, and utilities present in
  the sparse upstream checkout
- `docs/source/policy_act_README.md`, the target of ACT's upstream README
  symlink

This is a provenance-preserving reference copy for the ExcavatorVLA benchmark.
Do not add this directory to `PYTHONPATH`: these files import other LeRobot
modules that are supplied by the installed `lerobot==0.4.4` package. Benchmark
adapters and excavator-specific configuration belong in this submodule, while
large checkpoints belong in the parent project's `models/` directory.

The upstream license is preserved in `LICENSE`.
