"""Check PI05 assets/environment; optionally fetch only licensed tokenizer files."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/pi05/smoke.yaml')
    parser.add_argument('--fetch-tokenizer', action='store_true')
    args = parser.parse_args()
    from excavator_vlm.policies.pi05.config import resolve_settings
    settings = resolve_settings(args.config)
    if args.fetch_tokenizer:
        os.environ['HF_HUB_OFFLINE'] = '0'
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
        try:
            snapshot_download('google/paligemma-3b-pt-224', revision='35e4f46485b4d07967e7e9935bc3786aad50687c',
                              allow_patterns=['tokenizer*', 'special_tokens_map.json', 'added_tokens.json',
                                              'config.json', 'README.md', 'LICENSE*', 'NOTICE'],
                              local_dir=settings['assets']['tokenizer_path'], token=True)
        except GatedRepoError as exc:
            raise SystemExit('Accept the google/paligemma-3b-pt-224 license and run hf auth login on this server. '
                             'Do not put tokens in source/configuration files.') from exc
    os.environ['HF_HUB_OFFLINE'] = '1'
    from excavator_vlm.policies.pi05.inference import check_environment
    from excavator_vlm.policies.pi05.checkpoints import verified_base
    from transformers import AutoTokenizer
    check_environment()
    base = verified_base(settings['assets']['base_path'], settings['assets']['base_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(settings['assets']['tokenizer_path'], local_files_only=True)
    print(json.dumps({'status': 'passed', 'base': str(base), 'tokenizer': settings['assets']['tokenizer_path'],
                      'vocab_size': len(tokenizer)}, indent=2))


if __name__ == '__main__':
    main()
