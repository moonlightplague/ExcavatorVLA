"""Rank scored rollouts and emit the subset worth reusing for Best-of-N.

The success evaluator already scores every rollout from the simulator's own
particle counts, so selection needs no new simulation: it reads
``episode_status.json`` plus, when present, ``trajectory/meta.json``. Frames are
never opened here. That is the reason the store keeps the score in a small file
of its own -- a 500 episode pool ranks in well under a second, while touching
its frames would mean loading ~30 GB to discard most of it.

Two pool shapes are supported on purpose. A pool that already has trajectories
(``--require-trajectory``) narrows to what can be converted into a fine-tuning
dataset immediately. A pool that only has scores is still decision-grade: the
ranking says which seeds are worth re-running with ``--save-trajectory``, and
``selected_seeds`` is exactly that re-run list. Because inference is seeded per
episode, re-running a chosen seed reproduces the scored trajectory rather than
drawing a new one, so selecting first and collecting frames second costs nothing
in quality.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys

from .rollout_store import trajectory_reward

SCHEMA_VERSION = 'excavator_rollout_selection_v1'
EPISODE_STATUS = 'episode_status.json'
TRAJECTORY_META = Path('trajectory') / 'meta.json'
MODES = ('success', 'threshold', 'top_k', 'per_seed_best')
REJECT_REASONS = ('not_completed', 'missing_trajectory', 'too_short',
                  'excluded_failure_mode', 'excluded_seed', 'policy_mismatch',
                  'checkpoint_mismatch', 'duplicate')


def _integer(value, default=0):
    """Episode metrics are null on episodes that errored before their first observation."""
    return default if value is None else int(value)


def find_runs(paths):
    """Resolve ``--runs`` entries to run directories containing ``episodes/``.

    Accepting a parent directory as well keeps the common case of pointing at
    ``outputs/eval`` and letting discovery find every run inside it working,
    instead of requiring the caller to enumerate run folders by hand.
    """
    runs = []
    for entry in paths:
        folder = Path(entry)
        if not folder.is_dir():
            raise ValueError(f'Not a directory: {folder}')
        if (folder / 'episodes').is_dir():
            runs.append(folder)
        else:
            nested = sorted(child for child in folder.iterdir()
                            if child.is_dir() and (child / 'episodes').is_dir())
            if not nested:
                raise ValueError(f'No episodes/ under {folder}')
            runs.extend(nested)
    unique = list(dict.fromkeys(run.resolve() for run in runs))
    return unique


def load_run_config(run):
    path = run / 'config.json'
    if not path.exists():
        return {}, None
    return json.loads(path.read_text()), str(path)


def discover(runs):
    """Read every episode of every run into candidate records."""
    candidates = []
    for run in runs:
        config, config_path = load_run_config(run)
        for status_path in sorted(run.glob(f'episodes/*/{EPISODE_STATUS}')):
            status = json.loads(status_path.read_text())
            candidates.append(record(run, config_path, config, status_path, status))
    return candidates


def record(run, config_path, config, status_path, status):
    folder = status_path.parent
    meta_path = folder / TRAJECTORY_META
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    reward = trajectory_reward(status)
    return {
        'run': str(run),
        'run_config': config_path,
        'episode': folder.name,
        'path': str(folder),
        'scene_seed': status.get('scene_seed'),
        'policy_type': status.get('policy_type'),
        'checkpoint_sha256': status.get('checkpoint_sha256'),
        'evaluation_id': status.get('evaluation_id'),
        'execution_source': status.get('execution_source') or config.get('execution_source'),
        'replan_interval': config.get('replan_interval'),
        'status': status.get('status'),
        'steps': _integer(status.get('policy_steps_executed')),
        'failure_mode': status.get('failure_mode'),
        'spillage_fraction': status.get('spillage_fraction'),
        'trajectory_available': meta is not None,
        'trajectory_frames': _integer(meta.get('frames')) if meta else 0,
        'trajectory_meta': str(meta_path) if meta else None,
        'reward': reward,
        'score': reward['score'],
    }


def sort_key(candidate):
    """Deterministic best-first ordering.

    Score alone leaves ties, and ties must not resolve by filesystem order or a
    re-run of the same selection could return a different dataset. Falling back
    to the largest bucket load then to the seed makes the order total.
    """
    return (-candidate['score'],
            -candidate['reward']['max_bucket_particles'],
            candidate['scene_seed'] if candidate['scene_seed'] is not None else 0,
            candidate['path'])


def dedup_key(candidate):
    """Identity of one rollout experiment, so the same seed is never trained on twice."""
    return (candidate['checkpoint_sha256'], candidate['scene_seed'],
            candidate['replan_interval'], candidate['execution_source'],
            candidate['policy_type'])


def hard_reject(candidate, args):
    """Return a reject reason for candidates that cannot be used at all, else None."""
    if candidate['status'] != 'completed':
        return 'not_completed'
    if args.policy_type and candidate['policy_type'] != args.policy_type:
        return 'policy_mismatch'
    if args.checkpoint and candidate['checkpoint_sha256'] != args.checkpoint:
        return 'checkpoint_mismatch'
    if args.require_trajectory and not candidate['trajectory_available']:
        return 'missing_trajectory'
    frames = candidate['trajectory_frames'] or candidate['steps']
    if args.min_frames and frames < args.min_frames:
        return 'too_short'
    if candidate['failure_mode'] in args.exclude_failure_modes:
        return 'excluded_failure_mode'
    if candidate['scene_seed'] in args.exclude_seeds:
        return 'excluded_seed'
    return None


def apply_mode(pool, args):
    """Pick the survivors of the requested Best-of-N rule from an already sorted pool."""
    if args.mode == 'success':
        return [c for c in pool if c['reward']['success']]
    if args.mode == 'threshold':
        return [c for c in pool if c['score'] >= args.min_score]
    if args.mode == 'top_k':
        above = [c for c in pool if c['score'] >= args.min_score]
        return above[:args.top_k]
    # per_seed_best is the N>1 case: several rollouts of one seed, keep the best.
    best = {}
    for candidate in pool:
        best.setdefault(candidate['scene_seed'], candidate)
    return sorted(best.values(), key=sort_key)


def summarize(candidates):
    scores = sorted(c['score'] for c in candidates)
    if not scores:
        return {'episodes': 0}
    return {
        'episodes': len(scores),
        'successes': sum(1 for c in candidates if c['reward']['success']),
        'with_trajectory': sum(1 for c in candidates if c['trajectory_available']),
        'score_min': scores[0],
        'score_median': statistics.median(scores),
        'score_max': scores[-1],
        'score_mean': round(statistics.fmean(scores), 2),
    }


def build_selection(args):
    runs = find_runs(args.runs)
    candidates = discover(runs)
    if not candidates:
        raise ValueError(f'No {EPISODE_STATUS} found under {", ".join(map(str, runs))}')
    candidates.sort(key=sort_key)

    pool, rejected = [], []
    seen = {}
    for candidate in candidates:
        reason = hard_reject(candidate, args)
        if reason is None:
            key = dedup_key(candidate)
            previous = seen.get(key)
            if previous is not None:
                reason = 'duplicate'
                # Same seed and same checkpoint should replay identically. If the
                # scores differ, inference is not deterministic and Best-of-N has
                # to sample N rollouts per seed instead of reusing scored ones.
                if previous['score'] != candidate['score']:
                    args.determinism_warnings.append(
                        {'seed': candidate['scene_seed'], 'kept': previous['score'],
                         'dropped': candidate['score'], 'kept_path': previous['path'],
                         'dropped_path': candidate['path']})
            else:
                seen[key] = candidate
        if reason is None:
            pool.append(candidate)
        else:
            rejected.append({**candidate, 'reject_reason': reason})

    selected = apply_mode(pool, args)
    if args.max_selected:
        overflow = selected[args.max_selected:]
        selected = selected[:args.max_selected]
    overflow_ids = {id(c) for c in overflow} if args.max_selected else set()
    selected_ids = {id(c) for c in selected}
    rejected.extend({**c, 'reject_reason': 'above_limit' if id(c) in overflow_ids else 'not_selected'}
                    for c in pool if id(c) not in selected_ids)
    rejected.sort(key=lambda c: (c['reject_reason'], sort_key(c)))

    checkpoints = sorted({c['checkpoint_sha256'] for c in selected if c['checkpoint_sha256']})
    warnings = list(args.determinism_warnings)
    if len(checkpoints) > 1:
        warnings.append({'mixed_checkpoints': checkpoints,
                         'note': 'Fine-tuning on rollouts from several policies blends '
                                 'different behaviour distributions; prefer one checkpoint.'})
    if selected and not any(c['trajectory_available'] for c in selected):
        warnings.append({'no_trajectories': True,
                         'seeds': [c['scene_seed'] for c in selected],
                         'note': 'Selection is score-only. Re-run these seeds with '
                                 '--save-trajectory before converting to a dataset.'})
    if not selected:
        warnings.append({'empty_selection': True,
                         'note': 'No episode satisfied the criteria; loosen --min-score '
                                 'or --mode, or check --require-trajectory.'})

    return {
        'schema_version': SCHEMA_VERSION,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'criteria': {
            'runs': [str(run) for run in runs],
            'mode': args.mode,
            # json.dump would emit a bare -Infinity, which is not valid JSON.
            'min_score': args.min_score if args.min_score != float('-inf') else None,
            'top_k': args.top_k,
            'max_selected': args.max_selected,
            'min_frames': args.min_frames,
            'policy_type': args.policy_type,
            'checkpoint': args.checkpoint,
            'require_trajectory': args.require_trajectory,
            'exclude_failure_modes': sorted(args.exclude_failure_modes),
            'exclude_seeds': sorted(args.exclude_seeds),
        },
        'pool': summarize(candidates),
        'eligible_after_filters': len(pool),
        'totals': {
            'selected': len(selected),
            'selected_steps': sum(c['steps'] for c in selected),
            'selected_frames': sum(c['trajectory_frames'] for c in selected),
            'rejected': len(rejected),
            'rejected_by_reason': {reason: sum(1 for c in rejected if c['reject_reason'] == reason)
                                   for reason in REJECT_REASONS + ('not_selected', 'above_limit')},
        },
        'selected_seeds': [c['scene_seed'] for c in selected],
        'selected_summary': summarize(selected),
        'selected': selected,
        'rejected': rejected,
        'warnings': warnings,
    }


def print_table(title, rows, limit):
    print(f'\n{title} ({len(rows)})')
    if not rows:
        return
    header = (f'{"seed":>5} {"score":>6} {"pay":>5} {"spill":>6} {"maxb":>5} '
              f'{"steps":>5} {"ok":>5} {"failure":<18} {"traj":>5}  episode')
    print(header)
    print('-' * len(header))
    for candidate in rows[:limit]:
        reward = candidate['reward']
        print(f'{str(candidate["scene_seed"]):>5} {candidate["score"]:>6} '
              f'{reward["payload_particles"]:>5} {reward["spillage_particles"]:>6} '
              f'{reward["max_bucket_particles"]:>5} {candidate["steps"]:>5} '
              f'{str(reward["success"]):>5} {str(candidate["failure_mode"]):<18} '
              f'{candidate["trajectory_frames"] if candidate["trajectory_available"] else "-":>5}  '
              f'{Path(candidate["run"]).name}/{candidate["episode"]}')
    if len(rows) > limit:
        print(f'... {len(rows) - limit} more')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True,
                        help='Evaluation run directories (or a parent containing them).')
    parser.add_argument('--output', type=Path,
                        help='Write selection.json here. Without it the ranking is only printed.')
    parser.add_argument('--mode', choices=MODES, default='top_k')
    parser.add_argument('--min-score', type=int, default=None,
                        help='Lowest accepted score (payload - spillage). Required by threshold mode.')
    parser.add_argument('--top-k', type=int, default=None, help='Selection size for top_k mode.')
    parser.add_argument('--max-selected', type=int, default=None,
                        help='Cap the selection whatever the mode produced.')
    parser.add_argument('--min-frames', type=int, default=None,
                        help='Reject episodes shorter than this many executed steps.')
    parser.add_argument('--policy-type', help='Keep only this policy, e.g. smolvla or act.')
    parser.add_argument('--checkpoint', help='Keep only this exact checkpoint_sha256.')
    parser.add_argument('--require-trajectory', action='store_true',
                        help='Keep only episodes whose frames were already captured.')
    parser.add_argument('--exclude-failure-modes', default='',
                        help='Comma-separated failure modes to drop, e.g. lost_payload.')
    parser.add_argument('--exclude-seeds', default='',
                        help='Comma-separated scene seeds to hold out for evaluation.')
    parser.add_argument('--print-limit', type=int, default=40)
    args = parser.parse_args(argv)

    if args.mode == 'threshold' and args.min_score is None:
        parser.error('--mode threshold requires --min-score')
    if args.mode == 'top_k' and args.top_k is None:
        parser.error('--mode top_k requires --top-k')
    for name in ('top_k', 'max_selected', 'min_frames', 'print_limit'):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f'--{name.replace(chr(95), chr(45))} must be positive')
    if args.min_score is None:
        args.min_score = float('-inf')
    args.exclude_failure_modes = {m.strip() for m in args.exclude_failure_modes.split(',') if m.strip()}
    args.exclude_seeds = {int(s) for s in args.exclude_seeds.split(',') if s.strip()}
    args.determinism_warnings = []
    return args


def main(argv=None):
    args = parse_args(argv)
    selection = build_selection(args)
    pool = selection['pool']
    print(f'Pool: {pool.get("episodes", 0)} episodes from {len(selection["criteria"]["runs"])} run(s); '
          f'{pool.get("successes", 0)} successful, {pool.get("with_trajectory", 0)} with frames; '
          f'score {pool.get("score_min")}/{pool.get("score_median")}/{pool.get("score_max")} '
          f'(min/median/max)')
    print(f'Eligible after filters: {selection["eligible_after_filters"]}')
    print_table('SELECTED', selection['selected'], args.print_limit)
    counts = {k: v for k, v in selection['totals']['rejected_by_reason'].items() if v}
    print(f'\nRejected: {selection["totals"]["rejected"]} {counts}')
    for warning in selection['warnings']:
        print(f'WARNING {json.dumps(warning)}', file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(selection, indent=2) + '\n')
        temporary.replace(args.output)
        print(f'\nWrote {args.output}')
        print(f'Seeds: {",".join(str(s) for s in selection["selected_seeds"])}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
