"""Connect a policy entrypoint to the parent benchmark's shared Isaac runtime."""
import sys

from ..common.paths import PROJECT_ROOT


def main(policy_type, argv=None):
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from eval_smolvla_success import main as evaluate

    original_argv = sys.argv
    arguments = sys.argv[1:] if argv is None else argv
    # The shared runtime passes these same flags to its worker entrypoint.
    sys.argv = [original_argv[0], '--policy-type', policy_type, *arguments]
    try:
        evaluate()
    finally:
        sys.argv = original_argv
