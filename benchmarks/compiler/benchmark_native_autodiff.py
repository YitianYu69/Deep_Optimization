"""Qualify native AutoDiff ownership, fusion, runtime and saved-storage costs.

Uses the shared isolated same-recipe BF16 harness and includes the former
compiler training programs as an explicitly labeled migration baseline.
"""

from pathlib import Path
import sys


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(argument == "--fixtures" or argument.startswith("--fixtures=")
           or argument == "--fusion-worker" for argument in arguments):
        from isolated_fusion import main as run

        return run(arguments)
    from benchmark_bf16 import main as run

    return run(["--require-native-autodiff", "--include-legacy", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
