"""CLI bootstrap shared by the independently executable GPU programs."""

from pathlib import Path
import sys


def run(main,script):
    directory=Path(script).resolve().parent
    root=directory if (directory/"analysis").is_dir() else directory.parent
    sys.path.insert(0,str(root))
    from analysis.scheduler import guarded_main
    return guarded_main(main,sys.argv[1:],script)


def ensure(script,argv=None,device="cuda:0"):
    directory=Path(script).resolve().parent
    root=directory if (directory/"analysis").is_dir() else directory.parent
    sys.path.insert(0,str(root))
    from analysis.scheduler import ensure_lease
    return ensure_lease(list(argv) if argv is not None else sys.argv[1:],script,device)
