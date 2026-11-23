"""Entry point for the shared standalone GPU lease bootstrap."""

from pathlib import Path
import sys


def ensure(script,argv=None,device="cuda:0"):
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from analysis._gpu_lease import ensure as invoke
    return invoke(script,argv,device)
