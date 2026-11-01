"""Batch loading; worker-state support is imported only when requested."""

from importlib import import_module

__all__ = ["ResumableDataLoader", "SkipBatchSampler", "SkipDataLoader", "StatefulDataLoader", "build_loaders", "DistributedEvalSampler",
           "SeededTransformDataset", "sample_seed"]
_MODULES = {"ResumableDataLoader": ".resumable", "SkipBatchSampler": ".skip",
            "SkipDataLoader": ".skip", "StatefulDataLoader": ".stateful", "build_loaders": ".loaders",
            "DistributedEvalSampler": ".loaders", "SeededTransformDataset": ".seeded", "sample_seed": ".seeded"}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
