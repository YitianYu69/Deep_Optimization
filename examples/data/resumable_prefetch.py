"""Replay seeded transforms with custom collation and cursor-preserving rebatching.

python examples/data/resumable_prefetch.py
python examples/data/resumable_prefetch.py --workers 2

This CPU example uses the built-in loader, without TorchData or downloads.
It snapshots after one batch of four, then restores the remaining samples in
batches of three with zero workers. The snapshot stays in memory; no files
are created. Model/optimizer recovery belongs to the checkpointing examples.
"""

import argparse
import json

import torch
from torch.utils.data import TensorDataset

from deepopt.data import ResumableDataLoader, SeededTransformDataset
from deepopt.logger import log_output


def augment(sample, *, generator):
    index, features = sample
    return index, features + .01 * torch.randn(features.shape, generator=generator)


def collate_samples(samples):
    # Keep definitions at module scope so spawn workers can import them.
    indices, features = zip(*samples)
    return {"indices": torch.stack(indices), "features": torch.stack(features)}


def make_loader(batch_size, workers):
    base = TensorDataset(torch.arange(13), torch.arange(26, dtype=torch.float32).reshape(13, 2) / 10)
    dataset = SeededTransformDataset(base, augment, seed=17, transform_id="noise-v1")
    options = ({"prefetch_factor": 2, "persistent_workers": True,
                "multiprocessing_context": "spawn"} if workers else {})
    return ResumableDataLoader(
        dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(31),
        num_workers=workers, collate_fn=collate_samples, **options,
    )


def snapshot_and_reference(workers):
    # The scope releases its iterator/workers before the replacement is created.
    loader = make_loader(batch_size=4, workers=workers)
    iterator = iter(loader)
    consumed = next(iterator)
    snapshot = loader.state_dict()
    # Finish this copy only to obtain the uninterrupted reference values.
    remaining = list(iterator)
    return consumed, snapshot, remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0,
                        help="Initial worker count; restore always uses zero workers")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")

    consumed, snapshot, expected = snapshot_and_reference(args.workers)
    restored = make_loader(batch_size=3, workers=0)
    restored.load_state_dict(snapshot)
    actual = list(restored)
    for key in ("indices", "features"):
        torch.testing.assert_close(torch.cat([batch[key] for batch in actual]),
                                   torch.cat([batch[key] for batch in expected]), rtol=0, atol=0)
    log_output(json.dumps({
        "initial_workers": args.workers,
        "resumed_workers": 0,
        "consumed_indices": consumed["indices"].tolist(),
        "remaining_indices": torch.cat([batch["indices"] for batch in actual]).tolist(),
        "resumed_batch_sizes": [len(batch["indices"]) for batch in actual],
        "replay_matches": True,
    }))


if __name__ == "__main__":
    main()
