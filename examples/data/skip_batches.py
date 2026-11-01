"""Skip index batches without fetching skipped samples; optional worker loading.

python examples/data/skip_batches.py --workers 2
This is manual index replay, not model/worker-state checkpoint recovery.
"""

import argparse

import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, TensorDataset

from deepopt.data import SkipBatchSampler, SkipDataLoader
from deepopt.logger import init_logger

logger = init_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    dataset = TensorDataset(torch.arange(10))
    batch_sampler = SkipBatchSampler(BatchSampler(SequentialSampler(dataset), 2, drop_last=False), skip_batches=3)
    explicit = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.workers)
    convenient = SkipDataLoader(dataset, batch_size=2, skip_batches=3, num_workers=args.workers)
    for name, loader in (("sampler", explicit), ("loader", convenient)):
        logger.info("%s remaining indices: %s", name, torch.cat([batch[0] for batch in loader]).tolist())
        logger.info("%s next epoch indices: %s", name, torch.cat([batch[0] for batch in loader]).tolist())


if __name__ == "__main__":
    main()
