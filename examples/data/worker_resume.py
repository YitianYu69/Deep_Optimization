"""Optional worker-aware step resume: pip install -e '.[workers]'.

python examples/data/worker_resume.py
This example uses two workers and temporary checkpoints; stochastic dataset
state is captured explicitly so skipped worker RNG draws are not guessed.
"""

import gc
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from deepopt import Checkpoint, EveryNSteps, Trainer
from deepopt.data import StatefulDataLoader
from deepopt.logger import init_logger

logger = init_logger(__name__)


class NoisyData(Dataset):
    def __len__(self):
        return 17

    def __getitem__(self, index):
        inputs = torch.tensor([index / 17, 1.0]) + torch.randn(2) * 0.01
        return inputs, inputs.sum().reshape(1)

    def state_dict(self):
        return {"rng": torch.get_rng_state()}

    def load_state_dict(self, state):
        torch.set_rng_state(state["rng"])


class SavedInterruption(Exception):
    pass


def components():
    torch.manual_seed(7)
    loader = StatefulDataLoader(NoisyData(), batch_size=4, num_workers=2, multiprocessing_context="spawn",
                                generator=torch.Generator().manual_seed(17))
    model = nn.Linear(2, 1)
    trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=0.01), loss_fn=F.mse_loss)
    return trainer, loader


def interrupt_after_save(trainer, event):
    raise SavedInterruption()


def interrupted_run(directory):
    trainer, loader = components()
    checkpoint = Checkpoint(directory, every_steps=2)
    try:
        trainer.fit(loader, epochs=1, callbacks=[checkpoint], progress_bar=False,
                    step_callbacks=[EveryNSteps(interrupt_after_save, 2)])
    except SavedInterruption:
        logger.info("Saved two updates; reconstructing trainer and workers")


def main():
    with tempfile.TemporaryDirectory(prefix="deepopt-workers-") as directory:
        interrupted_run(Path(directory))
        gc.collect()  # Release interrupted loader workers before reconstructing.
        trainer, loader = components()
        checkpoint = Checkpoint(directory, every_steps=2)
        checkpoint.restore(trainer, loader)
        trainer.fit(loader, epochs=1, callbacks=[checkpoint], progress_bar=False)
        logger.info("Completed epoch: %s", checkpoint.history[-1])


if __name__ == "__main__":
    main()
