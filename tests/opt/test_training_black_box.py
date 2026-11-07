"""FP8 training through the public trainer and durable checkpoint callback."""

from pathlib import Path
import tempfile
import unittest

from deepopt import Checkpoint
from tests.opt.fp8_support import InterruptedTraining, make_training
from tests.state_support import assert_state_equal


class FP8TrainingBlackBoxTests(unittest.TestCase):
    def test_step_and_epoch_resume_match_uninterrupted_fp8_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("full", "modules"):
                reference, loader = make_training(mode)
                reference.fit(loader, epochs=2, progress_bar=False, monitor=None)
                for boundary in ("step", "epoch"):
                    with self.subTest(mode=mode, boundary=boundary):
                        trainer, loader = make_training(mode)
                        checkpoint = Checkpoint(root / mode / boundary,
                                                every_steps=1 if boundary == "step" else 0)

                        def interrupt_step(current, event):
                            if current.state.optimizer_steps == 1:
                                raise InterruptedTraining()

                        def interrupt_epoch(current, row):
                            if current.state.completed_epochs == 1:
                                raise InterruptedTraining()

                        with self.assertRaises(InterruptedTraining):
                            trainer.fit(
                                loader, epochs=2, progress_bar=False, monitor=None,
                                callbacks=[checkpoint] + ([interrupt_epoch] if boundary == "epoch" else []),
                                step_callbacks=[interrupt_step] if boundary == "step" else [],
                            )
                        resumed, resumed_loader = make_training(mode)
                        restored = Checkpoint(root / mode / boundary,
                                              every_steps=1 if boundary == "step" else 0)
                        restored.restore(resumed, resumed_loader)
                        resumed.fit(resumed_loader, epochs=2, callbacks=[restored],
                                    progress_bar=False, monitor=None)
                        assert_state_equal(self, resumed.model.state_dict(), reference.model.state_dict())
                        assert_state_equal(self, resumed.optimizer.state_dict(), reference.optimizer.state_dict())
                        assert_state_equal(self, resumed.scheduler.state_dict(), reference.scheduler.state_dict())
                        self.assertEqual(resumed.state.optimizer_steps, 6)
                        self.assertEqual(resumed.state.completed_epochs, 2)


if __name__ == "__main__":
    unittest.main()
