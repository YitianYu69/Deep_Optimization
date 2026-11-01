"""Optional callback scheduling; the trainer owns update counters, not I/O."""


class EveryNSteps:
    """Run callback(trainer, event) every N completed optimizer updates.

    Skipped updates do not trigger callback-related device synchronization.
    Pass this to fit(step_callbacks=[...]). Plain callable callbacks still run
    on every update. The global update count survives checkpoint restoration.
    """

    def __init__(self, callback, every: int):
        if not callable(callback):
            raise TypeError("callback must be callable")
        if type(every) is not int or every < 1:
            raise ValueError("every must be a positive integer")
        self.callback, self.every = callback, every

    def should_run_step(self, optimizer_step: int) -> bool:
        return optimizer_step % self.every == 0

    def __call__(self, trainer, event):
        self.callback(trainer, event)
