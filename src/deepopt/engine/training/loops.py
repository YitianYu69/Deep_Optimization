"""Batch iteration and metric collection, without run state or file I/O."""

from time import perf_counter

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset
from deepopt.distributed import _process as distributed

from deepopt.engine.reporting.metric_lifecycle import MetricLifecycle
from deepopt.engine.execution.optimization import OptimizationWindow


class EpochRunner:
    def __init__(self, runtime, steps, *, accumulation_steps, loss_strategy, metric_factory=None):
        self.runtime, self.steps = runtime, steps
        self.accumulation_steps, self.loss_strategy = accumulation_steps, loss_strategy
        self.metric_factory = metric_factory
        self.metrics = MetricLifecycle(runtime.device, observe=runtime.observe)

    def new_metrics(self):
        return self.metrics.create(self.steps, self.metric_factory)

    def train(self, loader, progress, *, on_update, synchronize_updates=True, on_window_start=None) -> dict:
        window = OptimizationWindow(self.runtime, accumulation_steps=self.accumulation_steps,
                                    loss_strategy=self.loss_strategy)
        self.runtime.model.train()
        self.runtime.zero_grad()
        self.runtime.synchronize()
        started = perf_counter()
        previous_seconds, callback_seconds = progress.seconds, 0.0
        remaining = getattr(loader, "remaining_batches", None)
        # Do not look ahead: fetching a future batch would commit the loader's
        # delivered cursor before a checkpoint for the current update.
        total = None
        if not callable(remaining) and (isinstance(loader, (list, tuple)) or
                (isinstance(loader, DataLoader) and not isinstance(loader.dataset, IterableDataset)
                 and type(loader.batch_sampler) is BatchSampler and not hasattr(loader, "load_state_dict"))):
            total = len(loader)

        def after_update():
            nonlocal callback_seconds
            needs_sync = synchronize_updates() if callable(synchronize_updates) else synchronize_updates
            if not needs_sync:
                # Counters still advance at every optimizer boundary. With no
                # step callbacks, only the final epoch timing needs a CUDA wait.
                on_update(progress)
                return
            self.runtime.synchronize()
            progress.seconds = previous_seconds + perf_counter() - started - callback_seconds
            callback_start = perf_counter()
            try:
                on_update(progress)
            finally:
                callback_seconds += perf_counter() - callback_start

        try:
            for index, batch in enumerate(distributed.training_batches(loader), start=1):
                left = remaining() if callable(remaining) else (total - index if total is not None else None)
                should_sync = (left is None or left == 0 or window.pending_batches + 1 == self.accumulation_steps)
                with self.runtime.core.accumulation_context(synchronize=should_sync):
                    with distributed.collective_errors("training step"):
                        if window.pending_batches == 0 and on_window_start is not None:
                            on_window_start()
                        output = self.runtime.compute(batch, self.steps.training_step)
                    if self.runtime.diagnostics is None:
                        self.runtime.observe("loss", output.loss, name="loss")
                    with distributed.collective_errors("metric update"):
                        progress.metrics.update(output)
                    updated = window.backward(output)
                if progress.samples is not None:
                    progress.samples = (None if output.sample_count is None else
                                        progress.samples + output.sample_count)
                progress.batches += 1
                # Drop losses/predictions before callbacks or the next forward;
                # a previous batch must not keep its activations alive here.
                del output
                if updated:
                    after_update()
            if progress.batches == 0:
                raise ValueError("Training loader is empty")
            if window.flush():
                after_update()
            results = self.metrics.compute(progress.metrics, observe=self.runtime.observe)
        finally:
            self.runtime.zero_grad()
        self.runtime.synchronize()
        elapsed = previous_seconds + perf_counter() - started - callback_seconds
        progress.seconds = elapsed
        elapsed = distributed.reduce_number(elapsed, torch.distributed.ReduceOp.MAX)
        throughput = ({} if "samples" not in results else
                      {"samples_per_second": results["samples"] / max(elapsed, 1e-12)})
        return {**results, "seconds": elapsed, **throughput}

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        model = self.runtime.model
        previous_modes = {module: module.training for module in model.modules()}
        batches = 0
        error = None
        try:
            with self.runtime.core.check_errors("validation setup"):
                metrics = self.new_metrics()
                model.eval()
            self.runtime.sync_buffers()
            for batch in loader:
                output = self.runtime.compute(batch, self.steps.validation_step)
                self.runtime.observe("loss", output.loss, name="loss")
                metrics.update(output)
                del output
                batches += 1
        except Exception as exc:
            error = exc
        finally:
            # Restore mixed modes, including intentionally frozen BatchNorm.
            for module, training in previous_modes.items():
                module.training = training
        # Validation shards may be unequal or empty. Coordinate once, after
        # local iteration, before any rank enters metric collectives.
        try:
            with self.runtime.core.check_errors("validation"):
                if error is not None:
                    raise error
        except Exception:
            self.runtime.invalidate()
            raise
        if distributed.reduce_number(batches) == 0:
            raise ValueError("Validation loader is empty")
        return self.metrics.compute(metrics, observe=self.runtime.observe)
