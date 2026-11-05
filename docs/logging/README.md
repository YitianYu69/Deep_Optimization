# Logging

[Documentation index](../README.md) · [Runnable recipe](../../examples/observability/logging_recipe.py)

Use a normal Python logger in any recipe or module:

```python
from deepopt.logger import init_logger

logger = init_logger(__name__)
logger.info("Training with batch size %s", batch_size)
logger.debug("Detailed diagnostic: %s", details)
```

The API and compact level/time/source format are inspired by
[vLLM's logger](https://github.com/vllm-project/vllm/blob/main/vllm/logger.py).
DeepOpt implements its own scoped file routing; vLLM is not a dependency.
Messages go to stderr, with terminal color when available. Every line of a
multiline message or traceback gets a prefix. Files are plain UTF-8 text.

Run the offline CPU recipe from the repository root after installing the package:

```bash
python examples/observability/logging_recipe.py --log-dir runs/logging-demo --epochs 2
```

It retains a configuration-named `.log` file in the requested directory and
prints that path through the logger. It creates no checkpoint or structured
training-history files. Repeating identical arguments appends another session;
choose a new directory when you want independent artifacts.

## Shared defaults without implicit files

`load_config()` can supply logging policy through a `logging` table:

```toml
[logging]
level = "WARNING"
file_level = "DEBUG"
```

`logging_session(config=None)` inherits current metadata and levels, but its
default `directory=None` creates **no file**. Pass a destination explicitly:

```python
from deepopt import load_config
from deepopt.logger import logging_session

load_config("builtin:baseline")
with logging_session(directory="runs/my-recipe/logs", level="WARNING"):
    trainer.fit(train_loader, epochs=2)
```

Explicit level arguments override a session's metadata/configuration, which
overrides global defaults. A partial dictionary can also add arbitrary
JSON-compatible recipe metadata; it is not restricted to the experiment schema.
Full experiment/file sources select their own configuration. A session owns a
snapshot, so changing globals while it is active does not change its policy or
filename. Creating a session or asking `log_filename()` does not activate a
global configuration or create a file.

Outside a session, the shared logger uses the current configured console level.
When no level is configured, `DEEPOPT_LOGGING_LEVEL` (or INFO) applies. Explicit
`level=None` selects that environment/default behavior even if a global level
exists. File output still requires a directory or a later `session.bind(...)`.

## Save a recipe's logs

Wrap your recipe once. No trainer subclass, callback, or logger wrapper is needed:

```python
from deepopt.logger import init_logger, logging_session

logger = init_logger(__name__)
config = {
    "name": "regression",
    "training": {"epochs": 5, "batch_size": 32, "learning_rate": 0.001},
    "runtime": {"device": "cpu", "precision": "fp32"},
}
with logging_session(config, directory="logs") as session:
    trainer.fit(train_loader, epochs=config["training"]["epochs"])
    logger.info("Validation: %s", trainer.evaluate(validation_loader))
    logger.info("Log file: %s", session.path)
```

Use the same settings to construct your trainer and loaders. The logger accepts
any JSON-serializable mapping or dataclass instance; it neither prescribes a
model schema nor inspects tensors. Include all recipe settings you want to
identify. Keep credentials out of the configuration because the resolved
configuration is recorded at DEBUG level.

Use strings for non-JSON settings such as `torch.dtype`, `torch.device`, and
`Path` values in this logging metadata. In particular, an FP8 execution plan
contains dtype objects; record a JSON-friendly description or plan version
instead of passing that plan directly to `logging_session`.

The session includes framework logs, your `init_logger()` messages, and exceptions
that escape the context. It flushes each record and closes the file on exit,
including interruption. Trainer construction emits model parameter counts/percentages and current
model/optimizer tensor storage; construct it inside the session for file capture.
Direct `Trainer.fit()` emits start/epoch/end events but
creates no files by itself. Manual loops can use the same context and logger.
No logging is added to the per-tensor or backward hot path.

## File names and resume

`log_filename(config)` combines a sanitized experiment name, available batch
size/learning rate/precision, and the first 16 hex characters of a SHA-256 digest
of the full canonical JSON configuration. For example, the pattern is:

```text
regression-bs32-lr0.001-fp32-<configuration-digest>.log
```

Dictionary insertion order does not affect the name. Any setting change,
including the total epoch target, changes the digest. Identical settings append
to the same file, with session start/end markers; use separate directories for
independent runs. Filenames contain no user-supplied directory separators.

Configured `deepopt.run.train()` and CLI training automatically write
`RUN_DIR/logs/<configuration-name>.log`. Planning/restoration messages are
buffered until the actual selected batch size is known, then written into the
resolved configuration's log. Existing `metrics.jsonl`, `batch_size.jsonl`, and
`resume.jsonl` retain their structured formats and recovery behavior.

A resumed run with changed settings gets another file in the same `logs/`
directory. Evaluation appends to the file identified by its resolved config,
including device overrides. `inspect-config` prints resolved configuration and
does not create an implicit log directory.

When a configured run fails during reconstruction, restore, or memory planning
before opening its normal log, its buffered messages and traceback go to
`RUN_DIR/../logs/<requested-configuration-name>.log`. This preserves the existing
rule that a failed fresh preflight does not create a training output directory.
Argument parsing, unreadable checkpoints, and invalid configuration failures
before a logging session can be established are reported on stderr only.

## Verbosity and output

```bash
DEEPOPT_LOGGING_LEVEL=WARNING python examples/observability/logging_recipe.py
DEEPOPT_LOGGING_LEVEL=DEBUG python examples/observability/logging_recipe.py
DEEPOPT_LOGGING_COLOR=0 python examples/observability/logging_recipe.py
```

| Setting | Default | Behavior |
| --- | --- | --- |
| `DEEPOPT_LOGGING_LEVEL` | `INFO` | Console fallback when no configuration/explicit level is selected |
| `DEEPOPT_LOGGING_COLOR` | `auto` | `1` forces color; `0` disables it; auto uses a terminal check |
| `NO_COLOR` | unset | A nonempty value disables terminal color |
| `DEEPOPT_LOGGING_PREFIX` | empty | Adds a console prefix, such as `[experiment] ` |
| `logging_session(..., level="WARNING")` | `logging.level`, then environment/INFO | Override the console threshold for this context |
| `logging_session(..., file_level="INFO")` | `logging.file_level`, then `DEBUG` | File threshold, independent of console verbosity |

With the default file level, an explicitly opened file captures DEBUG diagnostics even when the console uses INFO.
Checkpoint-save messages use DEBUG. Standard logger methods (`warning`, `error`,
`exception`, etc.) work normally, including lazy `%s` formatting.

Machine-readable command results use a separate helper:

```python
import json
from deepopt.logger import log_output

log_output(json.dumps({"epoch": 1, "loss": 0.5}))
```

`log_output` preserves stdout text and also records it in the active log file.
It accepts print-style `sep`, `end`, `file`, and `flush` options. Without a session
it only writes its output stream. Console log levels do not suppress command
results. Existing examples route their result output through this helper;
the logging recipe demonstrates persistent capture explicitly.

Tqdm remains a terminal display. Diagnostic messages cooperate with its progress
bars. File logs contain completed-epoch metrics instead of carriage-return
redraws. `progress_bar=False` / CLI `--no-progress` hides the bar; the console
level controls ordinary diagnostic messages separately.

## Ownership and limits

Only the `deepopt` logger namespace is configured. `init_logger(__name__)` maps
external recipe names into that namespace. Root handlers and unrelated library
loggers remain unchanged. Nested sessions restore the outer routing on exit;
each thread starting its own session gets its own destination. Context does not
automatically propagate to newly created threads or worker processes. Do not
share a run directory between independent writers.

DeepOpt's router and `log_output()` emit only on global rank zero when a
distributed group is active (or `RANK` identifies a nonzero worker). A session
on other ranks has no file and `session.path` remains `None`. Run training and
checkpoint collectives on every rank; only reporting is filtered. For separate
per-rank diagnostics, explicitly own a different logger/file per rank instead
of expecting this router to capture all worker output.

This API does not intercept arbitrary `print()`, Python warnings, third-party
stdout/stderr, subprocess output, or native CUDA messages. Use `init_logger()`
or `log_output()` for recipe output that belongs in the file. It does not log
model tensors, activations, or every optimizer update automatically; observers
and callbacks can explicitly report those when needed. Logs are append-only
diagnostics, not checkpoint state, and are not rolled back during resume.

Advanced orchestration can start `logging_session(config, directory=None)` and
call `session.bind(directory, config=resolved_config)` once after configuration
resolution. `failure_directory` optionally provides a destination for failures
before binding. Ordinary recipes should pass a directory directly.
