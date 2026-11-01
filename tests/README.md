# Test groups

Run from the repository root after installing the package:

```bash
python -m unittest discover -s tests -v
```

| Folder | Coverage |
| --- | --- |
| `training/` | Fit/manual APIs, module hooks, runtime safety, progress and performance |
| `execution/` | Mode policies, reference-equivalent updates, compile behavior, transfers and profiler invariants |
| `checkpointing/` | Checkpoint durability, generator discovery, step resume and batch-size changes |
| `data/` | Loader construction, skipping, worker state and cursor-preserving rebatching |
| `distributed/` | Separate execution cores, real Gloo/torchrun training, sharding, rank-local recovery and NCCL when available |
| `metrics/` | Metric inputs, aggregation, saved state and optional counts |
| `optimization/` | Loss semantics, accumulation, strategies and optimizers |
| `memory/` | Storage accounting, fake estimation, planning and activation retention |
| `observability/` | Logging, numerical observers and their configuration |
| `application/` | Configuration, factories, configured runs and CLI |
| `onboarding/` | User workflows, published walkthrough and import compatibility |

Run one group or one file:

```bash
python -m unittest discover -s tests/memory -v
python -m unittest discover -s tests -p 'test_step_checkpoint.py' -v
```

Each group is a Python package so full discovery descends into it. CUDA and
optional TorchData tests skip when their prerequisites are unavailable; see the
[development guide](../docs/architecture/README.md) for those checks.

## White-box and black-box coverage

Use both perspectives for feature work. White-box tests inspect invariants,
state transitions, dispatch, ownership, failure boundaries and expensive or
forbidden operations. Black-box tests drive public interfaces and verify
observable outputs, parameter updates, artifacts, and restart behavior. Mocked
collectives belong only to white-box tests; they do not establish DDP correctness.

The review acceptance matrix is in `distributed/test_review_fixes_black_box.py`.
Set `DEEPOPT_EXTENDED_DDP=1` for four-rank Gloo coverage. It checks asymmetric
validation errors, update-window reduction counts, compiled mode changes, and
seeded crop/flip replay with persistent workers. Local torchrun tests use a
loopback rendezvous and terminate the whole worker group on timeout.

CI tiers are defined in [ci.yml](../.github/workflows/ci.yml): fast CPU checks,
worker/DDP checks, and opt-in GPU correctness/performance jobs. Syntax/name lint
checks cover the repository; strict typing currently targets the new policy
module. CUDA skips must not be reported as GPU validation.

| Area | White-box focus | Black-box focus |
| --- | --- | --- |
| Training and execution cores | Core selection, delegation, graph lifetime, failure state | Fit/manual updates, module hooks, stopping and scheduler behavior |
| Distributed | `distributed/test_cores_white_box.py` and `test_data_white_box.py`: core isolation, reducer boundaries, normalization, fake tracing and generator ownership | `distributed/test_ddp_black_box.py`: real Gloo/NCCL processes, torchrun, worker prefetch, global results, rank-local recovery |
| Data | Cursor/origin validation, dispatch versus delivery, generator ownership | Sample order, worker prefetch, custom collation, resume and rebatching |
| Checkpointing | Schema/topology validation, transactional writes, RNG restoration | Interrupted versus uninterrupted runs and public checkpoint/CLI recovery |
| Metrics | Registered state, synchronization, routing, detached ownership | Loss/statistic values over complete phases and distributed shards |
| Optimization | Divisors, gradient processing, optimizer ownership | Numerical updates against PyTorch reference computations |
| Memory | Storage identity, fake execution, allocation isolation | Public estimates/plans and applied batch-size decisions |
| Observability | Filtering, detachment, device retention, synchronization counts | Training failures, emitted records, logging and example workflows |
| Application and onboarding | Configuration validation, public imports, factory contracts | CLI commands, complete recipes and documented walkthroughs |

Run the two distributed perspectives independently with:

```bash
python -m unittest discover -s tests/distributed -p '*white_box.py' -v
python -m unittest discover -s tests/distributed -p '*black_box.py' -v
```
