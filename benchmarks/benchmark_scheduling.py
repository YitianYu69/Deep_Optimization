"""CPU controller-only overhead; excludes training, compilation and CUDA work."""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

from deepopt.scheduling import ScheduleController, LinearWarmup


def measure(updates, repeats, tensor):
    samples = []
    for _ in range(repeats):
        controller = ScheduleController({"coefficient": LinearWarmup(0, 1, updates)})
        if tensor:
            controller.tensor("coefficient")
        started = perf_counter()
        for step in range(1, updates + 1):
            controller.prepare_update(step)
            controller.complete_update(step)
        samples.append((perf_counter() - started) * 1e6 / updates)
        assert controller["coefficient"] == 1.
    return median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.updates < 1 or args.repeats < 1:
        parser.error("updates and repeats must be positive")
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "updates": args.updates, "repeats": args.repeats, "unit": "microseconds per update",
                      "python_coefficient": measure(args.updates, args.repeats, False),
                      "cpu_tensor_coefficient": measure(args.updates, args.repeats, True)}))


if __name__ == "__main__":
    main()
