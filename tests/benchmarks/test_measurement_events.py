"""Event allocation/priming stays outside throughput samples without CUDA."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.analysis import measurement


class FakeEvent:
    def __init__(self, owner, identifier):
        self.owner = owner
        self.identifier = identifier
        self.synchronized = False

    def record(self):
        self.synchronized = False
        self.owner.log.append(("record", self.identifier))

    def synchronize(self):
        self.synchronized = True
        self.owner.log.append(("event_sync", self.identifier))

    def elapsed_time(self, other):
        if not other.synchronized:
            raise AssertionError("Elapsed time read before end-event synchronization")
        self.owner.log.append(("elapsed", self.identifier, other.identifier))
        return 6.0


class FakeCuda:
    def __init__(self, log):
        self.log = log
        self.events = []

    def Event(self, *, enable_timing):
        if enable_timing is not True:
            raise AssertionError("Event timing must be enabled")
        identifier = len(self.events)
        self.log.append(("allocate", identifier))
        event = FakeEvent(self, identifier)
        self.events.append(event)
        return event

    def synchronize(self, device):
        self.log.append(("device_sync", device.type))


class MeasurementEventTests(unittest.TestCase):
    def run_measurement(self, *, device_type="cuda", warmup=2, iterations=3, repeats=5):
        log = []
        cuda = FakeCuda(log)
        torch = SimpleNamespace(cuda=cuda)
        device = SimpleNamespace(type=device_type)
        calls = {"count": 0}

        def call():
            log.append(("call", calls["count"]))
            calls["count"] += 1

        # Each repeat has 1ms host and 3ms synchronized wall per iteration.
        clock = iter(value for index in range(repeats)
                     for value in ((index, index + iterations * .001,
                                    index + iterations * .0012, index + iterations * .0028,
                                    index + iterations * .003) if device_type == "cuda" else
                                   (index, index + iterations * .001, index + iterations * .003)))

        def now():
            log.append(("clock",))
            return next(clock)

        with patch.object(measurement, "perf_counter", side_effect=now):
            result = measurement.measure(call, torch, device, warmup, iterations, repeats)
        return result, log, cuda

    def test_exactly_two_events_are_allocated_once_for_all_repeats(self):
        _result, log, cuda = self.run_measurement(repeats=5)
        self.assertEqual(len(cuda.events), 2)
        self.assertEqual([row for row in log if row[0] == "allocate"],
                         [("allocate", 0), ("allocate", 1)])
        self.assertEqual(log.count(("record", 0)), 6)
        self.assertEqual(log.count(("record", 1)), 6)
        self.assertEqual(log.count(("event_sync", 1)), 6)
        self.assertEqual(log.count(("elapsed", 0, 1)), 5)

    def test_both_handles_are_primed_and_synchronized_before_first_sample(self):
        _result, log, _cuda = self.run_measurement(warmup=2)
        first_clock = log.index(("clock",))
        first_measured_call = log.index(("call", 2))
        priming_end = log.index(("event_sync", 1))
        self.assertLess(log.index(("allocate", 0)), log.index(("record", 0)))
        self.assertLess(log.index(("allocate", 1)), log.index(("record", 1)))
        self.assertLess(log.index(("record", 0)), priming_end)
        self.assertLess(log.index(("record", 1)), priming_end)
        self.assertLess(priming_end, first_clock)
        self.assertLess(priming_end, first_measured_call)
        self.assertLess(log.index(("call", 1)), log.index(("allocate", 0)))

    def test_sync_occurs_at_sample_boundaries_and_not_between_iterations(self):
        iterations, repeats = 3, 5
        _result, log, _cuda = self.run_measurement(iterations=iterations, repeats=repeats)
        self.assertEqual(log.count(("device_sync", "cuda")), repeats + 1)
        # Exclude the first event-0 record, which initializes the event handle.
        starts = [index for index, row in enumerate(log) if row == ("record", 0)][1:]
        for start in starts:
            end = next(index for index in range(start + 1, len(log))
                       if log[index] == ("record", 1))
            interval = log[start + 1:end]
            self.assertEqual(sum(row[0] == "call" for row in interval), iterations)
            self.assertFalse(any(row[0] in ("allocate", "device_sync", "event_sync") for row in interval))
            self.assertEqual(log[end + 1:end + 5],
                             [("clock",), ("event_sync", 1), ("clock",), ("elapsed", 0, 1)])

    def test_raw_samples_and_three_metric_summaries_preserve_nonadditive_contract(self):
        result, _log, _cuda = self.run_measurement(iterations=3, repeats=5)
        for metric, expected in (("host_submission_ms", 1.0), ("cuda_stream_ms", 2.0), ("wall_ms", 3.0)):
            with self.subTest(metric=metric):
                value = result[metric]
                self.assertEqual(len(value["samples"]), 5)
                for sample in value["samples"]:
                    self.assertAlmostEqual(sample, expected)
                self.assertAlmostEqual(value["median"], expected)
                self.assertAlmostEqual(value["min"], expected)
                self.assertAlmostEqual(value["max"], expected)
                self.assertAlmostEqual(value["p95"], expected)
        self.assertIn("not additive", result["note"])
        self.assertIn("submission gaps", result["note"])
        self.assertAlmostEqual(result["host_end_event_record_ms"]["median"], .2)
        self.assertAlmostEqual(result["host_event_synchronize_ms"]["median"], 1.6)
        self.assertAlmostEqual(result["host_event_elapsed_query_ms"]["median"], .2)
        self.assertAlmostEqual(result["wall_minus_cuda_stream_ms"]["median"], 1.)

    def test_sample_telemetry_runs_after_timing_and_before_the_next_sample(self):
        log = []
        torch = SimpleNamespace(cuda=FakeCuda(log))
        observed = []
        def callback(index, metrics):
            self.assertEqual(log[-1], ("elapsed", 0, 1))
            observed.append((index, metrics))
        result = measurement.measure(lambda: None, torch, SimpleNamespace(type="cuda"),
                                     0, 1, 2, sample_callback=callback)
        self.assertEqual([index for index, _ in observed], [0, 1])
        self.assertEqual([m["wall_ms"] for _, m in observed], result["wall_ms"]["samples"])

    def test_cpu_timing_allocates_no_cuda_events_or_synchronization(self):
        result, log, cuda = self.run_measurement(device_type="cpu", warmup=1, iterations=2, repeats=4)
        self.assertEqual(cuda.events, [])
        self.assertFalse(any(row[0] in ("allocate", "record", "device_sync", "event_sync", "elapsed") for row in log))
        self.assertEqual(sum(row[0] == "call" for row in log), 9)
        self.assertIsNone(result["cuda_stream_ms"])
        self.assertEqual(len(result["wall_ms"]["samples"]), 4)
        self.assertEqual(len(result["host_submission_ms"]["samples"]), 4)
        self.assertAlmostEqual(result["wall_ms"]["median"], 3.0)
        self.assertAlmostEqual(result["host_submission_ms"]["median"], 1.0)


if __name__ == "__main__":
    unittest.main()
