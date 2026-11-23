"""Host residency is explicit and cannot weaken the native CUDA saved ABI."""

from copy import deepcopy
import unittest

from deepopt.compiler.saved_state import classify_saved_state
from tests.compiler import test_saved_activation_contracts as saved_contracts


def _host_region():
    region = saved_contracts.SavedStateClassifierTests().region()
    def cuda(value):
        if isinstance(value, dict):
            return {key: ("cuda:0" if key in ("device", "logical_device") else cuda(child))
                    for key, child in value.items()}
        if isinstance(value, list):
            return [cuda(child) for child in value]
        if isinstance(value, tuple):
            return tuple(cuda(child) for child in value)
        return value
    region = cuda(region)
    for value in region["saved_state"]:
        strides, size = [], 1
        for dimension in reversed(value["shape"]):
            strides.insert(0, size)
            size *= dimension
        value["stride"] = strides
    components = [{**value, "saved_index": index} for index, value in enumerate(region["saved_state"])]
    region.update(saved_state_schema_version=4, packed_activation_offload={"transport_envelope_version": 1,
        "plan": {"schema_version": 1, "bundles": [{"bundle_id": "affine-0", "components": components}]}})
    rows = []
    for index, value in enumerate(region["saved_state"]):
        abi = {key: deepcopy(value[key]) for key in ("shape", "dtype", "device", "stride")}
        rows.append({**value, "device": "cpu", "presence": "present", "transport": dict(version=1, generation=1,
            bundle_id="affine-0", saved_index=index, residency="host_pinned",
            producer_abi=abi, consumer_abi=deepcopy(abi))})
    region["runtime_saved_state"] = dict(metadata_only=True, bindings=rows)
    return region


class PackedOffloadSavedStateTests(unittest.TestCase):
    def test_host_snapshot_and_CUDA_consumer_are_distinct_contracts(self):
        rows = classify_saved_state(_host_region(), compressed=True)
        self.assertEqual([row["category"] for row in rows], ["packed_payload", "codec_metadata", "codec_metadata"])
        self.assertTrue(all(row["storage_device"] == "cpu" and row["consumer_device"] == "cuda:0" for row in rows))

    def test_transport_cannot_hide_wrong_snapshot_or_consumer_metadata(self):
        for target, key, value in (("snapshot", "shape", [1]), ("snapshot", "dtype", "torch.float64"),
                                   ("snapshot", "stride", [1, 2048]), ("consumer", "stride", [1, 2048]),
                                   ("consumer", "device", "cpu"), ("consumer", "shape", [1]),
                                   ("transport", "version", True), ("transport", "saved_index", -1),
                                   ("transport", "saved_index", 1), ("transport", "bundle_id", "other"),
                                   ("transport", "generation", 0)):
            with self.subTest(target=target, key=key):
                region = _host_region()
                observation = region["runtime_saved_state"]["bindings"][0]
                node = observation if target == "snapshot" else observation["transport"] if target == "transport" else observation["transport"]["consumer_abi"]
                node[key] = value
                with self.assertRaises(AssertionError):
                    classify_saved_state(region, compressed=True)

    def test_transport_requires_versioned_offload_report_and_envelope(self):
        for schema in (1, 2, 3):
            region = _host_region()
            region["saved_state_schema_version"] = schema
            with self.assertRaisesRegex(AssertionError, "schema 4"):
                classify_saved_state(region, compressed=True)
        region = _host_region()
        region.pop("packed_activation_offload")
        with self.assertRaisesRegex(AssertionError, "Incompatible"):
            classify_saved_state(region, compressed=True)


if __name__ == "__main__":
    unittest.main()
