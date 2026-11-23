"""Fusion observation requires the selected candidate's actual CUDA work."""

from copy import deepcopy
import unittest

from benchmarks.analysis.measurement import compiler_launch_evidence, fusion_candidate_evidence, pack_sharing_evidence


class FusionCandidateEvidenceTests(unittest.TestCase):
    def archived_driver_trace(self):
        # Compact excerpt of the supplied H100 smoke trace, with one common
        # timestamp origin subtracted. Triton's External id=514 points outside
        # the fusion range (519), but CUPTI correlation=88 remains intact.
        name = "deepopt.fusion::50ed99209a461b6011a5bf26"
        return {"traceEvents": [
            {"ph": "X", "cat": "user_annotation", "name": name,
             "pid": 2372143, "tid": 2372202, "ts": 0., "dur": 1938.44,
             "args": {"External id": 519}},
            {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernelEx",
             "pid": 2372143, "tid": 2372202, "ts": 1855.977, "dur": 21.38,
             "args": {"External id": 514, "correlation": 88}},
            {"ph": "X", "cat": "kernel", "name": "_bf16_gemm",
             "pid": 0, "tid": 7, "ts": 1874.329, "dur": 2.336,
             "args": {"External id": 514, "device": 0, "context": 1, "stream": 7, "correlation": 88}},
        ]}

    def trace_profile(self, trace, steps=1):
        return {"compiler_regions": compiler_launch_evidence([], steps, trace=trace)}

    def candidate(self, **changes):
        return {"candidate_id": "graph0:projection:forward:gemm_gelu", "selected": True,
                "phase": "train", "execution_phase": "forward",
                "profile_range": "deepopt.compiler.projection.forward.forward",
                "implementation_fingerprint": "gemm-source-sha256",
                "kernel_evidence": {"required_kernel_roles": {"gemm_activation": ["_bf16_gemm"]},
                                    "forbidden_kernel_symbols": ["standalone_gelu"]}, **changes}

    def profile(self, candidate, *symbols):
        return {"compiler_regions": {candidate["profile_range"]: {
            "range_calls_per_step": 1, "linked_cuda_launches_per_step": len(symbols), "kernel_names": list(symbols)}}}

    def qualify(self, candidate, profile):
        return fusion_candidate_evidence({"fusion_candidates": [candidate]}, profile)

    def test_unrelated_linked_db_kernel_cannot_qualify_any_activation_pattern(self):
        for role, symbol in (("gemm_activation", "_bf16_gemm"),
                             ("norm_activation", "_layer_norm_forward")):
            with self.subTest(role=role):
                candidate = self.candidate(kernel_evidence={"required_kernel_roles": {role: [symbol]}})
                # Even a db launch inside the right range is insufficient.
                observed = self.qualify(candidate, self.profile(candidate, "_bias_gradient_reduce"))
                self.assertFalse(observed["all_selected_observed"])
                self.assertEqual(observed["status"], "not_observed")
                self.assertEqual(observed["candidates"][0]["missing_kernel_roles"], [role])

    def test_other_callsite_or_phase_with_the_right_symbol_remains_unknown(self):
        candidate = self.candidate()
        other = self.candidate(profile_range="deepopt.compiler.other.backward.dx")
        evidence = self.qualify(candidate, self.profile(other, "_bf16_gemm"))
        self.assertEqual(evidence["status"], "unknown")
        self.assertEqual(evidence["candidates"][0]["reason"], "candidate_range_missing")

    def test_every_declared_norm_pass_must_execute_but_alternative_routes_are_allowed(self):
        candidate = self.candidate(kernel_evidence={"required_kernel_roles": {
            "statistics": ["_norm_stats", "_norm_chunk_stats"],
            "activation_apply": ["_norm_apply_activation"],
            "affine_finish": ["_norm_affine_finish"]}})
        complete = self.profile(candidate, "_norm_chunk_stats", "_norm_apply_activation", "_norm_affine_finish")
        evidence = self.qualify(candidate, complete)
        self.assertTrue(evidence["all_selected_observed"])
        self.assertEqual(evidence["candidates"][0]["observed_cuda_launches"], 3)
        self.assertEqual(evidence["candidates"][0]["execution_phase"], "forward")
        incomplete = deepcopy(complete)
        incomplete["compiler_regions"][candidate["profile_range"]]["kernel_names"].remove("_norm_affine_finish")
        self.assertFalse(self.qualify(candidate, incomplete)["all_selected_observed"])

    def test_missing_profile_attribution_or_implementation_contract_remains_unknown(self):
        candidate = self.candidate()
        cases = [(candidate, None), (candidate, {}),
                 (self.candidate(implementation_fingerprint=None), self.profile(candidate, "_bf16_gemm")),
                 (self.candidate(kernel_evidence={}), self.profile(candidate, "_bf16_gemm")),
                 (self.candidate(candidate_id=None), self.profile(candidate, "_bf16_gemm"))]
        for selected, profile in cases:
            with self.subTest(candidate=selected, profile=profile):
                evidence = self.qualify(selected, profile)
                self.assertEqual(evidence["status"], "unknown")
                self.assertFalse(evidence["all_selected_observed"])

    def test_remaining_standalone_work_rejects_positive_observation(self):
        candidate = self.candidate()
        evidence = self.qualify(candidate, self.profile(candidate, "_bf16_gemm", "standalone_gelu"))
        self.assertEqual(evidence["status"], "not_observed")
        self.assertEqual(evidence["candidates"][0]["reason"], "standalone_work_remaining")

        candidate["kernel_evidence"]["removed_profile_ranges"] = ["deepopt.compiler.projection.forward.gelu"]
        profile = self.profile(candidate, "_bf16_gemm")
        self.assertTrue(self.qualify(candidate, profile)["all_selected_observed"])
        profile["compiler_regions"][candidate["kernel_evidence"]["removed_profile_ranges"][0]] = {
            "linked_cuda_launches_per_step": 1, "kernel_names": ["standalone_gelu"]}
        self.assertFalse(self.qualify(candidate, profile)["all_selected_observed"])
        profile["compiler_regions"][candidate["kernel_evidence"]["removed_profile_ranges"][0]] = {
            "linked_cuda_launches_per_step": None, "kernel_names": []}
        evidence = self.qualify(candidate, profile)
        self.assertEqual(evidence["status"], "unknown")
        self.assertEqual(evidence["candidates"][0]["reason"], "standalone_work_attribution_missing")

    def test_unfused_and_partially_observed_graphs_do_not_report_qualified_fusion(self):
        candidate = self.candidate(selected=False)
        evidence = self.qualify(candidate, self.profile(candidate, "_bf16_gemm"))
        self.assertEqual(evidence["status"], "unfused")
        self.assertFalse(evidence["all_selected_observed"])
        self.assertEqual(evidence["candidates"][0]["observation"], "not_selected")
        first = self.candidate()
        second = self.candidate(candidate_id="other", profile_range="deepopt.compiler.other.forward.forward")
        evidence = fusion_candidate_evidence({"fusion_candidates": [first, second]}, self.profile(first, "_bf16_gemm"))
        self.assertEqual(evidence["observed_candidates"], 1)
        self.assertFalse(evidence["all_selected_observed"])

    def test_duplicate_candidate_identity_cannot_reuse_one_observation(self):
        candidate = self.candidate()
        evidence = fusion_candidate_evidence({"fusion_candidates": [candidate, deepcopy(candidate)]},
                                             self.profile(candidate, "_bf16_gemm"))
        self.assertEqual(evidence["status"], "unknown")
        self.assertEqual(evidence["observed_candidates"], 0)
        self.assertTrue(all(item["reason"] == "duplicate_candidate_id" for item in evidence["candidates"]))

    def test_shared_pack_requires_one_encoding_launch_with_separate_cse_evidence(self):
        candidate = self.candidate(profile_range="deepopt.pack::encoding-id", transformation="pack_cse",
            kernel_fusion=False, kernel_evidence={"required_kernel_roles": {"encode": ["_quantize_pack"]},
                                                "maximum_kernel_role_launches": {"encode": 1}})
        report = {"fusion_candidates": [], "pack_sharing_decisions": [candidate]}
        profile = self.profile(candidate, "_quantize_pack_groups", "_partial_min_max")
        for count, calls, expected in ((1, 1, "observed"), (2, 1, "not_observed"), (None, 1, "unknown"),
                                       (2, 2, "observed"), (4, 2, "not_observed")):
            with self.subTest(encode_launches=count, range_calls=calls):
                profile["compiler_regions"][candidate["profile_range"]]["range_calls_per_step"] = calls
                profile["compiler_regions"][candidate["profile_range"]]["kernel_launches_per_step"] = {
                    "_quantize_pack_groups": count, "_partial_min_max": 1}
                evidence = pack_sharing_evidence(report, profile)
                self.assertEqual(evidence["status"], expected)
                self.assertEqual(fusion_candidate_evidence(report, profile)["status"], "unfused")

    def test_archived_triton_driver_launch_retains_candidate_correlation(self):
        trace = self.archived_driver_trace()
        candidate = self.candidate(profile_range=trace["traceEvents"][0]["name"])
        profile = self.trace_profile(trace)
        evidence = self.qualify(candidate, profile)
        self.assertTrue(evidence["all_selected_observed"], evidence)
        region = profile["compiler_regions"][candidate["profile_range"]]
        self.assertEqual(region["kernel_launches_per_step"], {"_bf16_gemm": 1})
        self.assertEqual(region["correlation_ids"], [88])
        self.assertEqual(region["range_event_ids"], [519])

    def test_gpu_timestamp_overlap_cannot_replace_host_ownership_and_correlation(self):
        for change in ("wrong_thread", "wrong_process", "host_outside_range", "wrong_correlation", "missing_thread"):
            with self.subTest(change=change):
                trace = self.archived_driver_trace()
                launch = trace["traceEvents"][1]
                if change == "wrong_thread":
                    launch["tid"] += 1
                elif change == "wrong_process":
                    launch["pid"] += 1
                elif change == "host_outside_range":
                    launch["ts"] = -30
                elif change == "wrong_correlation":
                    launch["args"]["correlation"] += 1
                else:
                    del launch["tid"]
                candidate = self.candidate(profile_range=trace["traceEvents"][0]["name"])
                evidence = self.qualify(candidate, self.trace_profile(trace))
                self.assertEqual(evidence["status"], "unknown")
                self.assertEqual(evidence["candidates"][0]["reason"], "candidate_cuda_attribution_missing")

    def test_correlated_asynchronous_kernel_can_start_after_cpu_range_returns(self):
        trace = self.archived_driver_trace()
        trace["traceEvents"][2]["ts"] = 10000
        candidate = self.candidate(profile_range=trace["traceEvents"][0]["name"])
        self.assertTrue(self.qualify(candidate, self.trace_profile(trace))["all_selected_observed"])

    def test_trace_counts_deduplicate_driver_runtime_links_and_preserve_launch_limits(self):
        trace = self.archived_driver_trace()
        runtime = deepcopy(trace["traceEvents"][1])
        runtime.update(cat="cuda_runtime", name="cudaLaunchKernel")
        trace["traceEvents"].append(runtime)
        candidate = self.candidate(profile_range=trace["traceEvents"][0]["name"], kernel_evidence={
            "required_kernel_roles": {"gemm_activation": ["_bf16_gemm"]},
            "maximum_kernel_role_launches": {"gemm_activation": 1}})
        evidence = self.qualify(candidate, self.trace_profile(trace))
        self.assertTrue(evidence["all_selected_observed"], evidence)
        self.assertEqual(evidence["candidates"][0]["observed_cuda_launches"], 1)
        second_launch, second_kernel = deepcopy(trace["traceEvents"][1:3])
        second_launch["args"]["correlation"] = second_kernel["args"]["correlation"] = 89
        trace["traceEvents"].extend([second_launch, second_kernel])
        evidence = self.qualify(candidate, self.trace_profile(trace))
        self.assertEqual(evidence["status"], "not_observed")
        self.assertEqual(evidence["candidates"][0]["reason"], "kernel_role_launch_limit_exceeded")

    def test_trace_forbidden_kernel_inside_candidate_is_still_rejected(self):
        trace = self.archived_driver_trace()
        launch, kernel = deepcopy(trace["traceEvents"][1:3])
        launch["args"]["correlation"] = kernel["args"]["correlation"] = 89
        kernel["name"] = "standalone_gelu"
        trace["traceEvents"].extend([launch, kernel])
        candidate = self.candidate(profile_range=trace["traceEvents"][0]["name"])
        evidence = self.qualify(candidate, self.trace_profile(trace))
        self.assertEqual(evidence["status"], "not_observed")
        self.assertEqual(evidence["candidates"][0]["forbidden_kernel_symbols"], ["standalone_gelu"])

    def test_trace_reused_correlation_across_processes_is_ambiguous(self):
        trace = self.archived_driver_trace()
        other = deepcopy(trace["traceEvents"][1])
        other["pid"] += 1
        trace["traceEvents"].append(other)
        candidate = self.candidate(profile_range=trace["traceEvents"][0]["name"])
        self.assertEqual(self.qualify(candidate, self.trace_profile(trace))["status"], "unknown")

    def test_compressed_trace_and_multiple_profile_steps_keep_per_step_counts(self):
        import gzip
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        trace = self.archived_driver_trace()
        second = deepcopy(trace["traceEvents"])
        for event in second:
            event["ts"] += 10000
            if "correlation" in event["args"]:
                event["args"]["correlation"] += 1
        trace["traceEvents"].extend(second)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump(trace, stream)
            regions = compiler_launch_evidence([], 2, trace=path)
        region = regions[trace["traceEvents"][0]["name"]]
        self.assertEqual(region["range_calls_per_step"], 1)
        self.assertEqual(region["linked_cuda_launches_per_step"], 1)
        self.assertEqual(region["kernel_launches_per_step"], {"_bf16_gemm": 1})


if __name__ == "__main__":
    unittest.main()
