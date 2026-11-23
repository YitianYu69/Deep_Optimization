"""Real VAE pack-sharing controls preserve matching-recipe training semantics."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from deepopt import autocast
from deepopt.compiler import compilation_report
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model
from examples.vae.mbconv_vae import MBConvVAE_V2
from examples.vae.train_vae import vae_loss
from benchmarks.analysis.measurement import SavedStorageLifetimes


class VAEFusionPackSharingTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(6102)

    def test_shared_convolution_packs_preserve_full_outputs_gradients_and_updates(self):
        recipe = dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)
        model = MBConvVAE_V2(base_channels=4, max_channels=16, blocks_per_stage=1,
                            latent_channels=2).to(self.device)
        model = prepare_fp8_model(model, False, activation_compression=recipe, config={})[0]
        source = torch.randn(1, 3, 16, 16, device=self.device)
        controls, storage, decisions = [], [], []
        for share in (None, False, True):
            torch.compiler.reset()
            current = deepcopy(model)
            x = source.clone().requires_grad_()
            initial = {name: value.detach().clone() for name, value in current.named_parameters()}
            if share is not None:
                compile_model(current, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    options={"require_full_aot": True, "use_native_autodiff": True,
                             "share_compatible_packs": share}), config={})
            torch.manual_seed(7103)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(7103)
            tracker = SavedStorageLifetimes((*current.parameters(), *current.buffers()))
            with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
                with autocast(self.device, dtype=torch.bfloat16, fp8=False,
                              activation_compression=recipe, config={}):
                    outputs = current(x)
                loss, _ = vae_loss(outputs[0], x, outputs[1], outputs[2], beta=1e-6)
            storage.append(tracker.snapshot()["live_nonpersistent_bytes"])
            # Native derivatives and shared source ownership survive scope exit.
            loss.backward()
            self.assertEqual(tracker.snapshot()["live_unique_bytes"], 0)
            optimizer = torch.optim.SGD(current.parameters(), lr=0.001)
            gradients = {name: value.grad.detach().clone() if value.grad is not None else None
                         for name, value in current.named_parameters()}
            optimizer.step()
            controls.append({"outputs": tuple(value.detach().clone() for value in outputs),
                "input_grad": x.grad.detach().clone(), "gradients": gradients,
                "updates": {name: value.detach() - initial[name] for name, value in current.named_parameters()}})
            if share is not None:
                report = compilation_report(current).as_dict()
                self.assertEqual(report["runtime_fallbacks"], 0)
                self.assertFalse(report["legacy_backward_operations"])
                decisions.append([item for graph in report["graphs"] for item in graph["pack_sharing_decisions"]
                                  if item["selected"]])
                if share and self.device.startswith("cuda"):
                    from benchmarks.analysis.measurement import compiler_launch_evidence, fusion_candidate_evidence

                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
                        with autocast(self.device, dtype=torch.bfloat16, fp8=False,
                                      activation_compression=recipe, config={}):
                            profiled = current(x)
                        sum(value.float().square().mean() for value in profiled).backward()
                        torch.cuda.synchronize()
                    with TemporaryDirectory(prefix="vae-pack-sharing-trace-") as directory:
                        trace = Path(directory) / "trace.json"
                        profile.export_chrome_trace(str(trace))
                        evidence = fusion_candidate_evidence({"fusion_candidates": decisions[-1]}, {
                            "compiler_regions": compiler_launch_evidence(profile.events(), 1, trace=trace)})
                    self.assertTrue(evidence["all_selected_observed"], evidence)
        # This is single-update correctness and physical saved-storage coverage,
        # not a CUDA allocator-peak or custom-optimizer qualification claim.
        tolerance = dict(rtol=0, atol=0) if self.device == "cpu" else dict(rtol=0.02, atol=0.002)
        for result in controls[1:]:
            for actual, expected in zip(result["outputs"], controls[0]["outputs"]):
                torch.testing.assert_close(actual, expected, **tolerance)
            torch.testing.assert_close(result["input_grad"], controls[0]["input_grad"], **tolerance)
            for category in ("gradients", "updates"):
                for name, expected in controls[0][category].items():
                    actual = result[category][name]
                    if expected is None:
                        self.assertIsNone(actual, name)
                    else:
                        torch.testing.assert_close(actual, expected, **tolerance, msg=name)
        self.assertFalse(decisions[0])
        self.assertTrue(decisions[1])
        self.assertTrue(all(row["compatibility"]["group_size"] == 1024 for row in decisions[1]))
        self.assertGreater(sum(row["estimated_removed_component_bytes"] for row in decisions[1]), 0)
        self.assertLess(storage[2], storage[1])


@unittest.skipUnless(torch.cuda.is_available(), "VAE generated pack sharing requires CUDA")
class VAEFusionPackSharingCudaTests(VAEFusionPackSharingTests):
    device = "cuda:0"
