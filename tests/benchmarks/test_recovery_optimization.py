"""Behavioral contracts for recovery diagnostics and full attention boundaries."""

import unittest
from types import SimpleNamespace
from pathlib import Path
import os
import subprocess
import tempfile
import torch

from benchmarks.analysis.example_model import build_attention_block
from benchmarks.analysis.measurement import errors, classify_kernel
from benchmarks.analysis.reporting import summarize
from benchmarks.analysis.benchmark_vit_attention_stability import graph_launch_evidence, parse_args, resolve_run


class RecoveryOptimizationTests(unittest.TestCase):
    def test_attention_block_layouts_preserve_outputs_and_all_gradients(self):
        previous = None
        for layout in ('interleaved', 'bshd', 'bhsd'):
            torch.manual_seed(14)
            package = build_attention_block(torch.device('cpu'), batch_size=2, sequence_length=5,
                                            width=16, heads=2, source_layout=layout)
            x, = package['inputs']
            output = package['model'](x)
            output.square().sum().backward()
            values = (output, x.grad, *(p.grad for p in package['model'].parameters()))
            self.assertTrue(all(v is not None and torch.isfinite(v).all() for v in values))
            if previous is not None:
                for a, b in zip(values, previous):
                    torch.testing.assert_close(a, b)
            previous = tuple(v.detach().clone() for v in values)

    def test_direction_diagnostics_do_not_relax_acceptance(self):
        result = errors({'parameter_grad.x': torch.tensor([1., 1.])},
                        {'parameter_grad.x': torch.tensor([1., 0.])}, rtol=.01, atol=.01)
        value = result['tensors']['parameter_grad.x']
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(value['angle_degrees'], 45.)
        self.assertAlmostEqual(value['parallel_scale'], 1.)
        self.assertAlmostEqual(value['orthogonal_relative_l2'], 1.)
        zero = errors({'x': torch.zeros(2)}, {'x': torch.zeros(2)}, rtol=0, atol=0)['tensors']['x']
        self.assertIsNone(zero['angle_degrees'])

    def test_no_comparisons_does_not_claim_quality(self):
        report = summarize([], {})
        self.assertEqual(report['qualification']['status'], 'not_compared')
        self.assertEqual(report['qualification']['codec_oracle'], 'not_measured_by_this_matrix')

    def test_numerical_rejections_remain_visible_after_successful_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = []
            for case, value in [('torch_bf16', 1.), ('do_bf16', 2.)]:
                path = Path(folder) / (case + '.pt')
                torch.save({'output.0': torch.tensor([value])}, path)
                rows.append({'status': 'ok', 'initial_state_sha256': 'same', 'snapshot': str(path),
                    'task': {'variant_id': 'same', 'case': case, 'workload': {'name': 'tiny', 'op': 'linear'}},
                    'timing': {'wall_ms': {'median': 1.}}})
            result = summarize(rows, {'default': {'rtol': .01, 'atol': .01}})
        self.assertEqual(result['counts']['ok'], 2)
        self.assertEqual(result['qualification']['bf16_numerical_rejected'], 1)
        self.assertIsNone(result['comparisons'][0]['qualified_wall_speedup_vs_torch'])

    def test_kernel_categories_keep_decode_separate(self):
        self.assertEqual(classify_kernel('_dequantize_unpack_spatial'), 'decode_or_dequantize')
        self.assertEqual(classify_kernel('cudnn_depthwise'), 'convolution')
        self.assertEqual(classify_kernel('layer_norm_forward'), 'normalization')

    def test_graph_evidence_counts_actual_handles_and_restores_method(self):
        class Graph:
            def replay(self):
                return 'replayed'
        original = Graph.replay
        fake = SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph))
        a, b = Graph(), Graph()
        with graph_launch_evidence(fake) as snapshot:
            self.assertEqual(a.replay(), 'replayed')
            a.replay(); b.replay()
            self.assertEqual(snapshot()['total_replays'], 3)
            self.assertEqual(snapshot()['distinct_graph_handles'], 2)
        self.assertIs(Graph.replay, original)

    def test_real_training_policy_override_is_resolved_before_model_allocation(self):
        args = parse_args(['--output', '/tmp/deepopt-recovery-test', '--optimizer', 'test:Optimizer',
                           '--learning-rate', '.0001', '--convolution-policy', 'native_spatial',
                           '--model-name', 'efficientnet_b0'])
        _, config = resolve_run(args)
        self.assertEqual(config['runtime']['fp8']['convolution_policy'], 'native_spatial')
        self.assertEqual(config['training']['learning_rate'], .0001)

    def test_candidate_configs_have_bounded_declared_coverage(self):
        from analysis.benchmark_do_ops import parser, make_plan
        root = Path(__file__).resolve().parents[2] / 'benchmarks/analysis/configs'
        for name, sweep, count in [('hotspots', 'cartesian', 20), ('convolution_candidates', 'cartesian', 45),
                                    ('vit_packing_candidates', 'cartesian', 20),
                                    ('attention_block_candidates', 'one-at-a-time', 54), ('update_alignment', 'cartesian', 40)]:
            file = 'h100_optimization_hotspots.json' if name == 'hotspots' else 'h100_' + name + '.json'
            options = ['--config', str(root / file), '--sweep', sweep]
            if name == 'attention_block_candidates':
                options += ['--cases', 'torch_bf16', 'do_bf16', 'compression']
            self.assertEqual(len(make_plan(parser().parse_args(options))['tasks']), count)


class RecoveryLauncherTests(unittest.TestCase):
    def run_launcher(self, stage='all', fail='', extra_env=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / 'fake-python'
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
                                  'case "$*" in *"$FAIL_MATCH"*) if [ -n "$FAIL_MATCH" ]; then exit 1; fi;; esac\n')
            executable.chmod(0o700)
            log = root / 'calls.log'
            launcher = Path(__file__).resolve().parents[2] / 'benchmarks/analysis/run_optimization_recovery.sh'
            result = subprocess.run(['bash', str(launcher), str(root / 'results'), '--resume'],
                env={**os.environ, 'BENCHMARK_PYTHON': str(executable), 'CALL_LOG': str(log),
                     'FAIL_MATCH': fail, 'RECOVERY_STAGE': stage, **(extra_env or {})}, capture_output=True, text=True)
            calls = log.read_text().splitlines() if log.exists() else []
            return result, calls

    def test_preflight_failure_prevents_expensive_jobs(self):
        result, calls = self.run_launcher(fail='benchmark_attention_preflight.py')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), 1)

    def test_independent_stages_continue_and_resume_is_forwarded(self):
        result, calls = self.run_launcher(fail='h100_convolution_candidates.json')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all('--resume' in c for c in calls[1:]))
        self.assertNotIn('--resume', calls[0])

    def test_stage_filter_and_invalid_stage(self):
        result, calls = self.run_launcher(stage='vit')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn('h100_vit_packing_candidates.json', calls[1])
        result, calls = self.run_launcher(stage='unknown')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])

    def test_training_stage_uses_declared_models_static_batches_and_actual_optimizer(self):
        result, calls = self.run_launcher(stage='training', extra_env={
            'RECOVERY_TRAIN_CONFIG': 'actual.toml', 'RECOVERY_TRAIN_OPTIMIZER': 'test:Optimizer',
            'RECOVERY_TRAIN_INITIALIZER': 'test:initialize'})
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all('--drop-last' in c and '--optimizer test:Optimizer' in c and
                            '--initializer test:initialize' in c and '--resume' in c for c in calls[1:]))
        self.assertIn('--model-name resnet18 --model-library torchvision', calls[2])
        self.assertIn('--model-name efficientnet_b0 --model-library timm', calls[3])
