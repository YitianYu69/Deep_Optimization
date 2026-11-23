"""Performance-only results retain fair comparisons and valid resume identity."""

from copy import deepcopy
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.analysis.artifacts import snapshot_valid
from benchmarks.analysis.benchmark_do_ops import parser, make_plan
from benchmarks.analysis.measurement import classify_kernel
from benchmarks.analysis.reporting import summarize, write_reports
from benchmarks.analysis.scheduler import effective_task


class RuntimeSpeedBenchmarkTests(unittest.TestCase):
    def test_interactions_inherit_variability_drift_and_rng_qualification(self):
        task = make_plan(parser().parse_args(['--ops','linear','--small','--device','cpu']))['tasks'][0]
        task['workload']['calls_per_step'] = 1
        rows = []
        for case, ms in (('do_bf16', 2.), ('compression', 2.5), ('fp8', 1.5), ('fp8_compression', 2.2)):
            t = deepcopy(task); t['case'] = case
            rows.append({'task': t, 'status': 'ok', 'initial_state_sha256': 'same',
                'rng_baseline_sha256': 'rng', 'measurement_policy': {'numerical_comparison': False},
                'finite_after_timing': True, 'timing': {'wall_ms': {'median': ms, 'stdev': 0.,
                    'samples': [ms] * 3, 'relative_stdev': 0.}, 'cuda_stream_ms': None}})
        report = summarize(rows, {})
        self.assertEqual(report['interactions'][0]['status'], 'ok')
        self.assertAlmostEqual(report['interactions'][0]['performance_interaction_wall_ms'], .2)
        for field, value, flag in (("timing", None, 'repeat_variability_passed'),
                                   ('control_drift', {'passed': False}, 'control_drift_passed'),
                                   ('rng_baseline_sha256', 'changed', 'same_rng_baseline')):
            changed = deepcopy(rows)
            if field == 'timing':
                changed[-1]['timing']['wall_ms']['relative_stdev'] = .04
            else:
                changed[-1][field] = value
            result = summarize(changed, {})
            interaction = result['interactions'][0]
            self.assertEqual(interaction['status'], 'unqualified')
            self.assertFalse(interaction[flag])
            self.assertIsNone(interaction['performance_interaction_wall_ms'])
            self.assertTrue(interaction['noise_warning'])
            self.assertEqual(result['ranked_by_model_contribution'], [])

    def test_graph_comparisons_match_settings_and_exclude_changed_or_noisy_controls(self):
        task = make_plan(parser().parse_args(['--ops','linear','--small','--device','cpu']))['tasks'][0]
        rows = []
        for mode, ms in (('eager',2.), ('graph',1.5)):
            t=deepcopy(task);t['settings']['runtime']['execution']=mode;t['variant_id']=mode
            rows.append({'task':t,'status':'ok','initial_state_sha256':'same','rng_baseline_sha256':'rng',
                         'measurement_policy':{'numerical_comparison':False},'finite_after_timing':True,
                         'timing':{'wall_ms':{'median':ms,'relative_stdev':.01}},
                         'execution':{'convolution_launches':[]}})
        report=summarize(rows,{})
        pair=report['eager_vs_graph'][0]
        self.assertEqual(pair['status'],'ok')
        self.assertAlmostEqual(pair['performance_graph_speedup'],4/3)
        self.assertAlmostEqual(pair['latency_reduction_percent'],25)
        with tempfile.TemporaryDirectory() as folder:
            write_reports(folder,{**report,'results':rows})
            with (Path(folder)/'eager-vs-graph.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),1)
        rows[1]['timing']['wall_ms']['relative_stdev']=.04
        self.assertIsNone(summarize(rows,{})['eager_vs_graph'][0]['performance_graph_speedup'])
        rows[1]['timing']['wall_ms']['relative_stdev']=.01
        rows[1]['initial_state_sha256']='changed'
        self.assertFalse(summarize(rows,{})['eager_vs_graph'][0]['same_initial_state'])
        rows[1]['task']['settings']['compression']['bits']=8
        self.assertFalse(summarize(rows,{})['eager_vs_graph'])

    def test_full_model_configs_preserve_all_models_feature_modes_and_execution(self):
        root=Path(__file__).resolve().parents[2]/'benchmarks/analysis/configs'
        for name in ('comparison','native'):
            plan=make_plan(parser().parse_args(['--config',str(root/f'h100_runtime_speed_{name}.json'),'--sweep','cartesian']))
            self.assertEqual(len(plan['tasks']),40)
            self.assertEqual({t['case'] for t in plan['tasks']},{'torch_bf16','do_bf16','compression','fp8','fp8_compression'})
            self.assertEqual({t['settings']['runtime']['execution'] for t in plan['tasks']},{'eager','graph'})
            self.assertEqual(len({t['workload']['name'] for t in plan['tasks']}),4)
            self.assertTrue(all(t['workload']['shape']==[512,3,224,224] for t in plan['tasks']))
            self.assertTrue(all(not t['workload']['factory_kwargs'].get('pretrained',False) for t in plan['tasks']))

    def test_native_template_accessors_are_not_packing(self):
        for operation, expected in (("batch_norm_backward_kernel", "normalization"),
                                    ("conv_depthwise2d_grad_weight_kernel", "convolution"),
                                    ("max_pool2d_kernel", "pooling")):
            self.assertEqual(classify_kernel(f"void at::native::{operation}<GenericPackedTensorAccessor<float>>"), expected)
        self.assertEqual(classify_kernel("_quantize_pack_groups"), "pack_or_fused_cast_pack")
        self.assertEqual(classify_kernel("_activation_forward"), "activation")
        self.assertEqual(classify_kernel("_channels_last_apply"), "normalization")
        self.assertEqual(classify_kernel("_channels_last_activation_dx"), "activation")

    def test_performance_options_are_in_resume_identity(self):
        args = ['--ops', 'linear', '--small', '--device', 'cpu', '--cases', 'do_bf16']
        old = make_plan(parser().parse_args(args))
        fast = make_plan(parser().parse_args(args + ['--no-numerical-comparison']))
        self.assertNotEqual(old['tasks'][0]['id'], fast['tasks'][0]['id'])
        self.assertFalse(fast['options']['numerical_comparison'])
        with self.assertRaisesRegex(ValueError, 'requires'):
            make_plan(parser().parse_args(args + ['--no-numerical-comparison', '--strict-accuracy']))

    def test_inactive_producer_settings_deduplicate(self):
        task = make_plan(parser().parse_args(['--ops','linear','--small','--device','cpu']))['tasks'][0]
        for case in ('torch_bf16', 'do_bf16', 'compression', 'fp8', 'fp8_compression'):
            first, second = deepcopy(task), deepcopy(task)
            first['case'] = second['case'] = case
            second['settings']['runtime']['producer_pack'] = 'all'
            if case == 'fp8_compression':
                self.assertNotEqual(effective_task(first), effective_task(second))
            else:
                self.assertEqual(effective_task(first), effective_task(second))

    def test_performance_only_reporting_never_loads_or_claims_numerical_results(self):
        task = make_plan(parser().parse_args(['--ops','linear','--small','--device','cpu']))['tasks'][0]
        rows = []
        for case, ms in (('torch_bf16',2.), ('do_bf16',2.), ('compression',1.8), ('fp8',1.5), ('fp8_compression',1.4)):
            t = deepcopy(task); t['case'] = case
            rows.append({'task': t, 'status': 'ok', 'initial_state_sha256': 'same',
                'snapshot': None, 'snapshot_file': None, 'numerical_snapshot': {'status':'not_requested'},
                'measurement_policy': {'numerical_comparison':False}, 'finite_after_timing': True,
                'timing': {'wall_ms': {'median':ms, 'stdev':0.,'samples':[ms]*3,'relative_stdev':0.}, 'cuda_stream_ms':None}})
        with patch('torch.load', side_effect=AssertionError('no numerical snapshots')):
            report = summarize(rows, {})
        self.assertEqual(report['qualification']['status'], 'performance_only')
        self.assertEqual(report['qualification']['comparisons_checked'], 0)
        self.assertEqual(report['qualification']['performance_timing_accepted'], 4)
        self.assertTrue(all(c['accuracy_passed'] is None for c in report['comparisons']))
        self.assertTrue(all(c['qualified_wall_speedup_vs_torch'] is None for c in report['comparisons']))
        self.assertTrue(snapshot_valid(rows[0]))
        rows[0]['measurement_policy']['numerical_comparison'] = True
        self.assertFalse(snapshot_valid(rows[0]))

    def test_cpu_performance_only_execution_creates_no_tensor_artifact(self):
        from benchmarks.analysis.engine import run_task
        with tempfile.TemporaryDirectory() as folder:
            plan = make_plan(parser().parse_args(['--ops','linear','--small','--device','cpu',
                '--cases','do_bf16','--no-numerical-comparison','--warmup','0','--iterations','1',
                '--repeats','1','--phase-steps','0','--no-memory']))
            row = run_task(plan['tasks'][0], plan['options'], Path(folder))
            self.assertEqual(row['status'], 'ok')
            self.assertIsNone(row['snapshot'])
            self.assertEqual(row['numerical_snapshot']['status'], 'not_requested')
            self.assertTrue(row['finite_after_timing'])
            self.assertFalse(list(Path(folder).glob('*.pt')))


class RuntimeSpeedLauncherTests(unittest.TestCase):
    def run_launcher(self,stage='compare',fail='',final=False,extra_env=None):
        with tempfile.TemporaryDirectory(prefix='runtime speed ') as folder:
            root=Path(folder);executable=root/'fake-python';log=root/'calls.jsonl'
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
                                  'case "$*" in *"$FAIL_MATCH"*) if [ -n "$FAIL_MATCH" ]; then exit 1; fi;; esac\n')
            executable.chmod(0o700)
            script=Path(__file__).resolve().parents[2]/'benchmarks/analysis/run_runtime_speed.sh'
            command=['bash',str(script),str(root/'results'),'--resume']+(['--final'] if final else [])
            result=subprocess.run(command,env={**os.environ,'BENCHMARK_PYTHON':str(executable),
                'CALL_LOG':str(log),'FAIL_MATCH':fail,'SPEED_STAGE':stage,**(extra_env or {})},capture_output=True,text=True)
            return result,log.read_text().splitlines() if log.exists() else []

    def test_preflight_blocks_expensive_work_and_final_defaults_are_forwarded(self):
        result,calls=self.run_launcher(fail='benchmark_attention_preflight.py')
        self.assertEqual(result.returncode,1);self.assertEqual(len(calls),1)
        self.assertIn('--runtime-speed',calls[0])
        result,calls=self.run_launcher(final=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(len(calls),2)
        self.assertIn('--no-numerical-comparison',calls[1]);self.assertIn('--resume',calls[1])
        self.assertIn('--warmup 10 --iterations 30 --repeats 5',calls[1])
        self.assertIn('--timing-isolation exclusive',calls[1])
        self.assertIn('--cpu-workers 1', calls[1])

    def test_references_are_explicit_and_preserve_torchao_eager_label(self):
        result, calls = self.run_launcher('references')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        self.assertIn('h100_runtime_speed_compiled.json', calls[1])
        self.assertIn('--cases torch_bf16', calls[1])
        self.assertIn('--implementations bf16 deepopt-current torchao', calls[2])
        source = Path(__file__).resolve().parents[2]/'benchmarks/fp8/benchmark_fp8_linear_compare.py'
        self.assertIn(str(source.parent.parent/'analysis/../fp8/benchmark_fp8_linear_compare.py'), calls[2])
        self.assertTrue(source.is_file())
        result, calls = self.run_launcher('references', extra_env={'BENCHMARK_REFERENCE_DEVICE': 'cuda:2'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--device cuda:2', calls[2])

    def test_selected_profiles_use_only_current_list_and_the_winning_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'selected-configs.txt').write_text('selected-winner.json\n')
            (root/'selected-winner.json').write_text('{}')
            (root/'selected-stale.json').write_text('{}')
            result, calls = self.run_launcher('profiles', extra_env={'SPEED_SELECTED_CONFIGS': str(root)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(calls), 2)
            self.assertIn(str(root/'selected-winner.json'), calls[1])
            self.assertNotIn('selected-stale', calls[1])
            self.assertIn('profiles-selected-winner', calls[1])

    def test_all_stages_continue_after_candidate_failure_and_reject_unknown_stage(self):
        result,calls=self.run_launcher('all',fail='h100_runtime_speed_convolution.json')
        self.assertEqual(result.returncode,1);self.assertEqual(len(calls),8)
        self.assertTrue(all('--resume' in call for call in calls[1:]))
        result,calls=self.run_launcher('invalid')
        self.assertEqual(result.returncode,2);self.assertFalse(calls)

    def test_training_runs_both_execution_modes_with_existing_recipe(self):
        result,calls=self.run_launcher('training',extra_env={'RECOVERY_TRAIN_CONFIG':'actual.toml',
            'RECOVERY_TRAIN_OPTIMIZER':'test:Optimizer','RECOVERY_TRAIN_INITIALIZER':'test:initialize'})
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(len(calls),7)
        self.assertEqual(sum('--no-graph' in call for call in calls),3)
        self.assertEqual(sum('--graph ' in call for call in calls),3)
        self.assertTrue(all('--batch-size 512 --drop-last' in call and '--initializer test:initialize' in call
                            and '--optimizer test:Optimizer' in call for call in calls[1:]))


class RuntimeSpeedOracleTests(unittest.TestCase):
    def test_cpu_oracle_keeps_exact_codes_blocks_and_current_scales(self):
        import torch
        from deepopt.opt.backends.reference import convolution as reference
        from deepopt.opt.nn.convolution import ConvGeometry,source_matrix
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from tests.cuda.test_runtime_speed import convolution_cpu_reference
        torch.manual_seed(14)
        x=torch.randn(2,4,5,7);w=torch.randn(8,4,3,3)
        geometry=ConvGeometry.from_tensors(x,w,padding=1)
        for block in (None,(4,2)):
            qx,qw,qg=[quantize_fp8(v,block_shape=block) for v in
                (source_matrix(x),w.reshape(8,-1),source_matrix(torch.randn(geometry.output_shape)))]
            expected=(reference.forward(qx,qw,geometry),reference.input_gradient(qg,qw,geometry),
                      reference.weight_gradient(qx,qg,geometry))
            with patch.object(reference,'forward',wraps=reference.forward) as forward,\
                 patch.object(reference,'input_gradient',wraps=reference.input_gradient) as dx,\
                 patch('deepopt.opt.quantization.quantizer.quantize_fp8',side_effect=AssertionError('Do not requantize the oracle')):
                actual=convolution_cpu_reference(qx,qw,qg,geometry)
            copies=(forward.call_args.args[0],forward.call_args.args[1],dx.call_args.args[0])
            for copied,source in zip(copies,(qx,qw,qg)):
                self.assertEqual(copied.data.device.type,'cpu')
                self.assertEqual(copied.block_shape,source.block_shape)
                self.assertNotEqual(copied.data.data_ptr(),source.data.data_ptr())
                self.assertNotEqual(copied.scale_inv.data_ptr(),source.scale_inv.data_ptr())
                torch.testing.assert_close(copied.data.view(torch.uint8),source.data.view(torch.uint8),rtol=0,atol=0)
                torch.testing.assert_close(copied.scale_inv,source.scale_inv,rtol=0,atol=0)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            before=copies[0].scale_inv.clone()
            qx.scale_inv.mul_(.5)
            updated=convolution_cpu_reference(qx,qw,qg,geometry)
            torch.testing.assert_close(copies[0].scale_inv,before,rtol=0,atol=0)
            torch.testing.assert_close(updated[0],actual[0]*.5)
            torch.testing.assert_close(updated[1],actual[1])
            torch.testing.assert_close(updated[2],actual[2]*.5)

    def test_tf32_rounding_can_fail_the_unmodified_strict_tolerance(self):
        import torch
        from torch.nn import functional as F
        from deepopt.opt.nn.convolution import ConvGeometry,source_matrix,restore_matrix
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from tests.cuda.test_runtime_speed import convolution_cpu_reference
        torch.manual_seed(14)
        x=torch.randn(2,32,9,11);w=torch.randn(64,32,3,3)
        geometry=ConvGeometry.from_tensors(x,w,padding=1)
        qx,qw,qg=[quantize_fp8(v) for v in (source_matrix(x),w.reshape(64,-1),
                                          source_matrix(torch.randn(geometry.output_shape)))]
        oracle=convolution_cpu_reference(qx,qw,qg,geometry)[0]
        def tf32(value):
            mantissa,exponent=torch.frexp(value)
            return torch.ldexp(torch.round(mantissa*2048)/2048,exponent)
        a=restore_matrix(qx.dequantize(),tuple(x.shape));b=qw.dequantize().reshape(w.shape)
        lowered=source_matrix(F.conv2d(tf32(a),tf32(b),padding=1))
        self.assertTrue(((lowered-oracle).abs()>(.003+.003*oracle.abs())).any())
        # The ideal raw-code product isolates scaling. It does not emulate
        # Hopper's reduced-precision native FP8 accumulator.
        raw=F.conv2d(restore_matrix(qx.data.float(),tuple(x.shape)),qw.data.float().reshape(w.shape),padding=1)
        strict=source_matrix(raw)*(qx.scale_inv*qw.scale_inv)
        torch.testing.assert_close(strict,oracle,rtol=.003,atol=.003)

    def test_failed_convolution_reports_the_direction_and_launch(self):
        import torch
        from tests.cuda.test_runtime_speed import assert_convolution_close
        zero=torch.zeros(1);actual=(zero,zero,zero+.004)
        with self.assertRaisesRegex(AssertionError,'strict/tuned/dw.*reference=CPU FP32'):
            assert_convolution_close(actual,(zero,zero,zero),variant='strict',policy='tuned',launches=[{'BLOCK_K':128}])


if __name__ == '__main__':
    unittest.main()
