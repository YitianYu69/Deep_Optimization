"""Candidate exports preserve compute policies and reject incomparable timings."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from analysis.benchmark_do_ops import make_plan, parser
from analysis.benchmark_select_candidates import main, select_candidates


class CandidateSelectionTests(unittest.TestCase):
    def row(self, identifier, ms, *, policy="strict", variability=0.):
        task = make_plan(parser().parse_args(['--ops', 'linear', '--small', '--device', 'cpu']))['tasks'][0]
        task['id'] = identifier
        task['case'] = 'fp8'
        task['settings']['fp8']['convolution_policy'] = policy
        return {'task': task, 'status': 'ok', 'initial_state_sha256': 'model', 'rng_baseline_sha256': 'rng',
                'shared_optimizer_baseline_sha256': 'optimizer', 'environment': {'deepopt_source_sha256': 'source'},
                'measurement_policy': {'gpu_uuid': 'gpu'}, 'finite_after_timing': True,
                'timing': {'wall_ms': {'median': ms, 'samples': [ms] * 3, 'relative_stdev': variability}}}

    def test_policies_are_separate_and_noisy_fastest_sample_is_rejected(self):
        rows = [self.row('baseline', 5), self.row('strict-winner', 4),
                self.row('noisy', 1, variability=.1), self.row('native', 2, policy='native_spatial')]
        selected, excluded = select_candidates({'results': rows})
        self.assertEqual({r['task']['id'] for r in selected}, {'strict-winner', 'native'})
        self.assertEqual(excluded, [{'task_id': 'noisy', 'reason': 'repeat_variability'}])

    def test_identity_mismatch_cannot_choose_a_winner(self):
        for field in ('initial_state_sha256', 'rng_baseline_sha256', 'shared_optimizer_baseline_sha256'):
            rows = [self.row('a', 5), self.row('b', 4)]
            rows[1][field] = 'different'
            selected, excluded = select_candidates({'results': rows})
            self.assertEqual(selected, [])
            self.assertTrue(all(r['reason'] == 'incomparable_baselines' for r in excluded))

    def test_quantization_and_accumulation_policies_are_never_ranked_together(self):
        rows = [self.row('base', 5)]
        for section, key, value in (('compression', 'bits', 4), ('compression', 'group_size', 512),
                                    ('fp8', 'scaling_recipe', 'delayed'),
                                    ('runtime', 'linear_forward_backend', 'native_precise')):
            row = self.row(section + key, 1)
            row['task']['settings'][section][key] = value
            rows.append(row)
        selected, excluded = select_candidates({'results': rows})
        self.assertEqual(len(selected), 5)
        self.assertEqual(excluded, [])

    def test_fresh_empty_optimizer_baseline_is_valid_but_missing_warmed_state_is_not(self):
        row = self.row('fresh', 1)
        row['shared_optimizer_baseline_sha256'] = None
        selected, excluded = select_candidates({'results': [row]})
        self.assertEqual(len(selected), 1)
        self.assertEqual(excluded, [])
        row['task']['settings']['runtime']['optimizer_warmup_steps'] = 1
        selected, excluded = select_candidates({'results': [row]})
        self.assertEqual(selected, [])
        self.assertEqual(excluded[0]['reason'], 'incomparable_baselines')

    def test_export_contains_exact_settings_provenance_and_current_file_list(self):
        row = self.row('winner', 4)
        row['task']['settings']['runtime']['producer_pack'] = 'gelu'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root/'summary.json'
            summary.write_text(json.dumps({'results': [row]}))
            output = root/'selected'
            self.assertEqual(main([str(summary), '--output', str(output)]), 0)
            names = (output/'selected-configs.txt').read_text().splitlines()
            self.assertEqual(len(names), 1)
            config = json.loads((output/names[0]).read_text())
            self.assertEqual(config['settings'], row['task']['settings'])
            self.assertEqual(config['workloads'], [row['task']['workload']])
            self.assertNotIn('cases', config)  # All controls remain available during profiling.
            manifest = json.loads((output/'selection.json').read_text())
            self.assertEqual(manifest['selections'][0]['source_sha256'], 'source')
            stale = output/'selected-stale.json'
            stale.write_text('{}')
            self.assertEqual(main([str(summary), '--output', str(output)]), 0)
            self.assertNotIn(stale.name, (output/'selected-configs.txt').read_text())
            self.assertEqual(config, json.loads((output/names[0]).read_text()))

    def test_shared_and_unverified_or_nonfinite_rows_are_excluded(self):
        baseline = self.row('a', 1)
        for update, reason in (({'scheduling': {'isolation': 'shared'}}, 'shared_timing'),
                               ({'finite_after_timing': False}, 'nonfinite_or_unchecked'),
                               ({'compression_coverage': {'status': 'unverified'}}, 'compression_unverified')):
            row = deepcopy(baseline); row.update(update)
            selected, excluded = select_candidates({'results': [row]})
            self.assertEqual(selected, [])
            self.assertEqual(excluded[0]['reason'], reason)


if __name__ == '__main__':
    unittest.main()
