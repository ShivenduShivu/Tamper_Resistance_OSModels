import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'capability'))
import phaseA_export as px

HUMANEVAL_IDS = ['HumanEval/%d' % i for i in range(164)]


def panel_source():
    rows = {('mmlu', 'm%d' % i): {'task': 'mmlu', 'id': 'm%d' % i} for i in range(4)}
    rows.update({('humaneval', pid): {'task': 'humaneval', 'id': pid} for pid in HUMANEVAL_IDS})
    rows[('xstest', 'x1')] = {'task': 'xstest', 'id': 'x1', 'prompt_class': 'safe', 'prompt': 'p1'}
    rows[('xstest', 'x2')] = {'task': 'xstest', 'id': 'x2', 'prompt_class': 'harmful', 'prompt': 'p2'}
    return rows


def write_run(root, model, panel_hash, panel_keys, kind='exploratory_clean_model_results',
              drop_generation=False, scored_mismatch=False, gate_state='pass'):
    root = Path(root)
    (root / 'scoring-v1').mkdir(parents=True)
    (root / 'coding-adapted-v2').mkdir()
    gen_rows = []
    for key in sorted(panel_keys):
        if drop_generation and key == sorted(panel_keys)[0]:
            continue
        gen_rows.append({'task': key[0], 'id': key[1], 'response': 'r', 'output_tokens': 3,
                         'eos_reached': True, 'skipped': False, 'truncated': False,
                         'classifier_refusal': False, 'classifier_confidence': 0.1,
                         'classifier_input_truncated': False})
    items = '\n'.join(json.dumps(r) for r in gen_rows) + '\n'
    (root / 'items.jsonl').write_text(items)
    scored_rows = []
    for r in gen_rows:
        row = dict(r)
        row['correct'] = 1
        row['forced_choice_correct'] = 1
        row['published_parser_correct'] = 1
        if scored_mismatch:
            row['response'] = 'different'
        scored_rows.append(row)
    (root / 'scoring-v1' / 'scored_items.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in scored_rows) + '\n')
    (root / 'scoring-v1' / 'summary.json').write_text(json.dumps({'scoring_provenance': {
        'classifier_repo': 'c', 'classifier_revision': 'cr', 'score_source_sha256': 'ss'}}))
    code_rows = [{'id': pid, 'correct': 1, 'status': 'pass'} for pid in HUMANEVAL_IDS]
    (root / 'coding-adapted-v2' / 'items.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in code_rows) + '\n')
    (root / 'coding-adapted-v2' / 'summary.json').write_text(json.dumps({
        'state': 'complete_valid', 'infra_errors': 0, 'model': model, 'protocol': 'coding-v2',
        'generation_items_sha256': hashlib.sha256(items.encode()).hexdigest(),
        'backend_gate': {'state': gate_state, 'canonical_passed': 164, 'canonical_n': 164,
                         'authored_fixture_failures': 0}}))
    (root / 'manifest.json').write_text(json.dumps({
        'state': 'complete', 'kind': kind, 'model': model, 'panel_items_sha256': panel_hash,
        'panel_id': 'panel', 'protocol': 'p', 'tokenizer': 't', 'effective_eos_ids': [1],
        'repo': 'repo', 'revision': 'rev', 'source_sha256': 'src', 'tasks': {}}))
    (root / 'COMPLETE').write_text('')
    for sub in (root, root / 'scoring-v1', root / 'coding-adapted-v2'):
        index = {}
        for f in sub.iterdir():
            if f.is_file() and f.name != 'artifact_index.json':
                data = f.read_bytes()
                index[f.name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        (sub / 'artifact_index.json').write_text(json.dumps(index))


class PhaseAExportTests(unittest.TestCase):
    def setUp(self):
        self.panel = panel_source()
        self.panel_hash = hashlib.sha256(b'panel').hexdigest()
        self.keys = set(self.panel)

    def test_verify_index_detects_tampering_and_uncovered_files(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / 'f').write_bytes(b'original')
            (p / 'artifact_index.json').write_text(json.dumps(
                {'f': {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}}))
            px.verify_index(p)
            (p / 'f').write_bytes(b'modified')
            with self.assertRaises(ValueError):
                px.verify_index(p)
            (p / 'f').write_bytes(b'original')
            (p / 'extra').write_bytes(b'unindexed')
            with self.assertRaises(ValueError):
                px.verify_index(p)

    def test_success_getter_maps_secondary_endpoints(self):
        row = {'correct': 1, 'forced_choice_correct': 0, 'published_parser_correct': 0}
        self.assertEqual(px.success_getter('mmlu')(row), 1)
        self.assertEqual(px.success_getter('mmlu_forced')(row), 0)
        self.assertEqual(px.success_getter('gsm8k_published')(row), 0)
        self.assertEqual(px.success_getter('humaneval')(row), 1)
        with self.assertRaises(ValueError):
            px.success_getter('xstest_safe')

    def test_endpoint_ids_and_safe_ids(self):
        for ep in ('mmlu', 'mmlu_forced'):
            self.assertEqual(px.endpoint_ids(self.panel, ep), ['m0', 'm1', 'm2', 'm3'])
        self.assertEqual(px.endpoint_ids(self.panel, 'humaneval'), sorted(HUMANEVAL_IDS))
        self.assertEqual(px.xstest_safe_ids(self.panel), ['x1'])
        with self.assertRaises(ValueError):
            px.endpoint_ids(self.panel, 'xstest')

    def test_load_model_binds_scored_rows_to_generation(self):
        with tempfile.TemporaryDirectory() as td:
            write_run(Path(td) / 'run', 'a1', self.panel_hash, self.keys)
            loaded = px.load_model('a1', Path(td) / 'run', self.panel, self.panel_hash)
            self.assertEqual(loaded['manifest']['model'], 'a1')
            self.assertEqual(loaded['items'][('humaneval', 'HumanEval/0')]['code_status'], 'pass')
            self.assertEqual(loaded['coding_protocol'], 'coding-v2')

    def test_load_model_rejects_pilot_wrong_model_and_wrong_panel(self):
        with tempfile.TemporaryDirectory() as td:
            write_run(Path(td) / 'pilot', 'a1', self.panel_hash, self.keys, kind='pilot_partial')
            with self.assertRaises(ValueError):
                px.load_model('a1', Path(td) / 'pilot', self.panel, self.panel_hash)
            write_run(Path(td) / 'wrong', 'a2', self.panel_hash, self.keys)
            with self.assertRaises(ValueError):
                px.load_model('a1', Path(td) / 'wrong', self.panel, self.panel_hash)
            write_run(Path(td) / 'panel', 'a1', 'otherhash', self.keys)
            with self.assertRaises(ValueError):
                px.load_model('a1', Path(td) / 'panel', self.panel, self.panel_hash)

    def test_load_model_rejects_incomplete_or_unbound_runs(self):
        with tempfile.TemporaryDirectory() as td:
            write_run(Path(td) / 'short', 'a1', self.panel_hash, self.keys, drop_generation=True)
            with self.assertRaises(ValueError):
                px.load_model('a1', Path(td) / 'short', self.panel, self.panel_hash)
            write_run(Path(td) / 'unbound', 'a1', self.panel_hash, self.keys, scored_mismatch=True)
            with self.assertRaises(ValueError):
                px.load_model('a1', Path(td) / 'unbound', self.panel, self.panel_hash)
            write_run(Path(td) / 'gate', 'a1', self.panel_hash, self.keys, gate_state='fail')
            with self.assertRaises(ValueError):
                px.load_model('a1', Path(td) / 'gate', self.panel, self.panel_hash)

    def test_render_readme_keeps_exploratory_caveats(self):
        entry = {'base': {'n': 2, 'success_n': 1}, 'v2': {'n': 2, 'success_n': 2},
                 'a1': {'n': 2, 'success_n': 1},
                 'a1_vs_v2': {'delta': -0.5, 'delta_ci': [-1.0, 0.0]}}
        scores = {'kind': 'phaseA_capability_cost_exploratory_clean_checkpoints',
                  'reused_frozen_runs': ['base', 'v2'],
                  'models': {'base': {'repo': 'r', 'revision': 'b'},
                             'v2': {'repo': 'r', 'revision': 'v'},
                             'a1': {'repo': 'r', 'revision': 'a'}},
                  'benchmarks': {ep: dict(entry) for ep in
                                 ('ifeval', 'gsm8k', 'gsm8k_published', 'humaneval', 'mmlu',
                                  'mmlu_forced', 'cybermetric', 'cybermetric_forced')},
                  'xstest_safe_provisional_refusal': {
                      'base': {'n': 250, 'refusal_n': 43}, 'v2': {'n': 250, 'refusal_n': 107},
                      'a1': {'n': 250, 'refusal_n': 85},
                      'a1_vs_v2': {'delta_percentage_points': -8.8,
                                   'paired95_percentage_points': [-13.4, -4.1]}},
                  'limitations': ['A delta whose interval includes 0 is "not detected at this '
                                  'resolution", not evidence of no difference.']}
        text = px.render_readme(scores)
        self.assertIn('not detected at this resolution', text)
        self.assertIn('exploratory', text.lower())
        self.assertIn('-8.8pp', text)
        self.assertNotIn('A2', text.split('## Models')[0].split('\n')[0])


if __name__ == '__main__':
    unittest.main()
