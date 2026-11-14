"""Phase A capability cost export: defended checkpoints A1/A2 against frozen v2/base.

Reuses the frozen PR #7 panel (exploratory-panel-20261113-v2) and the frozen
full-20261113-v2-{base,v2} generation/scoring/coding artifacts, so base and v2
rows are byte-identical to the published audit. A1/A2 come from new runs of the
unchanged harness (same lock.json, protocol, tokenizer and scorers). All
comparisons are exploratory clean-checkpoint measurements; no human validation
of refusal labels exists. XSTest safe-prompt responses are shareable per team
policy (safe prompts only); every other raw answer stays private.

Usage:
  python evals/general/phaseA_export.py --panel PANEL_DIR \
      --base RUN_DIR --v2 RUN_DIR [--a1 RUN_DIR] [--a2 RUN_DIR] --out OUT_DIR
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'capability'))

from score import verify_directory  # noqa: E402
import scoring  # noqa: E402

SCORED_TASKS = ('mmlu', 'cybermetric', 'gsm8k', 'ifeval')
DISPLAY = {
    'mmlu': 'MMLU generated letter',
    'mmlu_forced': 'MMLU forced choice (secondary)',
    'cybermetric': 'CyberMetric generated letter',
    'cybermetric_forced': 'CyberMetric forced choice (secondary)',
    'gsm8k': 'GSM8K strict',
    'gsm8k_published': 'GSM8K published numeric parser (secondary)',
    'ifeval': 'IFEval prompt strict',
    'humaneval': 'HumanEval pass@1 (isolated backend)',
    'xstest_safe': 'XSTest safe-prompt classifier refusal (provisional)',
}


def verify_index(root):
    root = Path(root)
    index = json.loads((root / 'artifact_index.json').read_text())
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != 'artifact_index.json'}
    if set(index) != actual:
        raise ValueError('indexed-file coverage mismatch: ' + str(root))
    for name, metadata in index.items():
        if Path(name).name != name:
            raise ValueError('invalid indexed path')
        data = (root / name).read_bytes()
        if metadata['bytes'] != len(data) or metadata['sha256'] != hashlib.sha256(data).hexdigest():
            raise ValueError('hash mismatch: ' + str(root / name))


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_model(model, path, panel_source, panel_hash):
    root = Path(path)
    manifest = verify_directory(root)
    verify_index(root)
    if manifest['kind'] != 'exploratory_clean_model_results':
        raise ValueError('pilot/partial run rejected: ' + model)
    if manifest['model'] != model or manifest['panel_items_sha256'] != panel_hash:
        raise ValueError('wrong model/panel: ' + model)
    score_root = root / 'scoring-v1'
    verify_index(score_root)
    scoring_summary = json.loads((score_root / 'summary.json').read_text())
    scored = read(score_root / 'scored_items.jsonl')
    generation = read(root / 'items.jsonl')
    if len(generation) != len(panel_source) or len(scored) != len(panel_source):
        raise ValueError('incomplete generation/scoring: ' + model)
    bound = {(g['task'], g['id']): g for g in generation}
    if set(bound) != set(panel_source):
        raise ValueError('wrong generation identities: ' + model)
    mapped = {}
    for r in scored:
        key = (r['task'], r['id'])
        if key in mapped or key not in panel_source:
            raise ValueError('duplicate/unknown scoring row: ' + model)
        for field in ('response', 'output_tokens', 'eos_reached', 'skipped'):
            if r.get(field) != bound[key].get(field):
                raise ValueError('scored output not bound to generation: ' + model)
        mapped[key] = r
    code_root = root / 'coding-adapted-v2'
    verify_index(code_root)
    summary = json.loads((code_root / 'summary.json').read_text())
    if summary['state'] != 'complete_valid' or summary['infra_errors'] or summary['model'] != model:
        raise ValueError('coding infrastructure incomplete: ' + model)
    if summary['generation_items_sha256'] != hashlib.sha256((root / 'items.jsonl').read_bytes()).hexdigest():
        raise ValueError('wrong coding generation: ' + model)
    gate = summary['backend_gate']
    if gate['state'] != 'pass' or gate['canonical_passed'] != 164 or gate['canonical_n'] != 164 \
            or gate['authored_fixture_failures']:
        raise ValueError('coding backend gate not valid: ' + model)
    code = {r['id']: r for r in read(code_root / 'items.jsonl')}
    expected = {key[1] for key in panel_source if key[0] == 'humaneval'}
    if len(code) != 164 or set(code) != expected:
        raise ValueError('coding coverage mismatch: ' + model)
    for pid, r in code.items():
        if r.get('correct') not in (0, 1):
            raise ValueError('invalid coding verdict: ' + model)
        mapped[('humaneval', pid)]['correct'] = r['correct']
        mapped[('humaneval', pid)]['code_status'] = r['status']
    return {
        'manifest': manifest,
        'items': mapped,
        'scoring_provenance': {k: scoring_summary['scoring_provenance'][k] for k in
                               ('classifier_repo', 'classifier_revision', 'score_source_sha256')},
        'coding_protocol': summary['protocol'],
    }


def success_getter(endpoint):
    if endpoint == 'mmlu':
        return lambda r: r['correct']
    if endpoint == 'mmlu_forced':
        return lambda r: r.get('forced_choice_correct')
    if endpoint == 'cybermetric':
        return lambda r: r['correct']
    if endpoint == 'cybermetric_forced':
        return lambda r: r.get('forced_choice_correct')
    if endpoint == 'gsm8k':
        return lambda r: r['correct']
    if endpoint == 'gsm8k_published':
        return lambda r: r.get('published_parser_correct')
    if endpoint == 'ifeval':
        return lambda r: r['correct']
    if endpoint == 'humaneval':
        return lambda r: r['correct']
    raise ValueError(endpoint)


def endpoint_ids(panel_source, endpoint):
    if endpoint.startswith('mmlu'):
        return sorted(pid for (t, pid) in panel_source if t == 'mmlu')
    if endpoint.startswith('cybermetric'):
        return sorted(pid for (t, pid) in panel_source if t == 'cybermetric')
    if endpoint.startswith('gsm8k'):
        return sorted(pid for (t, pid) in panel_source if t == 'gsm8k')
    if endpoint == 'ifeval':
        return sorted(pid for (t, pid) in panel_source if t == 'ifeval')
    if endpoint == 'humaneval':
        return sorted(pid for (t, pid) in panel_source if t == 'humaneval')
    raise ValueError(endpoint)


def xstest_safe_ids(panel_source):
    return sorted(pid for (t, pid), r in panel_source.items()
                  if t == 'xstest' and r.get('prompt_class') == 'safe')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--v2', required=True)
    ap.add_argument('--a1')
    ap.add_argument('--a2')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    panel_bytes = (Path(args.panel) / 'items.jsonl').read_bytes()
    panel_hash = hashlib.sha256(panel_bytes).hexdigest()
    panel_rows = read(Path(args.panel) / 'items.jsonl')
    panel_source = {(r['task'], r['id']): r for r in panel_rows}

    run_args = {'base': args.base, 'v2': args.v2, 'a1': args.a1, 'a2': args.a2}
    models = {}
    for name, path in run_args.items():
        if path:
            models[name] = load_model(name, path, panel_source, panel_hash)
    if 'a1' not in models and 'a2' not in models:
        raise ValueError('at least one of --a1/--a2 is required')

    # Protocol parity across every run: same generation settings or no pairing.
    reference = models['v2']['manifest']
    for name, m in models.items():
        for field in ('protocol', 'tokenizer', 'effective_eos_ids'):
            if m['manifest'][field] != reference[field]:
                raise ValueError('protocol mismatch between v2 and ' + name + ': ' + field)

    endpoints = ('mmlu', 'mmlu_forced', 'cybermetric', 'cybermetric_forced',
                 'gsm8k', 'gsm8k_published', 'ifeval', 'humaneval')
    safe_ids = xstest_safe_ids(panel_source)
    if len(safe_ids) != 250:
        raise ValueError('expected 250 XSTest safe prompts')

    table = {}
    for endpoint in endpoints:
        get = success_getter(endpoint)
        ids = endpoint_ids(panel_source, endpoint)
        entry = {}
        vectors = {}
        for name, m in models.items():
            task = 'mmlu' if endpoint.startswith('mmlu') else \
                   'cybermetric' if endpoint.startswith('cybermetric') else \
                   'gsm8k' if endpoint.startswith('gsm8k') else endpoint
            vec = [get(m['items'][(task, pid)]) for pid in ids]
            if any(v is None for v in vec):
                raise ValueError('missing endpoint value: ' + name + ' ' + endpoint)
            vectors[name] = [int(v) for v in vec]
            entry[name] = {'n': len(ids), 'success_n': sum(vectors[name])}
        for cand in ('a1', 'a2'):
            if cand not in vectors:
                continue
            for ref in ('v2', 'base'):
                if ref in vectors:
                    entry[cand + '_vs_' + ref] = scoring.mcnemar(vectors[cand], vectors[ref])
        if 'v2' in vectors and 'base' in vectors:
            entry['v2_vs_base'] = scoring.mcnemar(vectors['v2'], vectors['base'])
        table[endpoint] = entry

    # XSTest safe-prompt provisional classifier refusal.
    xs = {}
    xs_vectors = {}
    for name, m in models.items():
        vec = [int(bool(m['items'][('xstest', pid)]['classifier_refusal'])) for pid in safe_ids]
        xs_vectors[name] = vec
        xs[name] = {'n': 250, 'refusal_n': sum(vec)}
    for cand in ('a1', 'a2'):
        if cand not in xs_vectors:
            continue
        for ref in ('v2', 'base'):
            if ref in xs_vectors:
                res = scoring.mcnemar(xs_vectors[cand], xs_vectors[ref])
                xs[cand + '_vs_' + ref] = {
                    'n': 250,
                    'delta_percentage_points': res['delta'] * 100,
                    'paired95_percentage_points': [res['delta_ci'][0] * 100, res['delta_ci'][1] * 100],
                    'candidate_only_refusal': res['candidate_only_correct'],
                    'base_only_refusal': res['base_only_correct'],
                    'p_exact_exploratory': res['mcnemar_p_exact'],
                }
    if 'v2' in xs_vectors and 'base' in xs_vectors:
        res = scoring.mcnemar(xs_vectors['v2'], xs_vectors['base'])
        xs['v2_vs_base'] = {
            'n': 250,
            'delta_percentage_points': res['delta'] * 100,
            'paired95_percentage_points': [res['delta_ci'][0] * 100, res['delta_ci'][1] * 100],
            'candidate_only_refusal': res['candidate_only_correct'],
            'base_only_refusal': res['base_only_correct'],
            'p_exact_exploratory': res['mcnemar_p_exact'],
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    scores = {
        'kind': 'phaseA_capability_cost_exploratory_clean_checkpoints',
        'human_validation': 'none; classifier refusal labels are provisional',
        'panel_sha256': panel_hash,
        'panel_id': models['v2']['manifest']['panel_id'],
        'models': {name: {'repo': m['manifest']['repo'], 'revision': m['manifest']['revision'],
                          'run_elapsed_seconds': m['manifest'].get('elapsed_seconds'),
                          'gpu': m['manifest'].get('gpu'),
                          'source_sha256': m['manifest']['source_sha256']}
                   for name, m in models.items()},
        'reused_frozen_runs': ['base', 'v2'],
        'benchmarks': table,
        'xstest_safe_provisional_refusal': xs,
        'limitations': [
            'Exploratory paired comparisons; no preregistered noninferiority margin.',
            'Clean checkpoints only: post-tampering useful-task retention unmeasured.',
            'Refusal labels are ProtectAI classifier output under a 128-token generation cap; '
            'many answers truncate; classifier flags are not human-validated withholding judgments.',
            'CyberMetric is MCQ security knowledge; IFEval is instruction compliance.',
            'FP16 near-ties can vary with hardware/batching; recorded settings are bound per run.',
            'base and v2 rows are the frozen PR #7 artifacts re-read, not new generations.',
        ],
    }
    (out / 'phaseA_capability_scores.json').write_text(json.dumps(scores, indent=2) + '\n')

    with (out / 'xstest_safe_responses.jsonl').open('x') as stream:
        for pid in safe_ids:
            prompt = panel_source[('xstest', pid)]['prompt']
            for name in ('base', 'v2', 'a1', 'a2'):
                if name not in models:
                    continue
                r = models[name]['items'][('xstest', pid)]
                stream.write(json.dumps({
                    'id': pid, 'prompt_class': 'safe', 'prompt': prompt, 'model': name,
                    'model_repo': models[name]['manifest']['repo'],
                    'model_revision': models[name]['manifest']['revision'],
                    'response': r['response'], 'output_tokens': r['output_tokens'],
                    'eos_reached': r['eos_reached'], 'truncated': r['truncated'],
                    'classifier_refusal': r['classifier_refusal'],
                    'classifier_confidence': r['classifier_confidence'],
                    'classifier_input_truncated': r['classifier_input_truncated'],
                }, ensure_ascii=False) + '\n')

    (out / 'README.md').write_text(render_readme(scores))
    print(json.dumps({'out': str(out), 'models': sorted(models),
                      'benchmarks': {k: {m: v[m] for m in ('base', 'v2', 'a1', 'a2') if m in v}
                                     for k, v in table.items()},
                      'xstest': {m: xs[m] for m in ('base', 'v2', 'a1', 'a2') if m in xs}},
                     indent=2))


def fmt_cell(entry, name, endpoint):
    if name not in entry:
        return '-'
    n = entry[name]['success_n']
    total = entry[name]['n']
    return f"{n}/{total} ({100 * n / total:.1f}%)"


def fmt_delta(res, scale=100):
    if res is None:
        return '-'
    d = res['delta'] * scale
    lo, hi = res['delta_ci'][0] * scale, res['delta_ci'][1] * scale
    return f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def render_readme(scores):
    table = scores['benchmarks']
    xs = scores['xstest_safe_provisional_refusal']
    have_a1 = any(k.startswith('a1') for v in table.values() for k in v)
    have_a2 = any(k.startswith('a2') for v in table.values() for k in v)
    cand_cols = [c for c, have in (('a1', have_a1), ('a2', have_a2)) if have]
    lines = []
    lines.append('# Phase A capability cost: defended checkpoints A1' + ('/A2' if have_a2 else '')
                 + ' on the frozen clean panel')
    lines.append('')
    lines.append('Same harness, settings, prompts, parsers and scorers as the merged capability '
                 'audit (PR #7). base and v2 rows are the frozen PR #7 artifacts re-read '
                 '(`full-20261113-v2-{base,v2}`), so these comparisons stay exactly paired; '
                 + ' and '.join(c.upper() for c in cand_cols)
                 + ' ran the unchanged pipeline on the pinned revisions below. '
                 'All comparisons are exploratory clean-checkpoint measurements with paired '
                 'Newcombe 95% intervals; they do not establish equivalence or post-tampering '
                 'retention. No human validation of refusal labels was performed or planned.')
    lines.append('')
    header = '| Benchmark | base | v2 |'
    sep = '|---|---:|---:|'
    for c in cand_cols:
        header += f' {c.upper()} |'
        sep += '---:|'
    for c in cand_cols:
        header += f' {c.upper()} - v2 [95% CI] |'
        sep += '---:|'
    lines.append(header)
    lines.append(sep)
    for endpoint in ('ifeval', 'gsm8k', 'gsm8k_published', 'humaneval', 'mmlu', 'mmlu_forced',
                     'cybermetric', 'cybermetric_forced'):
        entry = table[endpoint]
        row = '| ' + DISPLAY[endpoint] + ' |'
        for name in ('base', 'v2'):
            row += ' ' + fmt_cell(entry, name, endpoint) + ' |'
        for c in cand_cols:
            row += ' ' + fmt_cell(entry, c, endpoint) + ' |'
        for c in cand_cols:
            row += ' ' + fmt_delta(entry.get(c + '_vs_v2')) + ' |'
        lines.append(row)
    row = '| ' + DISPLAY['xstest_safe'] + ' |'
    for name in ('base', 'v2'):
        row += f" {xs[name]['refusal_n']}/{xs[name]['n']} ({100 * xs[name]['refusal_n'] / xs[name]['n']:.1f}%) |"
    for c in cand_cols:
        row += f" {xs[c]['refusal_n']}/{xs[c]['n']} ({100 * xs[c]['refusal_n'] / xs[c]['n']:.1f}%) |"
    for c in cand_cols:
        res = xs.get(c + '_vs_v2')
        if res:
            d = res['delta_percentage_points']
            lo, hi = res['paired95_percentage_points']
            row += f" {d:+.1f}pp [{lo:+.1f}, {hi:+.1f}] |"
        else:
            row += ' - |'
    lines.append(row)
    lines.append('')
    lines.append('Delta units are percentage points for accuracy endpoints and for the refusal '
                 'rate. Full paired statistics (discordant counts, exact McNemar p, secondary '
                 'deltas against base) are in `phaseA_capability_scores.json`.')
    lines.append('')
    lines.append('## Models')
    lines.append('')
    for name, m in scores['models'].items():
        reused = ' (frozen PR #7 artifacts re-read)' if name in scores['reused_frozen_runs'] \
                 else ' (new run, unchanged harness)'
        lines.append(f"- `{name}`: `{m['repo']}` @ `{m['revision']}`{reused}")
    lines.append('')
    lines.append('## XSTest safe-prompt responses')
    lines.append('')
    lines.append('`xstest_safe_responses.jsonl` holds every 250 safe-prompt response per model '
                 'with truncation and classifier metadata. Prompts are safe by construction, so '
                 'responses are shareable. Refusal labels are provisional ProtectAI classifier '
                 'output under a 128-token generation cap; many answers truncate, and a '
                 'classifier flag is not a human-validated withholding judgment.')
    lines.append('')
    lines.append('## Limitations')
    lines.append('')
    for lim in scores['limitations']:
        lines.append('- ' + lim)
    lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
