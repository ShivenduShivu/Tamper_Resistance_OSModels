"""Answer-free summary of blinded AI judgments; never detector accuracy."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_judge import audit, read


def summarize(inputs, labels, key, roots, out):
    check = audit(inputs, labels)
    if check['state'] != 'pass':
        raise ValueError(json.dumps(check))
    identities = json.loads(Path(key).read_text())
    mapping = {r['blind_id']: r for r in identities}
    judged = read(labels)
    if len(mapping) != 150 or set(mapping) != {r['review_id'] for r in judged}:
        raise ValueError('expected all 150 frozen paired identities')
    source = {r['review_id']: r for r in read(inputs)}
    scored = {m: {(r['task'], r['id']): r for r in read(Path(p) / 'scoring-v1/scored_items.jsonl')}
              for m, p in roots.items()}
    rows = []
    counts = {}
    for label in judged:
        k = mapping[label['review_id']]
        s = scored[k['model']][k['task'], k['id']]
        if s['response'] != source[label['review_id']]['response']:
            raise ValueError('judge input not bound to saved generation')
        row = {**{f: k[f] for f in ('model', 'task', 'id', 'stratum')},
               'review_id': label['review_id'],
               **{f: label[f] for f in ('prompt_safety', 'refusal', 'decline_reason_stated',
                                        'usefulness', 'harm', 'confidence', 'output_truncated')},
               'classifier_refusal': s['classifier_refusal'],
               'human_validated': False}
        rows.append(row)
    for stratum in sorted({r['stratum'] for r in rows}):
        counts[stratum] = {}
        for model in roots:
            subset = [r for r in rows if r['stratum'] == stratum and r['model'] == model]
            disagreements = Counter()
            for r in subset:
                if r['refusal'] == 'uncertain':
                    disagreements['excluded_luna_uncertain'] += 1
                    continue
                lunar = r['refusal'] in ('partial', 'full')
                cl = bool(r['classifier_refusal'])
                disagreements['both_refusal' if cl and lunar else 'classifier_only_refusal' if cl
                              else 'luna_only_refusal' if lunar else 'both_no_refusal'] += 1
            counts[stratum][model] = {
                'n': len(subset),
                **{f: dict(Counter(r[f] for r in subset)) for f in
                   ('refusal', 'decline_reason_stated', 'usefulness', 'harm', 'confidence')},
                'truncated_n': sum(r['output_truncated'] for r in subset),
                'classifier_vs_luna_any_refusal': dict(disagreements),
            }
    result = {
        'kind': 'blinded_gpt6_luna_diagnostic_not_human_validation',
        'judge_model': 'gpt-6-luna', 'immutable_revision': 'not exposed by subagent interface',
        'sampling_parameters': 'not exposed by subagent interface',
        'human_validation': 'not performed and not planned', 'n': len(rows),
        'question_n': 50, 'session_blocks': ['S001-S010', 'S011-S030', 'S031-S050'],
        'input_sha256': hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(Path(labels).read_bytes()).hexdigest(),
        'private_key_sha256': hashlib.sha256(Path(key).read_bytes()).hexdigest(),
        'evidence_binding_check': check, 'strata': counts,
        'limitations': [
            'Luna and classifier disagreement is not false-positive/false-negative accuracy.',
            'Any-refusal comparison combines partial and full; both remain separately visible.',
            'Small stratified sample; no general domain or language accuracy claim.',
            'Exact evidence binding is not validation of interpretation or reference truth.',
            'Truncation is frequent and only visible saved text was assessed.',
            'Session split and evidence repairs are disclosed; AI review is exploratory.',
        ],
    }
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'luna_study_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (dest / 'luna_item_labels.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'labels', 'key', 'base', 'v2', 'v2b', 'out'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    summarize(a.inputs, a.labels, a.key, {m: getattr(a, m) for m in ('base', 'v2', 'v2b')}, a.out)
