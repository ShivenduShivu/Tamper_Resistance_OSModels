"""Export answer-free verdicts and provenance after independent verification."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze import analyze, read


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(panel, roots, destination):
    report = analyze(panel, roots)  # Fail closed before exporting any evidence.
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'clean_results.json').write_text(json.dumps(report, indent=2) + '\n')
    selected = read(Path(panel) / 'items.jsonl')
    public_items = []
    for row in selected:
        public_items.append({
            'task': row['task'], 'id': row['id'],
            'item_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
            'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
            **{k: row[k] for k in ('language', 'prompt_class') if k in row},
        })
    write_rows(out / 'selected_items.jsonl', public_items)
    verdicts = []
    provenance = {'kind': 'verified_private_evidence_public_provenance', 'models': {}}
    fields = ('task', 'id', 'correct', 'parsed', 'truncated', 'output_tokens',
              'eos_reached', 'classifier_refusal', 'refusal_classifier_language_scope',
              'published_parser_correct', 'forced_choice_correct', 'skipped')
    for model, path in roots.items():
        root = Path(path)
        code = {r['id']: r for r in read(root / 'coding-adapted-v2/items.jsonl')}
        for row in read(root / 'scoring-v1/scored_items.jsonl'):
            record = {'model': model, **{k: row[k] for k in fields if k in row}}
            if row['task'] == 'humaneval':
                c = code[row['id']]
                record.update(correct=c['correct'], execution_status=c['status'],
                              adapter_route=c.get('adapter_route'))
            verdicts.append(record)
        provenance['models'][model] = {
            'run_id': root.name,
            'manifest': json.loads((root / 'manifest.json').read_text()),
            'private_artifact_indexes': {
                sub or 'generation': json.loads((root / sub / 'artifact_index.json').read_text())
                for sub in ('', 'scoring-v1', 'coding-adapted-v2')},
            'scoring_provenance': json.loads((root / 'scoring-v1/summary.json').read_text())['scoring_provenance'],
            'coding_summary': json.loads((root / 'coding-adapted-v2/summary.json').read_text()),
        }
    write_rows(out / 'item_verdicts.jsonl', verdicts)
    (out / 'run_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    names = ('clean_results.json', 'selected_items.jsonl', 'item_verdicts.jsonl', 'run_provenance.json')
    (out / 'public_artifact_index.json').write_text(json.dumps({
        name: {'bytes': (out / name).stat().st_size, 'sha256': digest(out / name)} for name in names
    }, indent=2) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel', required=True)
    for model in ('base', 'v2', 'v2b'):
        ap.add_argument('--' + model, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    export(args.panel, {m: getattr(args, m) for m in ('base', 'v2', 'v2b')}, args.out)
