"""Independent schema/leakage checks on an externally downloaded frozen panel."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'capability'))
import tasks


def audit(panel):
    root=Path(panel);data=(root/'items.jsonl').read_bytes()
    manifest=json.loads((root/'data_audit.json').read_text())
    if hashlib.sha256(data).hexdigest()!=manifest['items_file_sha256']:raise ValueError('panel checksum mismatch')
    rows=[json.loads(l) for l in data.splitlines() if l.strip()]
    result={'kind':'independent_selected_data_checks',
        'duplicate_task_ids':len(rows)-len({(r['task'],r['id']) for r in rows}),
        'mcq_ambiguous_duplicate_choices':[],'math_gold_unparseable':[],
        'ifeval_bad_structure':[],'code_prompt_leak_n':0}
    for r in rows:
        if not isinstance(r['prompt'],str) or not r['prompt'].strip():raise ValueError('invalid prompt')
        if r['kind']=='mcq':
            if len({c.strip().casefold() for c in r['choices']})!=len(r['choices']):
                result['mcq_ambiguous_duplicate_choices'].append([r['task'],r['id']])
            if type(r['answer']) is not int or not 0<=r['answer']<len(r['choices']):raise ValueError('invalid MCQ gold')
        if r['kind']=='math' and tasks._to_number(str(r['answer'])) is None:
            result['math_gold_unparseable'].append([r['task'],r['id']])
        if r['task']=='ifeval' and (not r['instruction_id_list'] or len(r['instruction_id_list'])!=len(r['kwargs'])):
            result['ifeval_bad_structure'].append(r['id'])
        if r['kind']=='code':
            result['code_prompt_leak_n']+=int(r['canonical_solution'] in r['prompt'] or r['test'] in r['prompt'])
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('panel');ap.add_argument('--out',required=True)
    args=ap.parse_args()
    with Path(args.out).open('x') as f:json.dump(audit(args.panel),f,indent=2)
