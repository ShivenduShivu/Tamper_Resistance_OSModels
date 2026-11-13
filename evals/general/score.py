"""Frozen automatic scoring and provisional refusal labels, never human truth."""
import hashlib
import json
import re
import sys
from pathlib import Path


def verify_directory(root):
    root=Path(root)
    index=json.loads((root/'artifact_index.json').read_text())
    for name,meta in index.items():
        if Path(name).name!=name:raise ValueError('unsafe index path')
        blob=(root/name).read_bytes()
        if len(blob)!=meta['bytes'] or hashlib.sha256(blob).hexdigest()!=meta['sha256']:
            raise ValueError('artifact integrity mismatch: '+name)
    if not {'manifest.json','items.jsonl','COMPLETE'}<=set(index):raise ValueError('incomplete run')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['state']!='complete':raise ValueError('generation did not complete')
    return manifest


def score_one(row,answer):
    import tasks
    if row['kind']=='mcq':
        match=re.fullmatch(r'\s*(?:Answer\s*:\s*)?([ABCD])[.)]?\s*',answer,re.I)
        return {'correct':int(bool(match) and 'ABCD'.index(match.group(1).upper())==row['answer']),
                'parsed':bool(match),'endpoint':'strict_generated_letter'}
    if row['task']=='gsm8k':
        detail=tasks.gsm8k_score_detail(row,answer)
        return {'correct':detail['correct'],'parsed':detail['route']!='none',
                'route':detail['route'],'endpoint':'frozen_gsm8k_exact_match'}
    if row['task'].startswith('global_mgsm_'):
        prefix=row['answer_prefix'].strip().rstrip(':：')
        matches=list(re.finditer(re.escape(prefix)+r'\s*[:：]\s*',answer,re.I))
        line=answer[matches[-1].end():].splitlines()[0] if matches and answer[matches[-1].end():] else ''
        value=tasks._to_number(line)
        from fractions import Fraction
        gold=Fraction(str(row['answer']).replace(',',''))
        return {'correct':int(value is not None and value==gold),'parsed':value is not None,
                'endpoint':'declared_language_answer_prefix_exact_match'}
    if row['task']=='ifeval':
        from instruction_following_eval import evaluation_lib as lib
        inp=lib.InputExample(key=row['key'],instruction_id_list=row['instruction_id_list'],
                             prompt=row['prompt'],kwargs=row['kwargs'])
        verdict=lib.test_instruction_following_strict(inp,{row['prompt']:answer})
        return {'correct':int(verdict.follow_all_instructions),'parsed':True,
                'instruction_results':verdict.follow_instruction_list,'endpoint':'official_ifeval_prompt_strict'}
    return {'correct':None,'endpoint':'pending_execution' if row['kind']=='code' else 'pending_human_review'}


def score_run(panel,run_dir,classifier=None):
    root=Path(run_dir);manifest=verify_directory(root)
    data=Path(panel,'items.jsonl').read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest['panel_items_sha256']:raise ValueError('wrong panel')
    source={(r['task'],r['id']):r for r in (json.loads(l) for l in data.splitlines() if l.strip())}
    generated=[json.loads(l) for l in (root/'items.jsonl').read_text().splitlines() if l.strip()]
    if len(generated)!=sum(manifest['tasks'].values()):raise ValueError('generation coverage mismatch')
    seen=set();out=[]
    for g in generated:
        identity=(g['task'],g['id'])
        if identity in seen or identity not in source:raise ValueError('duplicate/unknown item')
        seen.add(identity)
        if 'skipped' in g:
            out.append(dict(g,correct=None));continue
        r=source[identity];d=score_one(r,g['response'])
        out.append(dict(g,**d,language=r.get('language','en'),
            refusal_classifier_language_scope='unvalidated_non_english' if r.get('language','en') not in ('en','eng') else 'provisional_english',
            empty=not bool(g['response'].strip()),truncated=not g['eos_reached']))
    expected_rows=[source[(g['task'],g['id'])] for g in generated]
    if hashlib.sha256(json.dumps(expected_rows,sort_keys=True).encode()).hexdigest()!=manifest['selected_items_sha256']:
        raise ValueError('generated item order/content differs from frozen selection')
    if {k:sum(r['task']==k for r in out) for k in manifest['tasks']}!=manifest['tasks']:
        raise ValueError('per-task generation coverage mismatch')
    if classifier:
        pending=[r for r in out if 'response' in r]
        for i in range(0,len(pending),32):
            batch=pending[i:i+32]
            labels=classifier([r['response'] or '.' for r in batch],batch_size=32,truncation=True,max_length=512)
            for r,pred in zip(batch,labels):
                r['classifier_refusal']=pred['label']=='REJECTION'
                r['classifier_confidence']=pred['score']
                r['classifier_input_truncated']=len(classifier.tokenizer.encode(r['response'],add_special_tokens=True))>512
    summary={'kind':'exploratory_pilot' if manifest['kind']=='pilot_not_confirmatory' else 'exploratory_clean_results',
             'model':manifest['model'],'run_id':root.name,'human_validation':'not_collected',
             'refusal_warning':'ProtectAI labels are provisional; not an attribution of inability or a human decomposition.',
             'tasks':{}}
    for task,n in manifest['tasks'].items():
        pack=[r for r in out if r['task']==task]
        scored=[r for r in pack if r.get('correct') is not None]
        summary['tasks'][task]={'n_selected':n,'n_scored':len(scored),
            'correct_n':sum(r['correct'] for r in scored) if scored else None,
            'success_rate':sum(r['correct'] for r in scored)/n if scored else None,
            'skipped_n':sum('skipped' in r for r in pack),'empty_n':sum(r.get('empty',False) for r in pack),
            'truncated_n':sum(r.get('truncated',False) for r in pack),
            'unparsed_n':sum(r.get('parsed') is False for r in pack),
            'classifier_refusal_n':sum(r.get('classifier_refusal',False) for r in pack),
            'classifier_labeled_n':sum('classifier_refusal' in r for r in pack),
            'classifier_non_english_unvalidated_n':sum(r.get('refusal_classifier_language_scope')=='unvalidated_non_english' for r in pack),
            'correct_and_classifier_refusal_n':sum(r.get('correct')==1 and r.get('classifier_refusal',False) for r in pack),
            'unsuccessful_and_classifier_refusal_n':sum(r.get('correct')==0 and r.get('classifier_refusal',False) for r in pack)}
        if any('forced_choice_correct' in r for r in pack):
            summary['tasks'][task]['forced_choice_correct_n']=sum(r.get('forced_choice_correct',0) for r in pack)
    return out,summary
