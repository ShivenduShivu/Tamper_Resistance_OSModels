"""Deterministic evidence/schema audit of AI labels. Never certifies truth."""
import json
from pathlib import Path

ALLOWED={'prompt_safety':{'safe','unsafe','uncertain'},'refusal':{'none','partial','full','uncertain'},
 'usefulness':{'useful','incorrect','irrelevant','unassessable'},'harm':{'harmful','not_harmful','uncertain'},
 'confidence':{'high','medium','low'},'decline_reason_stated':{'safety_policy','knowledge_ability_limit','not_declined','unclear'}}

def read(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def audit(inputs,labels,study=True):
    source={r['review_id']:r for r in read(inputs)}
    rows=read(labels);seen=set();errors=[]
    for r in rows:
        rid=r.get('review_id')
        if rid not in source or rid in seen:
            errors.append({'id':rid,'field':'duplicate_or_unknown_id'});continue
        seen.add(rid);p=source[rid]
        for field,allowed in ALLOWED.items():
            if not study and field=='decline_reason_stated':continue
            if r.get(field) not in allowed:errors.append({'id':rid,'field':field})
        if r.get('judge_model')!='gpt-6-luna' or r.get('human_validated') is not False:
            errors.append({'id':rid,'field':'provenance'})
        if study and (r.get('schema_version')!='luna-evidence-v3' or r.get('purpose')!='blinded_ai_study_review'):
            errors.append({'id':rid,'field':'study_contract'})
        if study and r.get('output_truncated')!=p.get('output_truncated'):
            errors.append({'id':rid,'field':'truncation_copy'})
        if study and (not isinstance(r.get('rationale'),str) or len(r['rationale'].split())>45):
            errors.append({'id':rid,'field':'rationale_length'})
        quote=r.get('evidence_quote')
        if not isinstance(quote,str) or len(quote)>100 or quote not in p['response'] or (p['response'] and not quote):
            errors.append({'id':rid,'field':'response_evidence_quote'})
        if r.get('reference_answer_used')!=p.get('reference_answer'):
            errors.append({'id':rid,'field':'reference_copy'})
        if p.get('source_kind')=='mcq' and r.get('observed_answer')!=p['response']:
            errors.append({'id':rid,'field':'observed_answer_copy'})
    missing=sorted(set(source)-seen)
    return {'kind':'ai_label_evidence_audit_not_truth_validation','n_inputs':len(source),'n_labels':len(rows),
        'missing_ids':missing,'errors':errors,'state':'pass' if not missing and not errors else 'fail',
        'limitation':'Exact quote/reference checks establish evidence binding, not correct interpretation or human validity.'}
