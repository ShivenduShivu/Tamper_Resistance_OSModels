"""Independent offline readback and paired analysis; public output has no answers."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'capability'))
import scoring
from score import verify_directory


def verify_index(root):
    root=Path(root);index=json.loads((root/'artifact_index.json').read_text())
    actual={p.name for p in root.iterdir() if p.is_file() and p.name!='artifact_index.json'}
    if set(index)!=actual:raise ValueError('indexed-file coverage mismatch')
    for name,metadata in index.items():
        if Path(name).name!=name:raise ValueError('invalid indexed path')
        data=(root/name).read_bytes()
        if metadata['bytes']!=len(data) or metadata['sha256']!=hashlib.sha256(data).hexdigest():
            raise ValueError('hash mismatch: '+str(root/name))


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def analyze(panel,roots):
    panel=Path(panel);panel_bytes=(panel/'items.jsonl').read_bytes()
    panel_hash=hashlib.sha256(panel_bytes).hexdigest()
    source={(r['task'],r['id']):r for r in read(panel/'items.jsonl')}
    manifests={};items={};coverage={};scoring_contexts={}
    for model,path in roots.items():
        root=Path(path);manifest=verify_directory(root);verify_index(root)
        if manifest['kind']!='exploratory_clean_model_results':raise ValueError('pilot/partial run rejected')
        if manifest['model']!=model or manifest['panel_items_sha256']!=panel_hash:raise ValueError('wrong model/panel')
        manifests[model]=manifest
        score_root=root/'scoring-v1';verify_index(score_root)
        scoring_summary=json.loads((score_root/'summary.json').read_text())
        provenance=scoring_summary['scoring_provenance']
        scoring_contexts[model]={k:provenance[k] for k in ('classifier_repo','classifier_revision','ifeval_verifier','secondary_math_parser','score_source_sha256','versions')}
        scored=read(score_root/'scored_items.jsonl')
        generation=read(root/'items.jsonl')
        if len(generation)!=len(source) or len(scored)!=len(source):raise ValueError('incomplete generation/scoring')
        bound={(g['task'],g['id']):g for g in generation}
        if set(bound)!=set(source):raise ValueError('wrong generation identities')
        selected=[source[(g['task'],g['id'])] for g in generation]
        if hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()!=manifest['selected_items_sha256']:
            raise ValueError('selected content hash mismatch')
        mapped={}
        for r in scored:
            key=(r['task'],r['id'])
            if key in mapped or key not in source:raise ValueError('duplicate/unknown scoring row')
            for field in ('response','output_tokens','eos_reached','forced_choice_correct','skipped'):
                if r.get(field)!=bound[key].get(field):raise ValueError('scored output not bound to generation')
            mapped[key]=r
        code_root=root/'coding-adapted-v2';verify_index(code_root)
        summary=json.loads((code_root/'summary.json').read_text())
        if summary['state']!='complete_valid' or summary['infra_errors'] or summary['model']!=model:
            raise ValueError('coding infrastructure incomplete')
        if summary['generation_items_sha256']!=hashlib.sha256((root/'items.jsonl').read_bytes()).hexdigest():
            raise ValueError('wrong coding generation')
        code=read(code_root/'items.jsonl')
        expected={key[1] for key in source if key[0]=='humaneval'}
        if len(code)!=164 or {r['id'] for r in code}!=expected:raise ValueError('coding coverage mismatch')
        for r in code:
            if r.get('correct') not in (0,1) or r['status'] not in ('passed','failed','timed_out'):
                raise ValueError('invalid coding verdict')
            mapped[('humaneval',r['id'])]['correct']=r['correct']
            mapped[('humaneval',r['id'])]['endpoint']=summary['protocol']
        items[model]=mapped;coverage[model]={'generation_rows':len(generation),'scored_rows':len(scored),'coding_rows':len(code)}
    if set(roots)!={'base','v2','v2b'}:raise ValueError('three declared conditions required')
    for model,manifest in manifests.items():
        for field in ('protocol','tokenizer','gpu','source_sha256'):
            if manifest[field]!=manifests['base'][field]:raise ValueError('unmatched '+field+' for '+model)
    if any(v!=scoring_contexts['base'] for v in scoring_contexts.values()):
        raise ValueError('models were scored under different protocols/environments')
    report={'kind':'exploratory_clean_retention_results','human_validation':'pending',
        'panel_sha256':panel_hash,'coverage':coverage,'domains':{},'paired':{},
        'limitations':['No noninferiority/equivalence test or claim; comparisons are exploratory, p-values unadjusted.',
            'Refusal labels are provisional AI labels, not validated human truth or causal inability attribution.',
            'IFEval measures instruction compliance, not semantic helpfulness.',
            'Everyday answer quality and unsafe-prompt harmfulness require human labels.',
            'Math sets overlap and are not independent; no pooled general-purpose score.',
            'Clean checkpoints only; no post-tampering capability conclusion.']}
    for task in dict.fromkeys(r['task'] for r in source.values()):
        keys=[key for key in source if key[0]==task]
        report['domains'][task]={}
        for model,mapped in items.items():
            rows=[mapped[key] for key in keys];done=[r for r in rows if r.get('correct') is not None]
            joint=Counter()
            for r in done:
                category=('success' if r['correct']==1 else 'non_success_refusal_flagged' if r.get('classifier_refusal')
                    else 'non_success_unparsed' if r.get('parsed') is False else 'non_success_truncated' if r.get('truncated')
                    else 'non_success_other')
                joint[category]+=1
            report['domains'][task][model]={'n_selected':len(keys),'n_scored':len(done),
                'success_n':sum(r['correct'] for r in done) if done else None,
                'success_pct':100*sum(r['correct'] for r in done)/len(done) if done else None,
                'confirmed_success_pct_of_selected':100*sum(r['correct'] for r in done)/len(keys) if done else None,
                'diagnostic_priority_counts_not_causal':dict(joint),
                'truncated_n':sum(r.get('truncated',False) for r in rows),
                'unparsed_n':sum(r.get('parsed') is False for r in rows),
                'refusal_flagged_n':sum(r.get('classifier_refusal',False) for r in rows),
                'refusal_language_unvalidated_n':sum(r.get('refusal_classifier_language_scope')=='unvalidated_non_english' for r in rows)}
        for model in ('v2','v2b'):
            common=[key for key in keys if items['base'][key].get('correct') is not None and items[model][key].get('correct') is not None]
            if common:
                comparison=scoring.mcnemar([items[model][k]['correct'] for k in common],
                                           [items['base'][k]['correct'] for k in common])
                comparison.update(exploratory=True,excluded_from_pairing=len(keys)-len(common))
                report['paired'][task+':'+model]=comparison
        if task=='gsm8k' or task.startswith('global_mgsm_'):
            for model in ('v2','v2b'):
                comparison=scoring.mcnemar([items[model][k]['published_parser_correct'] for k in keys],
                    [items['base'][k]['published_parser_correct'] for k in keys])
                comparison['status']='secondary_parser_sensitivity_after_pilot'
                report['paired'][task+':published_parser:'+model]=comparison
            for model in items:
                report['domains'][task][model]['secondary_published_parser_success_n']=sum(items[model][k]['published_parser_correct'] for k in keys)
        if task in ('mmlu','cybermetric'):
            for model in ('v2','v2b'):
                common=[k for k in keys if 'forced_choice_correct' in items[model][k] and 'forced_choice_correct' in items['base'][k]]
                if common:
                    result=scoring.mcnemar([items[model][k]['forced_choice_correct'] for k in common],
                        [items['base'][k]['forced_choice_correct'] for k in common])
                    result['excluded_from_pairing']=len(keys)-len(common)
                    report['paired'][task+':forced_choice:'+model]=result
    safe=[key for key,r in source.items() if r['task']=='xstest' and r['prompt_class']=='safe']
    report['xstest_safe_provisional_refusal']={model:{'n':len(safe),
        'refusal_n':sum(mapped[k]['classifier_refusal'] for k in safe)} for model,mapped in items.items()}
    report['paired_xstest_safe_provisional_refusal']={}
    for model in ('v2','v2b'):
        d=scoring.mcnemar([items[model][k]['classifier_refusal'] for k in safe],
                          [items['base'][k]['classifier_refusal'] for k in safe])
        report['paired_xstest_safe_provisional_refusal'][model]={
            'n':d['n'],'delta_percentage_points':100*d['delta'],
            'paired95_percentage_points':[100*v for v in d['delta_ci']],
            'candidate_only_refusal':d['candidate_only_correct'],'base_only_refusal':d['base_only_correct'],
            'p_exact_exploratory':d['mcnemar_p_exact']}
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',required=True);ap.add_argument('--base',required=True)
    ap.add_argument('--v2',required=True);ap.add_argument('--v2b',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();report=analyze(args.panel,{m:getattr(args,m) for m in ('base','v2','v2b')})
    with Path(args.out).open('x') as f:json.dump(report,f,indent=2)
