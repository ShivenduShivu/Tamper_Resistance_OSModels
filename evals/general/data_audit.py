"""Freeze an exploratory panel; audit source rows against actual defense training.

Exact normalization and word-shingle overlap are lexical screening, not proof
of semantic disjointness, pretraining cleanliness, or correctness of gold labels.
All item/training content is private; only aggregate audit reports are public.
"""
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


def normalize(s):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC',s).casefold()))


def shingles(s):
    words=normalize(s).split()
    return {tuple(words[i:i+5]) for i in range(max(0,len(words)-4))}


def overlap(text, corpus, threshold=.8):
    n=normalize(text)
    exact=[i for i,s in enumerate(corpus) if n==normalize(s)]
    q=shingles(text)
    near=[]
    if q:
        for i,s in enumerate(corpus):
            if i in exact: continue
            t=shingles(s)
            score=len(q&t)/len(q|t) if t else 0
            if score>=threshold: near.append({'row':i,'jaccard':round(score,6)})
    return exact,near


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def freeze(root, lock_path):
    import sys
    sys.path.insert(0,'/suite')
    import tasks
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    lock=json.loads(Path(lock_path).read_text()); rev=lock['datasets']
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    raw=Path(hf_hub_download('StarWarrior24x7/tiny-aya-extended-refusal','train.jsonl',
        repo_type='dataset',revision=rev['StarWarrior24x7/tiny-aya-extended-refusal'])).read_bytes()
    train=[json.loads(l) for l in raw.splitlines() if l.strip()]
    if not train or any(not isinstance(r.get('prompt'),str) for r in train):
        raise ValueError('training corpus schema invalid')
    corpus=[r['prompt'] for r in train]
    norm_corpus={normalize(s) for s in corpus}
    # Inverted 5-gram index: candidate generation, followed by exact Jaccard.
    train_grams=[shingles(s) for s in corpus]
    inverted=collections.defaultdict(set)
    for i,grams in enumerate(train_grams):
        for gram in grams: inverted[gram].add(i)
    def screen(text):
        grams=shingles(text); candidates=set()
        for gram in grams: candidates.update(inverted.get(gram,()))
        near=[]
        for i in sorted(candidates):
            other=train_grams[i]; similarity=len(grams&other)/len(grams|other)
            if similarity>=.8: near.append({'training_row':i,'jaccard':similarity})
        return normalize(text) in norm_corpus, near
    packs={}; report={'kind':'data_audit_not_model_results','lock':lock,
        'training':{'n':len(train),'sha256':hashlib.sha256(raw).hexdigest(),
                    'kinds':dict(collections.Counter(r.get('kind','MISSING') for r in train)),
                    'normalized_duplicate_n':len(corpus)-len(norm_corpus)},'tasks':{},
        'limitations':['5-word-shingle Jaccard >=0.8 screens lexical overlap only; no semantic/translation assurance.',
          'Original pretraining corpus unavailable. Screening covers published defense train.jsonl only.',
          'v2b border.jsonl is not in published train.jsonl. OR-Bench source pools are screened separately.',
          'Benchmark labels and reference answers have not been human-validated by this audit.']}
    meta={}
    for name,fn,n,repo in [('mmlu',tasks.mmlu_load,200,'cais/mmlu'),
                         ('gsm8k',tasks.gsm8k_load,200,'openai/gsm8k'),
                         ('cybermetric',tasks.cybermetric_load,500,'tihanyin/CyberMetric')]:
        rows=fn(n=n,seed=0,revision=rev[repo],meta_out=meta)
        packs[name]=[dict(r,task=name,prompt=(tasks.gsm8k_prompt(r) if name=='gsm8k' else tasks.mmlu_prompt(r)),
                           overlap_text=r['question'],kind='math' if name=='gsm8k' else 'mcq') for r in rows]
    rows=tasks.humaneval_load(revision=rev['openai/openai_humaneval'])
    packs['humaneval']=[dict(r,task='humaneval',prompt=tasks.humaneval_prompt(r),overlap_text=r['prompt'],kind='code') for r in rows]
    # Official IFEval input, all instruction types; scoring is a later pinned step.
    file=hf_hub_download('google/IFEval','ifeval_input_data.jsonl',repo_type='dataset',revision=rev['google/IFEval'])
    rows=[json.loads(l) for l in Path(file).read_text().splitlines() if l.strip()]
    random.Random(0).shuffle(rows)
    packs['ifeval']=[dict(r,id=str(r['key']),task='ifeval',kind='instructions',overlap_text=r['prompt']) for r in rows[:200]]
    # Native human-authored evaluation, not Dolly-derived prompts used in training.
    ds=load_dataset('CohereLabs/aya_evaluation_suite','aya_human_annotated',split='test',revision=rev['CohereLabs/aya_evaluation_suite'])
    groups=collections.defaultdict(list)
    for i,r in enumerate(ds): groups[r['language']].append((i,r))
    packs['everyday_aya']=[]
    report['everyday_language_population']={k:len(v) for k,v in sorted(groups.items())}
    for language,rows in sorted(groups.items()):
        random.Random(0).shuffle(rows)
        for i,r in rows[:20]:
            packs['everyday_aya'].append(dict(id=f'test#{i}',task='everyday_aya',kind='human_pending',
                prompt=r['inputs'],reference=r['targets'],language=language,overlap_text=r['inputs']))
    for language in ('en','fr','ar'):
        ds=load_dataset('CohereLabs/global-mgsm',language,split='test',revision=rev['CohereLabs/global-mgsm'])
        indices=list(range(len(ds)));random.Random(0).shuffle(indices)
        name='global_mgsm_'+language
        packs[name]=[dict(id=f'test#{i}',task=name,kind='math',question=ds[i]['question'],
            prompt=ds[i]['instruction']+'\n\n'+ds[i]['question'],answer=ds[i]['answer'],
            answer_prefix=ds[i]['answer_prefix'],language=language,overlap_text=ds[i]['question']) for i in indices[:50]]
    ds=load_dataset('natolambert/xstest-v2-copy',split='prompts',revision=rev['natolambert/xstest-v2-copy'])
    packs['xstest']=[dict(id=f'prompts#{i}',task='xstest',kind='refusal_pending',prompt=r['prompt'],
        prompt_class='unsafe' if r['type'].startswith('contrast') else 'safe',overlap_text=r['prompt']) for i,r in enumerate(ds)]
    # Reconstruct the v2b prompt SOURCE pool, not the unknown accepted subset.
    import csv
    or_pools=[]
    for file,cut in [('or-bench-hard-1k.csv',slice(200,None)),('or-bench-80k.csv',slice(0,500))]:
        p=hf_hub_download('bench-llm/or-bench',file,repo_type='dataset',revision=rev['bench-llm/or-bench'])
        with open(p) as f: or_pools.extend(r['prompt'] for r in list(csv.DictReader(f))[cut])
    or_norm={normalize(s) for s in or_pools}
    report['v2b_source_pool']={'n':len(or_pools),'revision':rev['bench-llm/or-bench'],
        'limitation':'Source revision used in training was not pinned. Current pinned pool is a reconstruction, not verified training provenance.'}
    excluded=[]; all_rows=[]
    for name,rows in packs.items():
        seen=set();kept=[];counts=collections.Counter()
        for r in rows:
            if not r['prompt'].strip(): raise ValueError('empty prompt')
            identity=normalize(r['overlap_text'])
            if identity in seen:
                # Preserve standard benchmark cases: punctuation-normalized
                # similarity is not semantic identity (especially for code).
                counts['normalized_prompt_repeat_retained']+=1
            seen.add(identity)
            exact,near=screen(r['overlap_text'])
            border=identity in or_norm
            if exact or near or border:
                reason='training_exact' if exact else 'training_lexical_near' if near else 'v2b_source_pool_exact'
                counts[reason]+=1;excluded.append({'task':name,'id':r['id'],'reason':reason,'near':near});continue
            kept.append(r)
        if not kept: raise ValueError('empty after screening: '+name)
        report['tasks'][name]={'selected_before_screening':len(rows),'selected_after_screening':len(kept),
            'screening_flags':dict(counts),'excluded_n':len(rows)-len(kept),'rows_sha256':sha(kept),
            'gold_scoring':'pending_human' if name in ('everyday_aya','xstest') else 'frozen_automatic_protocol'}
        all_rows.extend(kept)
    report['cross_pack_normalized_overlap_pairs']=[]
    cross=collections.defaultdict(list)
    for r in all_rows: cross[normalize(r['overlap_text'])].append((r['task'],r['id']))
    for identities in cross.values():
        if len(identities)>1:report['cross_pack_normalized_overlap_pairs'].append(identities)
    (root/'items.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows))
    (root/'excluded.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
    report['items_file_sha256']=hashlib.sha256((root/'items.jsonl').read_bytes()).hexdigest()
    (root/'data_audit.json').write_text(json.dumps(report,indent=2))
    return report
