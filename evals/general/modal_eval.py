"""Clean-model exploratory evaluation. Private outputs; no model-written code execution.

One L4 container per invocation, no retries. Full run is capped at 45 minutes;
per-item artifacts are committed after every batch. Pilot results are never
promoted to confirmatory evidence. Model/tokenizer/dataset revisions are frozen.
"""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('tamper-general-evaluation')
evidence=modal.Volume.from_name('tamper-general-evidence',create_if_missing=True)
cache=modal.Volume.from_name('tamper-general-model-cache',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.8.0','transformers==4.57.1','accelerate==1.10.1',
                 'datasets==4.1.1','huggingface-hub==0.34.4','sentencepiece==0.2.1')
    .env({'HF_HOME':'/cache/huggingface','TOKENIZERS_PARALLELISM':'false'})
    .add_local_dir(ROOT,remote_path='/general')
    .add_local_dir(ROOT.parent/'capability',remote_path='/suite'))

@app.function(image=image,gpu='L40S',cpu=4,memory=16384,timeout=2700,startup_timeout=300,retries=0,
    max_containers=1,volumes={'/evidence':evidence,'/cache':cache},
    secrets=[modal.Secret.from_name('huggingface-secret')])
def evaluate(panel_id,run_id,model_name,pilot):
    import collections
    import hashlib
    import importlib.metadata
    import os
    import platform
    import random
    import sys
    import time
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,PreTrainedTokenizerFast
    sys.path.insert(0,'/suite')
    import run as old
    lock=json.loads(Path('/general/lock.json').read_text())
    panel=Path('/evidence',panel_id)
    audit=json.loads((panel/'data_audit.json').read_text())
    data=(panel/'items.jsonl').read_bytes()
    if hashlib.sha256(data).hexdigest()!=audit['items_file_sha256']:raise ValueError('panel hash mismatch')
    rows=[json.loads(l) for l in data.splitlines() if l.strip()]
    if pilot:
        chosen=[];counts=collections.Counter()
        for r in rows:
            if counts[r['task']]<6:chosen.append(r);counts[r['task']]+=1
        rows=chosen
    root=Path('/evidence',run_id);root.mkdir(exist_ok=False)
    # Write manifest before loading weights. No raw response is printed.
    repo,sub=old.MODELS[model_name]; revision=lock['models'][model_name]
    started=time.time()
    manifest={'kind':'pilot_not_confirmatory' if pilot else 'exploratory_clean_model_results',
        'model':model_name,'repo':repo,'subfolder':sub,'revision':revision,
        'panel_id':panel_id,'panel_items_sha256':audit['items_file_sha256'],
        'selected_items_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        'tasks':dict(collections.Counter(r['task'] for r in rows)),
        'source_sha256':{f:hashlib.sha256(Path('/general',f).read_bytes()).hexdigest()
                         for f in ('modal_eval.py','data_audit.py','lock.json','tokenizer_audit.py')},
        'protocol':{'seed':0,'precision':'float16','batch_by_kind':{'mcq':64,'math':32,'code':32,'instructions':16,'human_pending':16,'refusal_pending':64},'do_sample':False,
            'max_prompt_tokens':4096,'max_new_tokens':{'mcq':64,'math':1024,
                'instructions':1024,'human_pending':512,'refusal_pending':128,'code':512},
            'prompt_overlong':'exclude and report; never silently truncate',
            'refusal_labels':'pending independent human review; no inability inference from non-success'},
        'environment':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        'python':platform.python_version(),'state':'running'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));evidence.commit()
    random.seed(0);torch.manual_seed(0)
    try:
        # Same original pinned tokenizer for every checkpoint.
        tok=AutoTokenizer.from_pretrained(old.MODELS['base'][0],revision=lock['models']['base'])
        tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
        manifest['tokenizer']={'repo':old.MODELS['base'][0],'revision':lock['models']['base'],
            **{k:v for k,v in old.check_tokenizer(tok).items() if k!='sample_rendered_prompt'}}
        kw={'subfolder':sub} if sub else {}
        sys.path.insert(0,'/general')
        import tokenizer_audit
        manifest['candidate_tokenizer_check']=tokenizer_audit.compare(
            repo,revision,sub,old.MODELS['base'][0],lock['models']['base'],
            original_tokenizer=tok,prompts=[r['prompt'] for r in rows])
        model=AutoModelForCausalLM.from_pretrained(repo,revision=revision,
            torch_dtype=torch.float16,device_map='cuda',**kw).eval()
        manifest['gpu']=torch.cuda.get_device_name()
        eos={tok.eos_token_id} if tok.eos_token_id is not None else set()
        e=model.generation_config.eos_token_id
        eos.update([e] if isinstance(e,int) else e or [])
        manifest['effective_eos_ids']=sorted(eos)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if pilot:
            # Throughput-only repeated-input probe; never part of task scores.
            probe=next(r['prompt'] for r in rows if r['kind']=='math')
            rendered=tok.apply_chat_template([{'role':'user','content':probe}],tokenize=False,add_generation_prompt=True)
            profile=[]
            for batch in (8,16,32,64):
                inputs=tok([rendered]*batch,return_tensors='pt',padding=True,add_special_tokens=False,
                           return_token_type_ids=False).to(model.device)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.time()
                with torch.no_grad():
                    model.generate(**inputs,do_sample=False,min_new_tokens=64,max_new_tokens=64)
                torch.cuda.synchronize();elapsed=time.time()-t
                profile.append({'batch':batch,'output_tokens':batch*64,'seconds':round(elapsed,3),
                    'output_tokens_per_second':round(batch*64/elapsed,2),
                    'peak_allocated_gib':round(torch.cuda.max_memory_allocated()/2**30,3)})
                del inputs
            manifest['throughput_probe']=profile
            print(json.dumps({'throughput_probe':profile}),flush=True)
        manifest['performance']={}
        with (root/'items.jsonl').open('x') as stream:
            for task in dict.fromkeys(r['task'] for r in rows):
                task_rows=[r for r in rows if r['task']==task]
                batch=manifest['protocol']['batch_by_kind'][task_rows[0]['kind']]
                task_start=time.time();task_tokens=0
                torch.cuda.reset_peak_memory_stats()
                for start in range(0,len(task_rows),batch):
                    chunk=task_rows[start:start+batch]
                    texts=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],
                        tokenize=False,add_generation_prompt=True) for r in chunk]
                    enc=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False,return_token_type_ids=False)
                    lengths=enc['attention_mask'].sum(1).tolist()
                    keep=[i for i,n in enumerate(lengths) if n<=4096]
                    for i,r in enumerate(chunk):
                        if i not in keep:
                            stream.write(json.dumps({'id':r['id'],'task':task,'model':model_name,
                                'skipped':'overlong_prompt','input_tokens':lengths[i]})+'\n')
                    if not keep:continue
                    maxlen=max(lengths[i] for i in keep)
                    inputs={k:v[keep][:,-maxlen:].to(model.device) for k,v in enc.items()}
                    restricted={}
                    with torch.no_grad():
                        if chunk[0]['kind']=='mcq':
                            logits=model(**inputs,logits_to_keep=1).logits[:,-1]
                            if not torch.isfinite(logits).all():raise ValueError('non-finite logits')
                            letters=[manifest['tokenizer']['letter_token_ids'][c] for c in 'ABCD']
                            option_scores=logits[:,letters].float().cpu().tolist()
                            for j,i in enumerate(keep):
                                pred=max(range(4),key=lambda k:option_scores[j][k])
                                restricted[i]={'forced_choice_correct':int(pred==chunk[i]['answer']),
                                    'option_scores':option_scores[j],'forced_choice':'ABCD'[pred]}
                            del logits
                        output=model.generate(**inputs,do_sample=False,
                            max_new_tokens=manifest['protocol']['max_new_tokens'][chunk[0]['kind']])
                    for j,i in enumerate(keep):
                        ids=output[j,inputs['input_ids'].shape[1]:].tolist()
                        stop=next((k for k,t in enumerate(ids) if t in eos),None)
                        if stop is not None:ids=ids[:stop+1]
                        task_tokens+=len(ids)
                        record={'id':chunk[i]['id'],'task':task,'model':model_name,
                            'input_tokens':lengths[i],'output_tokens':len(ids),'eos_reached':stop is not None,
                            'response':tok.decode(ids,skip_special_tokens=True),**restricted.get(i,{})}
                        stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    stream.flush();evidence.commit()
                    print(json.dumps({'task':task,'completed':min(start+batch,len(task_rows)),
                        'total':len(task_rows),'elapsed_seconds':round(time.time()-started)}),flush=True)
                elapsed=time.time()-task_start
                manifest['performance'][task]={'seconds':round(elapsed,3),'output_tokens':task_tokens,
                    'output_tokens_per_second':round(task_tokens/elapsed,2),
                    'peak_allocated_gib':round(torch.cuda.max_memory_allocated()/2**30,3),'batch_cap':batch}
        manifest['state']='complete';(root/'COMPLETE').write_text('generation only; scoring and human review separate\n')
    except Exception as exc:
        manifest['state']='failed';manifest['error_type']=type(exc).__name__
        (root/'FAILED').write_text(type(exc).__name__+'\n')
        raise
    finally:
        manifest['elapsed_seconds']=round(time.time()-started,2)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
        index={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
               for p in root.iterdir() if p.is_file() and p.name!='artifact_index.json'}
        (root/'artifact_index.json').write_text(json.dumps(index,indent=2))
        evidence.commit();cache.commit()
    return {'run_id':run_id,'state':manifest['state'],'n_selected':len(rows),'elapsed_seconds':manifest['elapsed_seconds']}

@app.function(image=image,cpu=2,memory=4096,timeout=300,retries=0,
    volumes={'/evidence':evidence,'/cache':cache},secrets=[modal.Secret.from_name('huggingface-secret')])
def preflight(panel_id):
    import sys
    sys.path[:0]=['/general','/suite']
    import tokenizer_audit
    import run as old
    from transformers import AutoTokenizer
    lock=json.loads(Path('/general/lock.json').read_text())
    rows=[json.loads(l) for l in Path('/evidence',panel_id,'items.jsonl').read_text().splitlines()]
    tok=AutoTokenizer.from_pretrained(old.MODELS['base'][0],revision=lock['models']['base'])
    report={}
    for name in ('base','v2','v2b'):
        repo,sub=old.MODELS[name]
        report[name]=tokenizer_audit.compare(repo,lock['models'][name],sub,
            old.MODELS['base'][0],lock['models']['base'],tok,[r['prompt'] for r in rows])
    cache.commit()
    return report

@app.local_entrypoint()
def main(panel_id:str,run_id:str='',model:str='v2',pilot:bool=True,preflight_only:bool=False):
    if preflight_only:
        print(json.dumps(preflight.remote(panel_id),indent=2));return
    models=['base','v2','v2b'] if model=='all' else [model]
    for name in models:
        current_run=run_id+'-'+name if model=='all' else run_id
        call=evaluate.spawn(panel_id,current_run,name,pilot)
        try:
            result=call.get(timeout=3000)
        except TimeoutError:
            call.cancel(terminate_containers=True)
            raise RuntimeError('evaluation exceeded wall-clock allowance; cancelled')
        print(json.dumps(result,indent=2),flush=True)
