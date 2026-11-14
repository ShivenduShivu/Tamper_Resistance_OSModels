"""Bounded CPU audit of pinned file access and source-data availability."""
import json
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent
app = modal.App('tamper-evaluation-access-audit')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('huggingface-hub==0.34.4')
         .add_local_file(ROOT/'dataset_lock.json','/lock.json'))

@app.function(image=image, cpu=1, memory=1024, timeout=180, retries=0, max_containers=1,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def audit():
    import hashlib
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata
    lock = json.loads(Path('/lock.json').read_text())
    api = HfApi()
    result = {'kind':'access_audit_not_capability_results','models':{},'datasets':{}}
    for name, repo, sub in [('base','CohereLabs/tiny-aya-global',''),
                           ('v2','StarWarrior24x7/tiny-aya-global-xref-v2',''),
                           ('v2b','StarWarrior24x7/tiny-aya-global-xref-v2','v2b/')]:
        cell={'revision':lock['models'][name], 'repo':repo,'subfolder':sub or None,'files':{}}
        for filename in ('config.json','tokenizer_config.json','tokenizer.json','model.safetensors.index.json'):
            try:
                path=hf_hub_download(repo,sub+filename,revision=cell['revision'])
                cell['files'][filename]={'accessible':True,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                if filename=='model.safetensors.index.json':
                    weights=sorted(set(json.loads(Path(path).read_text())['weight_map'].values()))
                    for weight in weights:
                        meta=get_hf_file_metadata(hf_hub_url(repo,sub+weight,revision=cell['revision']))
                        cell['files'][weight]={'head_accessible':True,'size':meta.size,'etag':meta.etag}
            except Exception as e:
                response=getattr(e,'response',None)
                cell['files'][filename]={'accessible':False,'error_type':type(e).__name__,
                                         'status_code':getattr(response,'status_code',None)}
        result['models'][name]=cell
    for repo in ('StarWarrior24x7/tiny-aya-extended-refusal','CohereLabs/aya_evaluation_suite',
                 'CohereLabs/global-mgsm','CohereLabs/Global-MMLU','google/IFEval',
                 'natolambert/xstest-v2-copy','bench-llm/or-bench'):
        try:
            info=api.dataset_info(repo)
            result['datasets'][repo]={'revision':info.sha,'files':[s.rfilename for s in info.siblings]}
        except Exception as e:
            result['datasets'][repo]={'error_type':type(e).__name__}
    return result

@app.local_entrypoint()
def main(out:str):
    result=audit.remote()
    with Path(out).open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
