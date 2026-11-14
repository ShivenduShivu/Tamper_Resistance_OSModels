"""CPU score of completed generation artifacts. Never executes generated code."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('tamper-general-score')
volume=modal.Volume.from_name('tamper-general-evidence')
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.8.0','transformers==4.57.1','huggingface-hub==0.34.4',
                    'absl-py==2.3.1','langdetect==1.0.9','nltk==3.9.2','immutabledict==4.2.1')
       .run_commands("python -c \"import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)\"")
       .add_local_dir(ROOT,remote_path='/general')
       .add_local_dir(ROOT.parent/'capability',remote_path='/suite'))
@app.function(image=image,cpu=4,memory=8192,timeout=1200,retries=0,max_containers=1,
              volumes={'/evidence':volume},secrets=[modal.Secret.from_name('huggingface-secret')])
def score(panel_id,run_id):
    import hashlib
    import importlib.metadata
    import random
    import sys
    from huggingface_hub import HfApi
    from transformers import pipeline
    from langdetect import DetectorFactory
    random.seed(0);DetectorFactory.seed=0
    sys.path[:0]=['/general','/general/reference','/suite']
    import score as sc
    verifier=json.loads(Path('/general/reference/instruction_following_eval/provenance.json').read_text())
    for name,digest in verifier['sha256'].items():
        if hashlib.sha256(Path('/general/reference/instruction_following_eval',name).read_bytes()).hexdigest()!=digest:
            raise ValueError('official verifier hash mismatch')
    math_parser=json.loads(Path('/general/reference/mgsm_provenance.json').read_text())
    if hashlib.sha256(Path('/general/reference/mgsm_parser.py').read_bytes()).hexdigest()!=math_parser['module_sha256']:
        raise ValueError('secondary maths parser hash mismatch')
    root=Path('/evidence',run_id)
    out=root/'scoring-v1';out.mkdir(exist_ok=False)
    repo='ProtectAI/distilroberta-base-rejection-v1'
    revision=HfApi().model_info(repo).sha
    classifier=pipeline('text-classification',model=repo,revision=revision,device=-1)
    records,summary=sc.score_run('/evidence/'+panel_id,root,classifier)
    summary['scoring_provenance']={'classifier_repo':repo,'classifier_revision':revision,
        'ifeval_verifier':verifier,'secondary_math_parser':math_parser,'score_source_sha256':hashlib.sha256(Path('/general/score.py').read_bytes()).hexdigest(),
        'versions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        'nltk_data_sha256':{str(p.relative_to('/root/nltk_data')):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path('/root/nltk_data').rglob('*') if p.is_file()}}
    (out/'scored_items.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    index={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in out.iterdir()}
    (out/'artifact_index.json').write_text(json.dumps(index,indent=2));volume.commit()
    return {k:v for k,v in summary.items() if k!='scoring_provenance'}
@app.local_entrypoint()
def main(panel_id:str,run_id:str,out:str):
    result=score.remote(panel_id,run_id)
    with Path(out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
