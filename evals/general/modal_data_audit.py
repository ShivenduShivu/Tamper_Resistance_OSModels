"""CPU data audit, durable private manifests, downloadable aggregate evidence."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('tamper-general-data-audit')
volume=modal.Volume.from_name('tamper-general-evidence',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('datasets==4.1.1','huggingface-hub==0.34.4')
       .add_local_dir(ROOT,remote_path='/general')
       .add_local_dir(ROOT.parent/'capability',remote_path='/suite'))
@app.function(image=image,cpu=2,memory=4096,timeout=600,retries=0,max_containers=1,
              volumes={'/evidence':volume},secrets=[modal.Secret.from_name('huggingface-secret')])
def run(panel_id):
    import hashlib
    import importlib.metadata
    import sys
    sys.path.insert(0,'/general')
    import data_audit
    report=data_audit.freeze('/evidence/'+panel_id,'/general/lock.json')
    report['source_sha256']={f:hashlib.sha256(Path('/general',f).read_bytes()).hexdigest()
                             for f in ('data_audit.py','lock.json')}
    report['environment']={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    Path('/evidence',panel_id,'data_audit.json').write_text(json.dumps(report,indent=2))
    volume.commit()
    return report
@app.local_entrypoint()
def main(panel_id:str,out:str):
    report=run.remote(panel_id)
    with Path(out).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('environment','lock')},indent=2))
