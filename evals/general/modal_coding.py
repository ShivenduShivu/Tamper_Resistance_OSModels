"""Live pinned HumanEval backend verification, then isolated candidate scoring.

This delegates execution to the existing fresh, network-blocked, secret-free
SandboxBackend. Worker CPU never executes model code. No paid retries.
"""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('tamper-general-coding')
volume=modal.Volume.from_name('tamper-general-evidence')
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('modal==1.4.2','datasets==4.1.1','huggingface-hub==0.34.4')
    .add_local_dir(ROOT,remote_path='/general')
    .add_local_dir(ROOT.parent/'capability',remote_path='/suite'))
@app.function(image=image,cpu=2,memory=2048,timeout=1800,retries=0,max_containers=1,nonpreemptible=True,
              volumes={'/evidence':volume})
def run(panel_id,run_id,preflight):
    import concurrent.futures as cf
    import hashlib
    import sys
    import time
    sys.path[:0]=['/general','/suite']
    import coding_exec as cx
    import coding_score as cs
    import score as sc
    import code_adapter
    backend=cx.SandboxBackend(app_name='tamper-general-isolated-code')
    panel=Path('/evidence',panel_id)
    rows=[json.loads(l) for l in (panel/'items.jsonl').read_text().splitlines() if l.strip()]
    problems={r['id']:r for r in rows if r['task']=='humaneval'}
    if len(problems)!=164:raise ValueError('standard coding set must contain all 164 problems')
    def prob(r):
        return {'task_id':r['id'],'prompt':r['overlap_text'],'test':r['test'],
                'entry_point':r['entry_point'],'canonical_solution':r['canonical_solution']}
    if preflight:
        root=Path('/evidence',run_id);root.mkdir(exist_ok=False)
        failures=cs.run_fixtures(backend,full_canonical=False)
        if failures:
            (root/'FAILED').write_text(json.dumps(failures));volume.commit()
            raise ValueError('live authored-fixture mismatch')
        # Explicit positive test of network denial and absence of injected secrets.
        network_prob=dict(cs.FIXTURE_PROBLEM)
        network_body=("    import os, socket\n    assert 'HF_TOKEN' not in os.environ\n"
            "    s = socket.socket(); s.settimeout(2)\n    connected = False\n"
            "    try:\n        s.connect(('1.1.1.1', 443)); connected = True\n"
            "    except OSError:\n        pass\n    finally:\n        s.close()\n"
            "    assert not connected\n    return a + b\n")
        network_result=cx.score_completion(network_prob,network_body,backend)
        if network_result.get('correct')!=1:raise ValueError('network/secrets probe failed')
        records=[]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(cx.score_completion,prob(r),r['canonical_solution'],backend):r['id']
                     for r in problems.values()}
            for f in cf.as_completed(futures):
                record=f.result();records.append(record)
                with (root/'canonical.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
                volume.commit()
        passed=sum(r.get('correct',0) for r in records)
        result={'kind':'live_backend_validation_not_model_scores','canonical_passed':passed,
            'canonical_n':len(records),'authored_fixture_failures':failures,
            'network_and_secret_probe':network_result,'sdk_version':modal.__version__,
            'image':cx.SANDBOX_IMAGE,'module_sha256':cx.REFERENCE_SHA256,
            'state':'pass' if passed==164 else 'fail'}
        (root/'summary.json').write_text(json.dumps(result,indent=2))
        if passed==164:(root/'COMPLETE').write_text('backend verified\n')
        volume.commit();return result
    root=Path('/evidence',run_id)
    generation_manifest=sc.verify_directory(root)
    panel_bytes=(panel/'items.jsonl').read_bytes()
    if hashlib.sha256(panel_bytes).hexdigest()!=generation_manifest['panel_items_sha256']:
        raise ValueError('coding panel does not match generation')
    gate=json.loads(Path('/evidence/coding-backend-20261113-v3/summary.json').read_text())
    if gate['state']!='pass':raise ValueError('live coding gate not passed')
    generated=[json.loads(l) for l in (root/'items.jsonl').read_text().splitlines() if l.strip()]
    if len({(g['task'],g['id']) for g in generated})!=len(generated):
        raise ValueError('duplicate generation rows')
    source={(r['task'],r['id']):r for r in rows}
    selected=[source[(g['task'],g['id'])] for g in generated]
    if hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()!=generation_manifest['selected_items_sha256']:
        raise ValueError('selected coding content is not bound to generation')
    answers=[r for r in generated if r['task']=='humaneval']
    if len(answers)!=generation_manifest['tasks']['humaneval']:raise ValueError('coding coverage mismatch')
    out=root/'coding-adapted-v2';out.mkdir(exist_ok=False)
    ctx={'expected_python':'3.13','model':generation_manifest['model'],
        'model_revision':generation_manifest['revision'],
        'generation_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'items_manifest_sha256':hashlib.sha256(panel_bytes).hexdigest(),
        'responses_sha256':hashlib.sha256((root/'items.jsonl').read_bytes()).hexdigest(),
        'dataset_revision':json.loads(Path('/general/lock.json').read_text())['datasets']['openai/openai_humaneval'],
        'protocol':cx.CODING_PROTOCOL_VERSION,'driver_sha256':cx.driver_sha256(),
        'measurement_config_sha256':cs.measurement_config_sha256(),
        'coding_exec_sha256':hashlib.sha256(Path(cx.__file__).read_bytes()).hexdigest(),
        'scorer_sha256':hashlib.sha256(Path('/general/modal_coding.py').read_bytes()).hexdigest(),
        'sandbox_image':cx.SANDBOX_IMAGE,'modal_sdk_required':cx.REQUIRED_MODAL_VERSION}
    records=[]
    def execute(g):
        if g['id'] not in problems:raise ValueError('unknown problem')
        p=prob(problems[g['id']])
        # Persist creation obligation before submit, and exact sandbox ID on creation.
        def on_create(sid):
            with (out/('ledger-'+g['id'].replace('/','_')+'.jsonl')).open('a') as ledger:
                ledger.write(json.dumps({'state':'created','sandbox_id':sid})+'\n')
            volume.commit()
        with (out/('ledger-'+g['id'].replace('/','_')+'.jsonl')).open('x') as ledger:
            ledger.write(json.dumps({'state':'pending','id':g['id']})+'\n')
        volume.commit()
        if 'skipped' in g:return dict(id=g['id'],status='skipped')
        completion,adapter_route=code_adapter.function_body(p,g['response'])
        rec=cx.score_completion(p,completion,backend,on_create=on_create)
        rec['adapter_route']=adapter_route
        rec['adapter_sha256']=hashlib.sha256(Path('/general/code_adapter.py').read_bytes()).hexdigest()
        payload,finish=cx.build_item_payload(p,completion)
        expected=hashlib.sha256(payload.encode()).hexdigest() if payload is not None else None
        rec.update(ctx)
        invalid=cs.validate_scored_row(rec,set(problems),ctx,
            expected_input_sha=expected,expected_over_cap=payload is None)
        if invalid:raise ValueError('invalid verdict evidence: '+invalid)
        return rec
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(execute,g) for g in answers]
        for f in cf.as_completed(futures):
            rec=f.result();records.append(rec)
            with (out/'items.jsonl').open('a') as stream:stream.write(json.dumps(rec)+'\n')
            volume.commit()
    infra=sum(r['status']=='infra_error' for r in records)
    result={'model':json.loads((root/'manifest.json').read_text())['model'],
        'n':len(records),'passed':sum(r.get('correct',0) for r in records),'infra_errors':infra,
        'state':'complete_valid' if infra==0 and len(records)==len(answers) else 'invalid',
        'protocol':cx.CODING_PROTOCOL_VERSION+'+declared-single-function-adapter-v1','backend_gate':gate,
        'generation_items_sha256':hashlib.sha256((root/'items.jsonl').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    index={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
           for p in out.iterdir() if p.is_file()}
    (out/'artifact_index.json').write_text(json.dumps(index,indent=2));volume.commit();return result
@app.function(image=image,cpu=1,memory=1024,timeout=120,retries=0)
def diagnose():
    import sys
    sys.path.insert(0,'/suite')
    import coding_exec as cx, coding_score as cs
    return cx.score_completion(cs.FIXTURE_PROBLEM,'    return a + b\n',cx.SandboxBackend('tamper-general-isolated-code'))

@app.local_entrypoint()
def main(panel_id:str='',run_id:str='',preflight:bool=False,out:str='',diagnostic:bool=False):
    result=diagnose.remote() if diagnostic else run.remote(panel_id,run_id,preflight)
    if out:
        with Path(out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
