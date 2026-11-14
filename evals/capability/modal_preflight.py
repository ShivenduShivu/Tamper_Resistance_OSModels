"""Public-data CPU preflight only. No secrets, model loading or capability claims.

Run: modal run evals/capability/modal_preflight.py --out /local/path/result.json
One CPU function, 300 second execution timeout, no retries, no GPU.
"""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App('tamper-capability-public-preflight')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('datasets==4.1.1', 'huggingface-hub==0.34.4')
         .add_local_dir(ROOT, remote_path='/suite'))

@app.function(image=image, cpu=1, memory=2048, timeout=300, retries=0,
              max_containers=1)
def check():
    import hashlib
    import importlib.metadata
    import sys
    import time
    sys.path.insert(0, '/suite')
    import tasks
    from huggingface_hub import HfApi
    started = time.time()
    lock = json.loads(Path('/suite/dataset_lock.json').read_text())
    result = {'kind': 'public_data_preflight_not_model_results', 'checks': {},
              'versions': {p: importlib.metadata.version(p)
                           for p in ('datasets', 'huggingface-hub')}}
    for task, fn, repo in (
        ('gsm8k', tasks.gsm8k_load, 'openai/gsm8k'),
        ('cybermetric', tasks.cybermetric_load, 'tihanyin/CyberMetric'),
        ('humaneval', tasks.humaneval_load, 'openai/openai_humaneval')):
        try:
            rev = lock['datasets'][repo]
            kw = {'revision': rev}
            if task != 'humaneval': kw.update(n=4, seed=0)
            rows = fn(**kw)
            result['checks'][task] = {'state': 'pass', 'n': len(rows),
                'revision': rev, 'selected_rows_sha256': hashlib.sha256(
                    json.dumps(rows, sort_keys=True).encode()).hexdigest()}
        except Exception as exc:
            result['checks'][task] = {'state': 'fail', 'error': type(exc).__name__ + ': ' + str(exc)[:500]}
    # Only inspect access; never load weights without the required credentials.
    for model in ('CohereLabs/tiny-aya-global', 'StarWarrior24x7/tiny-aya-global-xref-v2'):
        try:
            info = HfApi().model_info(model)
            result['checks'][model] = {'metadata_visible': True, 'gated': info.gated,
                                      'note': 'metadata visibility is not weight-download authorization'}
        except Exception as exc:
            result['checks'][model] = {'metadata_visible': False, 'error_type': type(exc).__name__}
    result['elapsed_seconds'] = round(time.time()-started, 2)
    return result

@app.local_entrypoint()
def main(out: str):
    result = check.remote()
    with Path(out).open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
