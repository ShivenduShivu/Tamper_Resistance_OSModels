"""Compare serialized tokenizers without relying on a saved Python class name."""
import json
from pathlib import Path

def compare(repo,revision,subfolder,base_repo,base_revision,original_tokenizer=None,prompts=None):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    def load(repo,revision,sub):
        prefix=(sub.strip('/')+'/') if sub else ''
        path=hf_hub_download(repo,prefix+'tokenizer.json',revision=revision)
        cfg=json.loads(Path(hf_hub_download(repo,prefix+'tokenizer_config.json',revision=revision)).read_text())
        template=cfg.get('chat_template')
        if template is None:
            template=Path(hf_hub_download(repo,prefix+'chat_template.jinja',revision=revision)).read_text()
        return Tokenizer.from_file(path),cfg,template
    original,oc,ot=load(base_repo,base_revision,None)
    candidate,cc,ct=load(repo,revision,subfolder)
    def special(c,key):
        value=c.get(key)
        return value.get('content') if isinstance(value,dict) else value
    probes=['Hello world',' नमस्ते','مرحبا بالعالم','你好','A\nB','def f(x):\n    return x > 0',
            'Réponse: 12','1,234.5','$42',special(oc,'bos_token') or '']
    oj=json.loads(original.to_str());cj=json.loads(candidate.to_str())
    result={'same_vocab':original.get_vocab()==candidate.get_vocab(),'same_chat_template':ot==ct,
            'same_bos':special(oc,'bos_token')==special(cc,'bos_token'),
            'same_eos':special(oc,'eos_token')==special(cc,'eos_token'),
            'same_probe_encoding':all(original.encode(t,add_special_tokens=False).ids==
                                     candidate.encode(t,add_special_tokens=False).ids for t in probes)}
    for component in ('model','normalizer','pre_tokenizer','post_processor','decoder'):
        result['same_'+component]=oj.get(component)==cj.get(component)
    if original_tokenizer is not None and prompts is not None:
        if isinstance(ct,list):ct={v['name']:v['template'] for v in ct}
        result['same_all_evaluation_renderings']=all(
            original_tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=True,
                add_generation_prompt=True,chat_template=ct)==
            original_tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=True,
                add_generation_prompt=True) for p in prompts)
    required={k:v for k,v in result.items() if k!='same_chat_template'}
    if not result['same_chat_template'] and not result.get('same_all_evaluation_renderings'):
        raise ValueError('saved chat template differs for evaluated inputs: '+json.dumps(result))
    if not all(required.values()):raise ValueError('serialized tokenizer difference: '+json.dumps(result))
    return result
