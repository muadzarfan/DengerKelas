"""Local complete resume state, distinct from loading model weights alone."""
import random
import hashlib
import numpy as np
import torch
from src.data.common import ROOT, write_json, read_json, digest


def identity():
    paths = ('configs/data_sources.json','configs/data_preparation.json',
             'data/manifests/fleurs_train.jsonl','data/manifests/fleurs_validation.jsonl',
             'data/manifests/demand_noise.jsonl','data/manifests/validation_noisy_recipes.jsonl',
             'data/manifests/step4_model.json')
    # Model identity uses immutable weight SHA, not the download's reused flag.
    result = {p:digest(ROOT/p) for p in paths if p != 'data/manifests/step4_model.json'}
    model = read_json(ROOT/'data/manifests/step4_model.json')
    result['pretrained_weight_sha256'] = model['local_sha256']
    result['dataset_revision'] = read_json(ROOT/'configs/data_sources.json')['fleurs']['revision']
    return result


def signature(cfg):
    return {k:cfg[k] for k in ('seed','physical_batch_size','gradient_accumulation','learning_rate',
                              'weight_decay','scheduler','warmup_ratio','max_grad_norm','augmentation',
                              'resolved_precision','gradient_checkpointing')}


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save(folder, model, processor, optimizer, scheduler, scaler, progress, cfg, data_identity, environment):
    if folder.exists():
        raise FileExistsError(f'Checkpoint already exists: {folder}')
    temporary = folder.with_name(folder.name+'.incomplete')
    temporary.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(temporary,safe_serialization=True)
    processor.save_pretrained(temporary)
    state = {'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
             'progress':progress.copy(),'rng':capture_rng(),'data_identity':data_identity,
             'config_signature':signature(cfg),'scheduler_total_steps':cfg['scheduler_total_steps']}
    torch.save(state,temporary/'training_state.pt')
    write_json(temporary/'run_config.json',cfg)
    write_json(temporary/'data_identity.json',data_identity)
    write_json(temporary/'environment.json',environment)
    write_json(temporary/'progress.json',progress)
    write_json(temporary/'artifact_checksums.json',{
        p.name:digest(p) for p in temporary.iterdir() if p.is_file()})
    temporary.replace(folder)
    return folder


def restore(folder, optimizer, scheduler, scaler, cfg, data_identity):
    checks = read_json(folder/'artifact_checksums.json')
    if any(digest(folder/name) != checksum for name,checksum in checks.items()):
        raise ValueError('Checkpoint integrity mismatch')
    # This is our own local state file, including Python/NumPy RNG objects.
    # Never use this API with an untrusted downloaded training_state.pt.
    state = torch.load(folder/'training_state.pt',map_location='cpu',weights_only=False)
    if state['data_identity'] != data_identity or state['config_signature'] != signature(cfg):
        raise ValueError('Resume data/config differs from saved training state')
    if state['scheduler_total_steps'] != cfg['scheduler_total_steps']:
        raise ValueError('Resume must retain the original scheduler horizon')
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler']); restore_rng(state['rng'])
    return state['progress']


def state_fingerprint(model, optimizer, scheduler, progress):
    """Stable tensor/state digest to compare uninterrupted vs complete resume."""
    def hash_value(value):
        h = hashlib.sha256()
        def update(x):
            if isinstance(x,torch.Tensor):
                x=x.detach().cpu().contiguous()
                h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
                h.update(x.numpy().tobytes())
            elif isinstance(x,dict):
                for key in sorted(x,key=str):
                    h.update(repr(key).encode()); update(x[key])
            elif isinstance(x,(list,tuple)):
                for item in x: update(item)
            else: h.update(repr(x).encode())
        update(value)
        return h.hexdigest()
    return {'model':hash_value(model.state_dict()),'optimizer':hash_value(optimizer.state_dict()),
            'scheduler':hash_value(scheduler.state_dict()),'progress':progress.copy(),
            'torch_rng':hash_value(torch.get_rng_state()),
            'numpy_rng':hash_value(np.random.get_state()[1].tolist()),
            'python_rng':hash_value(random.getstate())}
