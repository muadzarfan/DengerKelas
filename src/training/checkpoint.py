"""Local complete resume state, distinct from loading model weights alone."""
import random
import hashlib
import copy
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
    keys=('seed','physical_batch_size','gradient_accumulation','learning_rate',
                              'weight_decay','scheduler','warmup_ratio','max_grad_norm','augmentation',
          'resolved_precision','gradient_checkpointing','resolved_device','cpu_threads','kind','epochs',
          'train_recording_ids','validation_recording_ids','generation_max_new_tokens',
          'validation_conditions','selection_metric','train_metric_examples','eval_every_steps')
    return copy.deepcopy({k:cfg[k] for k in keys})


def validate_resume_config(previous,current):
    if signature(previous)!=signature(current):
        changed=[k for k in signature(previous) if signature(previous)[k]!=signature(current)[k]]
        raise ValueError(f'Resume data/order/evaluation/training config differs: {changed}')


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
             'config_signature':signature(cfg),'resume_schema':2,'scheduler_total_steps':cfg['scheduler_total_steps']}
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
    from .integrity import verify_checkpoint
    verify_checkpoint(folder)
    # This is our own local state file, including Python/NumPy RNG objects.
    # Never use this API with an untrusted downloaded training_state.pt.
    state = torch.load(folder/'training_state.pt',map_location='cpu',weights_only=False)
    if state['data_identity'] != data_identity:
        raise ValueError('Resume data/config differs from saved training state')
    previous=read_json(folder/'run_config.json')
    validate_resume_config(previous,cfg)
    # Older local checkpoints lack the expanded signature, but their checksummed
    # run_config contains the ordered IDs and evaluation settings for validation.
    expected=signature(previous)
    if any(k not in expected or expected[k]!=v for k,v in state['config_signature'].items()):
        raise ValueError('Saved resume signature disagrees with checkpoint run_config')
    if state['scheduler_total_steps'] != cfg['scheduler_total_steps']:
        raise ValueError('Resume must retain the original scheduler horizon')
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler']); restore_rng(state['rng'])
    return state['progress']


def state_fingerprint(model, optimizer, scheduler, progress, schema_version=2):
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
    result={'model':hash_value(model.state_dict()),'optimizer':hash_value(optimizer.state_dict()),
            'scheduler':hash_value(scheduler.state_dict()),'progress':progress.copy(),
            'torch_rng':hash_value(torch.get_rng_state()),
            'numpy_rng':hash_value(np.random.get_state()[1].tolist()),
            'python_rng':hash_value(random.getstate())}
    if schema_version==2:
        rng=np.random.get_state()
        result.update(schema_version=2,numpy_rng=hash_value([rng[0],rng[1].tolist(),*rng[2:]]),
                      cuda_rng=hash_value(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    elif schema_version!=1:
        raise ValueError('Unknown fingerprint schema')
    return result
