"""Explicit configurations and deterministic local runtime setup."""
import math
import os
import platform
import random
import sys
import importlib.metadata
from pathlib import Path
import numpy as np
import psutil
import torch
from src.data.common import ROOT, read_json


def load(path):
    cfg = read_json(path)
    if 'extends' in cfg:
        base = load(Path(path).parent / cfg.pop('extends'))
        base.update(cfg)
        cfg = base
    if cfg['augmentation'] not in ('clean', 'E2') or cfg['kind'] not in ('diagnostic', 'experiment'):
        raise ValueError('Invalid run kind or augmentation')
    if cfg['physical_batch_size'] < 1 or cfg['gradient_accumulation'] < 1:
        raise ValueError('Invalid batch configuration')
    if cfg['kind'] == 'experiment' and cfg['physical_batch_size'] * cfg['gradient_accumulation'] != 16:
        raise ValueError('Official single-device effective batch must stay 16')
    if cfg['save_total_limit'] < 2:
        raise ValueError('Retention needs room for the best and latest checkpoints')
    return cfg


def setup(cfg):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.use_deterministic_algorithms(True, warn_only=False)
    requested = cfg['device']
    device = 'cuda' if requested == 'auto' and torch.cuda.is_available() else 'cpu' if requested == 'auto' else requested
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    precision = cfg['precision']
    if precision == 'auto':
        precision = 'bf16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'fp16' if device == 'cuda' else 'fp32'
    if precision != 'fp32' and device != 'cuda':
        raise ValueError('Mixed precision is enabled only on verified CUDA hardware')
    cfg = {**cfg, 'resolved_device':device, 'resolved_precision':precision,
           'effective_batch_size':cfg['physical_batch_size']*cfg['gradient_accumulation']}
    return cfg


def environment():
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpu.append({'name':p.name,'vram_bytes':p.total_memory})
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    return {'python':sys.version,'os':platform.platform(),'cpu':cpu,
            'physical_cores':psutil.cpu_count(logical=False),'logical_cores':psutil.cpu_count(),
            'ram_bytes':psutil.virtual_memory().total,'available_ram_bytes':psutil.virtual_memory().available,
            'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','scipy','soundfile','safetensors')},
            'torch_cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpus':gpu,
            'cpu_threads':torch.get_num_threads(),'cpu_precision_used':'fp32',
            'cuda_bf16_supported':bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())}


def total_steps(cfg, size):
    return cfg['max_steps'] or math.ceil(math.ceil(size/cfg['physical_batch_size'])/cfg['gradient_accumulation'])*cfg['epochs']
