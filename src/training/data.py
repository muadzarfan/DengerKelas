"""Manifest-backed on-the-fly features; final test is inaccessible in this module."""
from collections import OrderedDict
import numpy as np
import soundfile as sf
import torch
from src.data.common import ROOT, read_json, read_jsonl, digest
from src.data.preprocessing import read_audio
from src.data.augmentation import recipe, materialize


def records(split):
    if split not in ('train', 'validation'):
        raise ValueError('Step 4 accepts train/validation only; final test is locked')
    path = ROOT / f'data/manifests/fleurs_{split}.jsonl'
    rows = [r for r in read_jsonl(path) if r['eligible']]
    if not rows or len({r['recording_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate manifest')
    return rows


def short_subset(rows, count):
    chosen, seen = [], set()
    for r in sorted(rows, key=lambda x:(x['duration_seconds'], x['recording_id'])):
        if r['source_sentence_id'] not in seen:
            chosen.append(r); seen.add(r['source_sentence_id'])
            if len(chosen) == count:
                return chosen
    raise ValueError('Insufficient distinct-content examples for fixed subset')


def fixed_metric_subset(rows, count, seed):
    # Selection is independent of recognition difficulty/model outputs.
    import hashlib
    return sorted(rows, key=lambda r:hashlib.sha256(f"{seed}|{r['recording_id']}".encode()).hexdigest())[:count]


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, rows, processor, mode='clean', condition='clean', cache_examples=0):
        if any(r['split'] not in ('train','validation') or not r['eligible'] for r in rows):
            raise ValueError('Excluded or final test record rejected')
        if mode == 'E2' and any(r['split'] != 'train' for r in rows):
            raise ValueError('Random E2 augmentation is train-only')
        if mode not in ('clean','E2','fixed_noise'):
            raise ValueError('Unknown dataset mode')
        if mode == 'fixed_noise' and any(r['split'] != 'validation' for r in rows):
            raise ValueError('Only fixed validation noise is allowed in Step 4')
        self.rows, self.proc, self.mode, self.condition = rows, processor, mode, condition
        self.epoch = 0
        self.capacity = int(cache_examples)
        self.cache = OrderedDict()
        self.cfg = read_json(ROOT/'configs/data_preparation.json')['augmentation']
        self.noise = {r['split']:r for r in read_jsonl(ROOT/'data/manifests/demand_noise.jsonl') if r['split'] != 'test'}
        # Only requested manifests and relevant train/validation audio are opened.
        for r in rows:
            path = ROOT/r['audio_path']
            info = sf.info(path)
            if (info.samplerate != 16000 or info.channels != 1 or info.frames != r['prepared_num_samples']
                or digest(path) != r['audio_sha256']):
                raise ValueError(f"Audio/manifest mismatch: {r['recording_id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.mode == 'clean' and index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        r = self.rows[index]
        if self.mode == 'clean':
            x, sr = read_audio(ROOT/r['audio_path'])
            plan = None
        else:
            snr = float(self.condition.removesuffix('dB')) if self.mode == 'fixed_noise' else None
            condition = f'controlled_noise_{snr:g}dB' if snr is not None else 'E2_training'
            plan = recipe(r, self.noise[r['split']], self.cfg, condition, snr=snr, epoch=0 if snr is not None else self.epoch)
            x, mix_info = materialize(r, plan)
            sr = 16000
        if sr != 16000 or x.ndim != 1 or len(x) != r['prepared_num_samples'] or len(x) > 480000 or not np.isfinite(x).all():
            raise ValueError('Invalid waveform; refusing implicit truncation')
        processed = self.proc.feature_extractor(x, sampling_rate=sr, return_attention_mask=True, return_tensors='np')
        labels = self.proc.tokenizer(r['training_target'], truncation=False).input_ids
        if len(labels) > 448 or len(labels) != r['target_token_count']:
            raise ValueError('Label limit/count mismatch')
        item = {'input_features':processed.input_features[0], 'attention_mask':processed.attention_mask[0],
                'labels':labels,'recording_id':r['recording_id'],'target':r['training_target'],
                'duration_seconds':len(x)/sr,'recipe':plan}
        if self.mode == 'clean' and self.capacity:
            self.cache[index] = item
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return item


class SpeechCollator:
    def __init__(self, processor, decoder_start_token_id):
        self.proc, self.start = processor, decoder_start_token_id

    def __call__(self, items):
        features = self.proc.feature_extractor.pad(
            [{'input_features':x['input_features']} for x in items], return_tensors='pt')
        labels = self.proc.tokenizer.pad([{'input_ids':x['labels']} for x in items], return_tensors='pt')
        target = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        if torch.all(target[:,0].eq(self.start)):
            target = target[:,1:]
        else:
            raise ValueError('Expected exactly one decoder start prefix per example')
        if torch.any(target.ne(-100).sum(dim=1) == 0) or target.shape[1] > 448:
            raise ValueError('Empty/oversized labels')
        return {'input_features':features.input_features,
                'attention_mask':torch.from_numpy(np.stack([x['attention_mask'] for x in items])).long(),
                'labels':target}
