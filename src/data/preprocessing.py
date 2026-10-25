"""Minimal text transforms and reusable single-window Whisper preparation."""
import csv
import unicodedata
import re
from pathlib import PurePosixPath
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from .common import ROOT, read_json


def training_text(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def suspect_mojibake(text):
    return bool(re.search('[\u00c2\u00c3][\u0080-\u00bf]',text))


def scoring_text(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())


def metadata_rows(path):
    # FLEURS TSV is not RFC-style quoted CSV. QUOTE_NONE preserves literal
    # quotation marks and avoids joining/truncating rows containing quotes.
    with open(path, encoding='utf-8', newline='') as handle:
        for line, row in enumerate(csv.reader(handle, delimiter='\t', quoting=csv.QUOTE_NONE), 1):
            if len(row) != 7:
                raise ValueError(f'Unexpected TSV schema at {path}:{line}: {len(row)} columns')
            source_id, filename, raw, normalized, graphemes, samples, gender = row
            if PurePosixPath(filename).name != filename or not filename.endswith('.wav'):
                raise ValueError('Unsafe or unexpected source audio filename')
            yield {'source_sentence_id': source_id, 'source_filename': filename,
                   'raw_transcription': raw, 'source_normalized_transcription': normalized,
                   'source_graphemes': graphemes, 'metadata_num_samples': int(samples),
                   'source_gender': gender, 'source_metadata_line': line}


def read_audio(path, target_rate=16000):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Empty or nonfinite audio')
    x = x.mean(axis=1, dtype=np.float32)
    if sr != target_rate:
        factor = gcd(int(sr), int(target_rate))
        x = resample_poly(x, target_rate // factor, sr // factor).astype(np.float32)
    return np.ascontiguousarray(x), target_rate


def processor():
    from transformers import WhisperProcessor
    p = WhisperProcessor.from_pretrained(str(ROOT/'data/checkpoints/whisper-tiny-processor'), local_files_only=True,
                                         language='Indonesian', task='transcribe')
    p.tokenizer.set_prefix_tokens(language='Indonesian', task='transcribe', predict_timestamps=False)
    return p


def prepare_example(record, proc, waveform=None, allow_final_test=False):
    if record['split']=='test' and not allow_final_test:
        raise ValueError('Final test feature access requires an explicit evaluation request')
    if not record['eligible']:
        raise ValueError('Excluded record cannot enter the compatible single-window subset')
    x, sr = read_audio(ROOT/record['audio_path']) if waveform is None else (np.asarray(waveform,np.float32),16000)
    cfg = read_json(ROOT/'configs/data_preparation.json')
    if len(x) > cfg['max_audio_seconds'] * sr:
        raise ValueError('Refusing waveform truncation with a full transcript')
    features = proc.feature_extractor(x, sampling_rate=sr, return_tensors='np').input_features[0]
    labels = proc.tokenizer(record['training_target']).input_ids
    if len(labels) > cfg['max_target_tokens']:
        raise ValueError('Target exceeds configured decoder limit')
    return {'input_features': features, 'labels': labels}


def padded_labels_for_smoke(examples, proc, decoder_start_token_id):
    padded = proc.tokenizer.pad([{'input_ids':x['labels']} for x in examples], return_tensors='np')
    labels = np.where(padded['attention_mask'] == 1, padded['input_ids'], -100)
    if np.all(labels[:,0] == decoder_start_token_id):
        labels = labels[:,1:]
    return labels
