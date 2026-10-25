"""Reproducible whole-utterance RMS-SNR mixing; never overwrite clean audio."""
import hashlib
import numpy as np
from .common import ROOT, read_json


def stable_seed(recording_id, condition, epoch=0, seed=20261025):
    data = f'{seed}|{recording_id}|{condition}|{epoch}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big')


def noise_segment(noise, offset, length):
    noise = np.asarray(noise, np.float32)
    if noise.ndim != 1 or not len(noise) or length < 1:
        raise ValueError('Noise and speech must be nonempty mono arrays')
    return noise[(np.arange(length, dtype=np.int64) + int(offset)) % len(noise)].copy()


def mix_at_snr(speech, noise, snr_db, peak_limit=.99):
    speech = np.asarray(speech, np.float32)
    noise = np.asarray(noise, np.float32)
    if speech.ndim != 1 or speech.shape != noise.shape or not len(speech):
        raise ValueError('Speech/noise need the same nonempty mono shape')
    if not np.isfinite(speech).all() or not np.isfinite(noise).all():
        raise ValueError('Nonfinite component')
    ps = float(np.mean(speech.astype(np.float64)**2))
    pn = float(np.mean(noise.astype(np.float64)**2))
    if pn <= 0:
        raise ValueError('Zero-energy noise cannot realize the requested SNR')
    if ps <= 0:
        return speech.copy(), speech.copy(), np.zeros_like(noise), {'applied':False, 'reason':'zero_speech_energy','measured_snr_db':None}
    alpha = np.sqrt(ps/(pn*10**(float(snr_db)/10)))
    clean_component = speech.astype(np.float64)
    noise_component = noise.astype(np.float64) * alpha
    peak = float(np.max(np.abs(clean_component + noise_component)))
    gain = min(1.0, peak_limit/peak) if peak else 1.0
    clean_component *= gain
    noise_component *= gain
    mixture = (clean_component + noise_component).astype(np.float32)
    snr = 10*np.log10(np.mean(clean_component**2)/np.mean(noise_component**2))
    return mixture, clean_component, noise_component, {'applied':True,'requested_snr_db':float(snr_db),
        'measured_snr_db':float(snr),'noise_gain':float(alpha),'common_gain':float(gain),'peak':float(np.max(np.abs(mixture)))}


def recipe(record, noise_record, config, condition, snr=None, epoch=0):
    if noise_record['split'] != record['split']:
        raise ValueError('Noise role does not match speech split')
    seed = stable_seed(record['recording_id'],condition,epoch,config['seed'])
    rng = np.random.default_rng(seed)
    apply = True if snr is not None else bool(rng.random() < config['noise_probability'])
    snr = float(snr) if snr is not None else float(rng.uniform(config['train_snr_db_min'],config['train_snr_db_max'])) if apply else None
    maximum = max(0,noise_record['num_samples']-record['prepared_num_samples'])
    offset = int(rng.integers(0,maximum+1)) if apply else None
    return {'recording_id':record['recording_id'],'split':record['split'],'condition':condition,'epoch':epoch,
        'applied':apply,'seed':seed,'global_seed':config['seed'],'snr_db':snr,
        'speech_audio_path':record['audio_path'],'speech_sha256':record['audio_sha256'],
        'noise_environment':noise_record['environment'] if apply else None,
        'noise_audio_path':noise_record['audio_path'] if apply else None,
        'noise_sha256':noise_record['audio_sha256'] if apply else None,
        'noise_offset_samples':offset,'noise_channel':noise_record['channel'] if apply else None,
        'sample_rate':16000,'num_samples':record['prepared_num_samples'],
        'peak_limit':config['peak_limit'],'snr_definition':config['snr_definition'],
        'target_policy':'unchanged training_target from speech manifest'}


def materialize(record, plan):
    from .preprocessing import read_audio
    speech, sr = read_audio(ROOT/record['audio_path'])
    if not plan['applied']:
        return speech.copy(), {'applied':False}
    noise, noise_sr = read_audio(ROOT/plan['noise_audio_path'])
    if sr != noise_sr or len(speech) != plan['num_samples']:
        raise ValueError('Recipe no longer matches waveform')
    segment = noise_segment(noise,plan['noise_offset_samples'],len(speech))
    mixture, _, _, metadata = mix_at_snr(speech,segment,plan['snr_db'],plan['peak_limit'])
    return mixture, metadata
