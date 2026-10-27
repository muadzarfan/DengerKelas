"""Fail before model loading or mixing when pinned local artifacts have changed."""
from pathlib import Path
import soundfile as sf
from src.data.common import ROOT, read_json, digest


def verify_checkpoint(folder):
    folder=Path(folder)
    checks=read_json(folder/'artifact_checksums.json')
    required={'model.safetensors','config.json','generation_config.json','tokenizer.json',
              'tokenizer_config.json','run_config.json','data_identity.json','progress.json','training_state.pt'}
    # Transformers 5 saves combined processor_config.json; earlier versions use
    # preprocessor_config.json. Verify the actual supported saved layout.
    if not required.issubset(checks) or not {'processor_config.json','preprocessor_config.json'} & checks.keys():
        raise ValueError('Incomplete checkpoint checksum inventory')
    for name,expected in checks.items():
        if Path(name).name!=name or not (folder/name).is_file() or digest(folder/name)!=expected:
            raise ValueError(f'Checkpoint integrity mismatch: {name}')


def verify_processor():
    manifest=read_json(ROOT/'data/manifests/step3_downloads.json')
    selected=[f for f in manifest['files'] if f['path'].startswith('data/checkpoints/whisper-tiny-processor/')]
    if not selected:
        raise ValueError('Missing pinned processor provenance')
    for record in selected:
        path=ROOT/record['path']
        if not path.is_file() or digest(path)!=record['sha256']:
            raise ValueError(f'Processor integrity mismatch: {record["path"]}')


def verify_pretrained():
    verify_processor()
    record=read_json(ROOT/'data/manifests/step4_model.json')
    path=ROOT/record['path']
    if (not record['verified'] or record['official_sha256']!=record['local_sha256']
        or not path.is_file() or path.stat().st_size!=record['bytes']
        or digest(path)!=record['official_sha256']):
        raise ValueError('Pretrained weight integrity mismatch')
    for name in ('config.json','generation_config.json'):
        if digest(path.parent/name)!=digest(ROOT/'data/checkpoints/whisper-tiny-processor'/name):
            raise ValueError(f'Pretrained config differs from pinned Step 3 processor: {name}')


def verify_noise(record, split, environment):
    path=ROOT/record['audio_path']
    if record['split']!=split or record['environment']!=environment or record['channel']!=1:
        raise ValueError('Noise split/environment/channel mismatch')
    info=sf.info(path)
    if (info.samplerate!=16000 or info.channels!=1 or info.frames!=record['num_samples']
        or digest(path)!=record['audio_sha256']):
        raise ValueError('Noise/manifest integrity mismatch')
