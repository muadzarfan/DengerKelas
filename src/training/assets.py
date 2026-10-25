"""Acquire only official pinned multilingual training weights, reusing Step 3 configs."""
import json
import shutil
import urllib.request
import time
from src.data.common import ROOT, read_json, write_json, digest


def acquire():
    sources = read_json(ROOT / 'configs/data_sources.json')
    revision = sources['processor']['revision']
    snapshot = ROOT / 'data/raw/metadata/whisper_repository.json'
    if snapshot.exists():
        metadata = read_json(snapshot)
    else:
        url = f'https://huggingface.co/api/models/openai/whisper-tiny/revision/{revision}?blobs=true'
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
    if metadata['sha'] != revision:
        raise ValueError('Weights metadata must match the Step 3 processor revision')
    spec = next(x for x in metadata['siblings'] if x['rfilename'] == 'model.safetensors')
    folder = ROOT / 'data/checkpoints/whisper-tiny-training'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'model.safetensors'
    expected = spec['lfs']['sha256']
    reused = path.is_file() and path.stat().st_size == spec['size'] and digest(path) == expected
    if not reused:
        if shutil.disk_usage(ROOT).free < spec['size'] * 8:
            raise OSError('Insufficient space for weights and diagnostic checkpoints')
        url = f'https://huggingface.co/openai/whisper-tiny/resolve/{revision}/model.safetensors'
        temporary = path.with_suffix('.safetensors.part')
        for attempt in range(3):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if offset >= spec['size']:
                    if offset == spec['size'] and digest(temporary) == expected:
                        temporary.replace(path)
                        break
                    temporary.unlink()
                    offset = 0
                request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request, timeout=120) as response:
                    resume = bool(offset and response.status == 206)
                    if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise OSError('Incorrect resume range')
                    with temporary.open('ab' if resume else 'wb') as handle:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                if temporary.stat().st_size != spec['size'] or digest(temporary) != expected:
                    raise OSError('Official weight size/SHA256 mismatch')
                temporary.replace(path)
                break
            except OSError as exc:
                print(json.dumps({'event':'weight_download_retry','attempt':attempt+1,'error':str(exc)}),flush=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    for name in ('config.json', 'generation_config.json'):
        shutil.copyfile(ROOT / 'data/checkpoints/whisper-tiny-processor' / name, folder / name)
    result = {'repository':'openai/whisper-tiny','revision':revision,'multilingual':True,
              'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,
              'official_sha256':expected,'local_sha256':digest(path),'verified':True,'reused':reused}
    write_json(ROOT / 'data/manifests/step4_model.json', result)
    print(json.dumps(result), flush=True)
    return folder


if __name__ == '__main__':
    acquire()
