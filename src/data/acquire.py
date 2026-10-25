"""Official source discovery and bounded-retry, integrity-checked downloads."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .common import ROOT, read_json, write_json, digest, relative, event


def get_json(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError) as exc:
            event('metadata_retry', url=url, attempt=attempt + 1, error=str(exc))
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def discover():
    path = ROOT / 'configs/data_sources.json'
    if path.exists():
        event('reuse_pinned_sources', path=relative(path))
        return read_json(path)
    fleurs = get_json('https://huggingface.co/api/datasets/google/fleurs?blobs=true')
    whisper = get_json('https://huggingface.co/api/models/openai/whisper-tiny?blobs=true')
    demand = get_json('https://zenodo.org/api/records/1227121')
    write_json(ROOT / 'data/raw/metadata/fleurs_repository.json', fleurs)
    write_json(ROOT / 'data/raw/metadata/whisper_repository.json', whisper)
    write_json(ROOT / 'data/raw/metadata/demand_record.json', demand)
    entries = []
    def hf_entry(repo, revision, file, destination, dataset=False):
        lfs = file.get('lfs') or {}
        result = {'name': file['rfilename'], 'url': f"https://huggingface.co/{'datasets/' if dataset else ''}{repo}/resolve/{revision}/{file['rfilename']}",
                  'path': destination, 'size': file.get('size'), 'official_git_blob_sha1': file.get('blobId')}
        if lfs.get('sha256'):
            result['official_checksum'] = {'algorithm': 'sha256', 'value': lfs['sha256']}
        return result
    wanted = [f'data/id_id/{name}.tsv' for name in ('train', 'dev', 'test')]
    wanted += [f'data/id_id/audio/{name}.tar.gz' for name in ('train', 'dev', 'test')]
    wanted += ['README.md']
    siblings = {row['rfilename']: row for row in fleurs['siblings']}
    for name in wanted:
        folder = 'metadata' if name == 'README.md' else 'fleurs'
        entries.append(hf_entry('google/fleurs', fleurs['sha'], siblings[name],
                               'data/raw/' + folder + '/' + ('FLEURS_README.md' if name == 'README.md' else Path(name).name), True))
    needed = ['config.json', 'generation_config.json', 'preprocessor_config.json', 'tokenizer.json',
              'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'normalizer.json']
    siblings = {row['rfilename']: row for row in whisper['siblings']}
    for name in needed:
        if name in siblings:
            entries.append(hf_entry('openai/whisper-tiny', whisper['sha'], siblings[name], 'data/checkpoints/whisper-tiny-processor/' + name))
    demand_names = {'OOFFICE_16k.zip', 'OHALLWAY_16k.zip', 'PCAFETER_16k.zip', 'DEMAND.pdf'}
    for file in demand['files']:
        if file['key'] in demand_names:
            algorithm, checksum = file['checksum'].split(':', 1)
            entries.append({'name': file['key'], 'url': file['links']['self'], 'path': 'data/raw/demand/' + file['key'],
                            'size': file['size'], 'official_checksum': {'algorithm': algorithm, 'value': checksum}})
    assert {e['name'] for e in entries} >= demand_names
    result = {'fleurs': {'repository': 'google/fleurs', 'revision': fleurs['sha'], 'config': 'id_id',
                         'split_mapping': {'train': 'train', 'validation': 'dev', 'test': 'test'},
                         'card_license': (fleurs.get('cardData') or {}).get('license')},
              'processor': {'repository': 'openai/whisper-tiny', 'revision': whisper['sha']},
              'demand': {'record': '1227121', 'version': demand['metadata'].get('version'),
                         'metadata_license': demand['metadata'].get('license'),
                         'description_html': demand['metadata'].get('description'),
                         'roles': {'train': 'OOFFICE', 'validation': 'OHALLWAY', 'test': 'PCAFETER'}, 'channel': 1},
              'files': entries}
    total = sum(row.get('size') or 0 for row in entries)
    free = shutil.disk_usage(ROOT).free
    result['storage_preflight'] = {'free_bytes': free, 'download_bytes': total,
        'estimated_additional_extraction_cache_bytes': total * 3, 'conservative_required_bytes': total * 4 + 1024**3}
    if free < result['storage_preflight']['conservative_required_bytes']:
        raise OSError('Insufficient workspace disk for the conservative acquisition/extraction estimate')
    write_json(path, result)
    event('sources_pinned', fleurs_revision=fleurs['sha'], processor_revision=whisper['sha'],
          files=len(entries), download_bytes=total, free_bytes=free,
          demand_metadata_license=result['demand']['metadata_license'])
    return result


def verify(path, spec):
    if not path.is_file() or (spec.get('size') is not None and path.stat().st_size != spec['size']):
        return False
    checksum = spec.get('official_checksum')
    if checksum:
        return digest(path, checksum['algorithm']) == checksum['value']
    if spec.get('official_git_blob_sha1'):
        data = path.read_bytes()
        return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == spec['official_git_blob_sha1']
    return False  # Do not silently treat a file without verifiable provenance as checked.


def fetch_file(spec):
    path = ROOT / spec['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if verify(path, spec):
        event('download_verified_skip', file=spec['name'], bytes=path.stat().st_size)
    else:
        temporary = path.with_suffix(path.suffix + '.part')
        for attempt in range(3):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if spec.get('size') is not None and offset >= spec['size']:
                    if verify(temporary, spec):
                        temporary.replace(path)
                        break
                    temporary.unlink()  # only this task-owned incomplete artifact
                    offset = 0
                headers = {'User-Agent': 'DengerKelas-data-preparation/step3'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                request = urllib.request.Request(spec['url'], headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    resume = offset and response.status == 206
                    if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise OSError('Server returned a mismatched byte range; refusing to append it')
                    with temporary.open('ab' if resume else 'wb') as handle:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                if not verify(temporary, spec):
                    raise OSError(f"Downloaded artifact failed official size/checksum verification: actual_bytes={temporary.stat().st_size}, expected_bytes={spec.get('size')}")
                temporary.replace(path)
                event('download_complete', file=spec['name'], bytes=path.stat().st_size)
                break
            except (OSError, urllib.error.URLError) as exc:
                event('download_retry', file=spec['name'], attempt=attempt + 1, error=str(exc))
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return {**spec, 'sha256': digest(path), 'verified': True, 'actual_bytes': path.stat().st_size}


def acquire(metadata_only=False):
    config = discover()
    specs = [f for f in config['files'] if not metadata_only or f['name'].endswith(('.tsv', '.md', '.json', '.txt', '.pdf'))]
    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(fetch_file, spec): spec for spec in specs}
        for task in as_completed(tasks):
            try:
                completed.append(task.result())
            except Exception as exc:
                failures.append({'file': tasks[task]['name'], 'error': str(exc)})
    record = {'files': sorted(completed, key=lambda x: x['path']), 'failures': failures, 'metadata_only': metadata_only}
    write_json(ROOT / ('data/manifests/step3_metadata_downloads.json' if metadata_only else 'data/manifests/step3_downloads.json'), record)
    if failures:
        raise RuntimeError(f'Acquisition incomplete: {failures}')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    acquire(args.metadata_only)
