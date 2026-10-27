"""Read-only project/data/evidence audit; never generates final-test transcripts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import subprocess
import time
import numpy as np
import soundfile as sf
from src.data.common import ROOT,read_json,read_jsonl,write_json,digest
from src.data.acquire import verify as official_verify
from src.data.preprocessing import training_text

OUTPUT=ROOT/'results/audit_20261027'
BASE_COMMIT='def24cf1747dcb08d826211ec35dc3ba792aca64'


def audit():
    start=time.perf_counter()
    result={'date':'2026-10-27','base_commit':BASE_COMMIT,'scope':'integrity only; no final-test ASR, tuning or listening'}
    failures=[]
    sources=read_json(ROOT/'configs/data_sources.json')
    bad=[r['path'] for r in sources['files'] if not official_verify(ROOT/r['path'],r)]
    if bad: failures.extend(bad)
    result['official_artifacts']={'checked':len(sources['files']),'failed':bad}
    rows={split:read_jsonl(ROOT/f'data/manifests/fleurs_{split}.jsonl') for split in ('train','validation','test')}
    flat=[r for group in rows.values() for r in group]
    if len({r['recording_id'] for r in flat})!=len(flat):failures.append('duplicate recording IDs')
    def check(row):
        try:
            path=ROOT/row['audio_path']
            if digest(path)!=row['audio_sha256']:raise ValueError('audio SHA256 differs')
            x,sr=sf.read(path,dtype='float32',always_2d=True)
            if sr!=16000 or x.shape!=(row['prepared_num_samples'],1) or not np.isfinite(x).all():
                raise ValueError('audio shape/rate/finite mismatch')
            if row['training_target']!=training_text(row['raw_transcription']):raise ValueError('target changed')
            if row['eligible']:
                if len(x)>480000 or row['target_token_count']>448:raise ValueError('eligible model limits exceeded')
            elif not row['exclusion_reasons']:raise ValueError('missing exclusion reason')
            return None
        except Exception as exc:
            return {'recording_id':row['recording_id'],'error':str(exc)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        errors=[x for x in pool.map(check,flat) if x is not None]
    failures.extend(errors)
    result['speech']={s:{'total':len(group),'eligible':sum(r['eligible'] for r in group),
                          'excluded':sum(not r['eligible'] for r in group),
                          'seconds':round(sum(r['duration_seconds'] for r in group),2)} for s,group in rows.items()}
    result['speech_integrity']={'waveforms_decoded':len(flat),'failures':errors}
    noise=read_jsonl(ROOT/'data/manifests/demand_noise.jsonl')
    for row in noise:
        info=sf.info(ROOT/row['audio_path'])
        if (row['environment']!=sources['demand']['roles'][row['split']] or row['channel']!=1
            or info.samplerate!=16000 or info.channels!=1 or info.frames!=row['num_samples']
            or digest(ROOT/row['audio_path'])!=row['audio_sha256']):
            failures.append('noise integrity/role: '+row['split'])
    result['noise_roles']={r['split']:r['environment'] for r in noise}
    matched=0
    for step,key in (('step3','files'),('step4','actual_report_sha256')):
        folder=ROOT/'docs/evidence'/step
        snapshot=read_json(folder/'snapshot_checksums.json')
        for name,expected in snapshot[key].items():
            if digest(folder/name)!=expected:failures.append('historical evidence hash: '+step+'/'+name)
            else:matched+=1
    source_snapshot=read_json(ROOT/'docs/evidence/step4/snapshot_checksums.json')['source_sha256_at_export']
    source_matches=0; patched=[]
    for name,expected in source_snapshot.items():
        prior=subprocess.check_output(['git','show',BASE_COMMIT+':'+name],cwd=ROOT)
        if hashlib.sha256(prior).hexdigest()!=expected:failures.append('historical source hash: '+name)
        else:source_matches+=1
        if digest(ROOT/name)!=expected:patched.append(name)
    step3=read_json(ROOT/'docs/evidence/step3/snapshot_checksums.json')
    locks={**{'data/manifests/'+n:v for n,v in step3['current_manifest_sha256'].items()},
           'configs/data_sources.json':step3['data_source_config_sha256'],
           'configs/data_preparation.json':step3['preprocessing_config_sha256'],
           'data/manifests/test_lock.json':step3['test_lock_sha256']}
    for name,expected in locks.items():
        if digest(ROOT/name)!=expected:failures.append('Step 3 lock changed: '+name)
    result['historical_evidence']={'report_hashes_verified':matched,'source_hashes_verified_against_original_commit':source_matches,
                                   'working_source_changes_since_step4':patched,'step3_locks_preserved':not any(str(x).startswith('Step 3 lock') for x in failures)}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    forbidden=[]
    for name in tracked:
        p=Path(name)
        if p.name=='.gitkeep':continue
        if (name.startswith(('data/raw/','data/processed/','data/checkpoints/','checkpoints/','.cache/','.venv/'))
            or p.suffix in ('.wav','.mp3','.flac','.pt','.pth','.safetensors','.bin','.ckpt')):
            forbidden.append(name)
    failures.extend(forbidden)
    result['git']={'tracked_forbidden_artifacts':forbidden,'full_experiment_directories':[
        p.name for p in (ROOT/'results/step4').iterdir() if p.is_dir() and p.name.startswith(('B_','E1_','E2_'))]}
    result.update(status='passed' if not failures else 'failed',failures=failures,elapsed_seconds=time.perf_counter()-start)
    write_json(OUTPUT/'integrity.json',result)
    print(result,flush=True)
    if failures:raise AssertionError('Audit found integrity failures; see integrity.json')
    return result


if __name__=='__main__':
    audit()
