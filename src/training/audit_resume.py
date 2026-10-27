"""Isolated v1→v2 checkpoint recovery check; historical Step 4 evidence is untouched."""
import subprocess
import sys
import time
from pathlib import Path
from src.data.common import ROOT,read_json,write_json
from .config import setup
from .data import records,SpeechDataset
from .model import load_processor
from .engine import Engine
from .checkpoint import state_fingerprint


def run():
    output=ROOT/'results/audit_20261027'
    historical=read_json(ROOT/'results/step4/latest_overfit.json')
    checkpoint=Path(historical['checkpoint'])
    cfg=setup(read_json(checkpoint/'run_config.json'))
    cfg['run_id']='audit_resume_reference_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    proc=load_processor(checkpoint)
    index={r['recording_id']:r for r in records('train')}
    train=SpeechDataset([index[i] for i in cfg['train_recording_ids']],proc,'clean',cache_examples=8)
    engine=Engine(cfg,proc,train,output/cfg['run_id'],resume=checkpoint)
    start=engine.progress['global_step']
    metric=engine.evaluate(train,'audit_train_before')
    migrated=engine.save('schema2-migrated-diagnostic-only')
    ref=engine.train_to(start+2,verify_update=True)
    write_json(engine.output/'uninterrupted_continuation.json',{
        'purpose':'two audit-only reference steps; no full experiment',
        'step_seconds':ref['step_seconds'],
        'fingerprint':state_fingerprint(engine.model,engine.optimizer,engine.scheduler,engine.progress)})
    summary={'checkpoint':str(migrated),'output':str(engine.output),'evaluations':[{'train':metric}]}
    path=output/'schema2_reference_summary.json'; write_json(path,summary)
    # Free the reference model before launching a separate recovery process.
    del engine,train,proc
    import gc
    gc.collect()
    result_path=output/'resume_verification.json'
    result=subprocess.run([sys.executable,'-u','-m','src.training.reload','--summary',str(path),
                           '--output-root',str(output),'--result-path',str(result_path)],cwd=ROOT)
    if result.returncode:
        raise RuntimeError('New-process audit recovery failed')
    checked=read_json(result_path)
    if checked['fingerprint_schema']!=2 or not checked['full_state_bitwise_equal']:
        raise AssertionError('Complete schema2 state comparison failed')
    return checked


if __name__=='__main__':
    print(run())
