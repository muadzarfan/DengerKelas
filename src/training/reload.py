"""New-process local reload and optimizer/scheduler/RNG/sampler resume verification."""
import argparse
import time
from pathlib import Path
from src.data.common import ROOT,read_json,read_jsonl,write_json
from .config import setup
from .data import records,SpeechDataset
from .model import load_processor
from .engine import Engine
from .checkpoint import state_fingerprint


def verify(summary_path,output_root=None,result_path=None):
    summary=read_json(summary_path)
    checkpoint=Path(summary['checkpoint'])
    cfg=setup(read_json(checkpoint/'run_config.json'))
    cfg['run_id']='diagnostic_reload_resume_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    proc=load_processor(checkpoint)
    index={r['recording_id']:r for r in records('train')}
    train=SpeechDataset([index[i] for i in cfg['train_recording_ids']],proc,'clean',cache_examples=8)
    output_root=Path(output_root) if output_root else ROOT/'results/step4'
    engine=Engine(cfg,proc,train,output_root/cfg['run_id'],resume=checkpoint)
    before_progress=engine.progress.copy()
    original=summary['evaluations'][-1]['train']
    metric=engine.evaluate(train,'reloaded_train',str(checkpoint))
    before=read_jsonl(Path(original['prediction_file']))
    after=read_jsonl(Path(metric['prediction_file']))
    predictions_equal=[r['generated_token_ids'] for r in before]==[r['generated_token_ids'] for r in after]
    targets_equal=[r['reference'] for r in before]==[r['reference'] for r in after]
    assert predictions_equal and targets_equal
    assert abs(original['loss']-metric['loss'])<1e-7
    assert original['wer']==metric['wer']
    trained=engine.train_to(before_progress['global_step']+2,verify_update=True)
    reference=read_json(Path(summary['output'])/'uninterrupted_continuation.json')['fingerprint']
    schema=reference.get('schema_version',1)
    state=state_fingerprint(engine.model,engine.optimizer,engine.scheduler,engine.progress,schema_version=schema)
    comparison={key:state[key]==reference[key] for key in reference}
    if not all(comparison.values()):
        write_json(engine.output/'resume_comparison_failed.json',{'comparisons':comparison,'reference':reference,'resumed':state})
        raise AssertionError(f'Resume differs from uninterrupted continuation: {comparison}')
    saved=engine.save('resumed-diagnostic-only')
    result={'status':'passed','new_process':True,'checkpoint':str(checkpoint),
            'reloaded_processor':str(checkpoint),'raw_generated_tokens_equal':predictions_equal,
            'references_equal':targets_equal,'reloaded_loss':metric['loss'],'reloaded_wer':metric['wer'],
            'resume_start_progress':before_progress,'resume_end_progress':engine.progress,
            'resume_optimizer_steps':trained['steps'],'resume_step_seconds':trained['step_seconds'],
            'uninterrupted_vs_resumed':comparison,'full_state_bitwise_equal':all(comparison.values()),
            'fingerprint_schema':schema,'resume_contract_schema':2,
            'resumed_checkpoint':str(saved),'output':str(engine.output),'no_final_test_access':True}
    write_json(engine.output/'verification.json',result)
    write_json(Path(result_path) if result_path else ROOT/'results/step4/reload_resume_verification.json',result)
    engine.log('reload_and_resume_verified',**result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',default=str(ROOT/'results/step4/latest_overfit.json'))
    parser.add_argument('--output-root'); parser.add_argument('--result-path')
    args=parser.parse_args()
    verify(args.summary,args.output_root,args.result_path)
