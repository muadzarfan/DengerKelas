"""Config-driven diagnostic/full runner. Full experiments require an explicit flag."""
import argparse
import math
import shutil
import time
import traceback
from pathlib import Path
from src.data.common import ROOT, write_json
from .config import load, setup, total_steps
from .data import records, short_subset, fixed_metric_subset, SpeechDataset
from .model import load_processor
from .engine import Engine


def run(config_path, allow_experiment=False, resume=None, output=None):
    cfg = load(config_path)
    if cfg['kind']=='experiment' and not allow_experiment:
        raise ValueError('Full B/E1/E2 runs are outside Step 4; use --allow-experiment only for Step 5')
    if cfg['kind']=='diagnostic' and cfg['max_steps']>64:
        raise ValueError('Diagnostic step budget exceeded')
    cfg = setup(cfg)
    if resume:
        from src.data.common import read_json
        previous = read_json(Path(resume)/'run_config.json')
        cfg['scheduler_total_steps'] = previous['scheduler_total_steps']
        cfg['train_recording_ids'] = previous['train_recording_ids']
        cfg['validation_recording_ids'] = previous['validation_recording_ids']
    cfg['run_id'] += '_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    output = Path(output) if output else ROOT/'results/step4'/cfg['run_id']
    proc = load_processor(resume)
    train_rows, val_rows = records('train'), records('validation')
    if cfg['kind']=='diagnostic':
        if resume:
            wanted = set(cfg['train_recording_ids']); train_rows = [r for r in train_rows if r['recording_id'] in wanted]
            wanted = set(cfg['validation_recording_ids']); val_rows = [r for r in val_rows if r['recording_id'] in wanted]
            # Restore saved order, not manifest iteration order.
            index = {r['recording_id']:r for r in train_rows}; train_rows = [index[i] for i in cfg['train_recording_ids']]
            index = {r['recording_id']:r for r in val_rows}; val_rows = [index[i] for i in cfg['validation_recording_ids']]
        else:
            train_rows = (fixed_metric_subset(train_rows,cfg['diagnostic_train_examples'],cfg['seed'])
                          if 'pilot' in cfg['run_id'] else short_subset(train_rows,cfg['diagnostic_train_examples']))
            val_rows = short_subset(val_rows,cfg['diagnostic_validation_examples'])
    cfg['train_recording_ids'] = [r['recording_id'] for r in train_rows]
    cfg['validation_recording_ids'] = [r['recording_id'] for r in val_rows]
    assert not set(cfg['train_recording_ids']) & set(cfg['validation_recording_ids'])
    train = SpeechDataset(train_rows,proc,cfg['augmentation'],cache_examples=cfg['feature_cache_examples'])
    metric_rows = train_rows if cfg['kind']=='diagnostic' else fixed_metric_subset(train_rows,cfg['train_metric_examples'],cfg['seed'])
    metric_train = SpeechDataset(metric_rows,proc,'clean',cache_examples=cfg['feature_cache_examples'])
    validation = SpeechDataset(val_rows,proc,'clean',cache_examples=cfg['feature_cache_examples'])
    engine = Engine(cfg,proc,train,output,resume)
    start = time.perf_counter()
    try:
        engine.inspect_batch(list(range(min(2,len(train)))))
        evaluations = []
        if cfg['kind']=='diagnostic' and 'overfit' in cfg['run_id'] and not resume:
            evaluations.append({'step':0,'train':engine.evaluate(metric_train,'tiny_train_before')})
        timings = []
        limit = total_steps(cfg,len(train))
        if engine.progress['global_step'] >= limit:
            raise ValueError('Checkpoint already reached the requested budget; no resume steps would be performed')
        while engine.progress['global_step'] < limit:
            target = min(limit,engine.progress['global_step']+cfg['eval_every_steps'])
            trained = engine.train_to(target,verify_update=engine.progress['global_step']==0)
            timings.extend(trained['step_seconds'])
            step = engine.progress['global_step']
            train_metrics = engine.evaluate(metric_train,f'train_subset_step{step}')
            val_metrics = engine.evaluate(validation,f'validation_clean_step{step}')
            extra = {}
            # Official conditions are supported; diagnostics only use tiny separate subsets.
            conditions = cfg['validation_conditions'] if cfg['kind']=='experiment' else []
            for condition in conditions:
                if condition != 'clean':
                    noisy = SpeechDataset(val_rows,proc,'fixed_noise',condition=condition)
                    extra[condition] = engine.evaluate(noisy,f'validation_{condition}_step{step}')
            evaluations.append({'step':step,'train':train_metrics,'validation_clean':val_metrics,'validation_noise':extra})
            current = val_metrics['wer']
            best = engine.progress['best_validation_wer']
            improve = best is None or current < best
            if improve: engine.progress['best_validation_wer'] = current
            # Tiny-overfit stopping uses training subset diagnostics, not validation tuning.
            stopping = (cfg['kind']=='diagnostic' and 'overfit' in cfg['run_id']
                        and step>=cfg['minimum_diagnostic_steps'] and train_metrics['wer']<=cfg['stop_train_wer']
                        and train_metrics['loss']<=cfg['stop_train_loss'])
            if cfg['kind']=='experiment' or step==limit or stopping:
                future = ROOT/'checkpoints'/cfg['run_id']/f'checkpoint-{step:06d}'
                if improve: engine.progress['best_checkpoint'] = str(future)
                saved = engine.save()
                if cfg['kind']=='experiment':
                    paths = sorted(saved.parent.glob('checkpoint-*'),key=lambda p:p.name)
                    keep = {saved,Path(engine.progress['best_checkpoint'])}
                    # Every deletion is a known checkpoint in this run's own folder.
                    for p in paths:
                        if len(paths)<=cfg['save_total_limit']: break
                        if (p not in keep and not p.is_symlink() and p.resolve().parent==saved.parent.resolve()
                            and (p/'training_state.pt').is_file()):
                            shutil.rmtree(p); paths.remove(p)
            if stopping: break
        if cfg['kind']=='diagnostic' and not list((ROOT/'checkpoints'/cfg['run_id']).glob('checkpoint-*')):
            saved = engine.save()
        result = {'status':'completed','diagnostic_only':cfg['kind']=='diagnostic','run_id':cfg['run_id'],
                  'train_examples':len(train_rows),'validation_examples':len(val_rows),
                  'train_metric_examples':len(metric_rows),'progress':engine.progress.copy(),'evaluations':evaluations,
                  'optimizer_step_seconds':timings,'elapsed_seconds':time.perf_counter()-start,
                  'checkpoint':str(saved),'output':str(output),'no_final_test_access':True,
                  'pretrained_initialization':str(resume) if resume else 'same pinned openai/whisper-tiny'}
        write_json(output/'summary.json',result)
        write_json(ROOT/'results/step4'/f"latest_{'overfit' if 'overfit' in cfg['run_id'] else 'pilot' if 'pilot' in cfg['run_id'] else 'smoke' if cfg['kind']=='diagnostic' else cfg['run_id'].split('_')[0]}.json",result)
        engine.log('run_completed',optimizer_steps=engine.progress['global_step'],elapsed_seconds=result['elapsed_seconds'])
        if cfg['kind']=='diagnostic' and 'overfit' in cfg['run_id'] and not resume:
            from .checkpoint import state_fingerprint
            # Independent reference branch for the new-process full-state resume check.
            reference_steps = engine.train_to(engine.progress['global_step']+2)
            reference = {'purpose':'two diagnostic continuation steps; not part of the overfit learning comparison',
                         'checkpoint_start':str(saved),'step_seconds':reference_steps['step_seconds'],
                         'fingerprint':state_fingerprint(engine.model,engine.optimizer,engine.scheduler,engine.progress)}
            write_json(output/'uninterrupted_continuation.json',reference)
        return result
    except BaseException as exc:
        engine.log('run_failed',error=repr(exc),traceback=traceback.format_exc(),progress=engine.progress)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--allow-experiment',action='store_true')
    p.add_argument('--resume'); p.add_argument('--output')
    args=p.parse_args()
    run(args.config,args.allow_experiment,args.resume,args.output)
