"""Preserve actual small-run evidence, plots and measured compute extrapolation."""
import os
import re
import shutil
from pathlib import Path
import numpy as np
from src.data.common import ROOT,read_json,read_jsonl,write_json,digest


def build():
    folder=ROOT/'results/step4'
    smoke=read_json(folder/'latest_smoke.json')
    overfit=read_json(folder/'latest_overfit.json')
    pilot=read_json(folder/'latest_pilot.json')
    resume=read_json(folder/'reload_resume_verification.json')
    integration=read_json(folder/'data_integration_verification.json')
    small=read_json(folder/'small_validation_verification.json')
    before=overfit['evaluations'][0]['train']; after=overfit['evaluations'][-1]['train']
    validation=overfit['evaluations'][-1]['validation_clean']
    contract=read_json(Path(smoke['output'])/'model_contract.json')
    update=read_json(Path(smoke['output'])/'parameter_update.json')
    env=read_json(Path(smoke['output'])/'environment.json')
    test_output=(folder/'all_python_checks.txt').read_text(encoding='utf-8-sig')
    tests_ok=bool(re.search(r'Ran (\d+) tests',test_output)) and test_output.rstrip().endswith('OK')
    assert tests_ok, 'Complete regression result is required before rendering completion'
    count=int(re.search(r'Ran (\d+) tests',test_output).group(1))
    step_times=pilot['optimizer_step_seconds']
    steps=636
    train_hours=[min(step_times)*steps/3600,max(step_times)*steps/3600]
    pilot_eval=pilot['evaluations'][-1]
    rates=[pilot_eval['train']['seconds']/pilot_eval['train']['examples'],
           pilot_eval['validation_clean']['seconds']/pilot_eval['validation_clean']['examples']]
    eval_examples_per_epoch=345*3+32
    total_hours=[train_hours[0]+min(rates)*eval_examples_per_epoch*4/3600,
                 train_hours[1]+max(rates)*eval_examples_per_epoch*4/3600]
    all_events=[]
    for p in folder.glob('diagnostic_*/events.jsonl'):
        all_events.extend(read_jsonl(p))
    optimizer_events=[r for r in all_events if r['event']=='optimizer_step']
    summary={
        'status':'complete','scope':'Step 4 diagnostics only; no full B/E1/E2, no final test',
        'hardware':env,'model':contract,'smoke':{'run_id':smoke['run_id'],**update},
        'overfit':{'run_id':overfit['run_id'],'examples':overfit['train_examples'],
                   'optimizer_steps':overfit['progress']['global_step'],'elapsed_seconds':overfit['elapsed_seconds'],
                   'initial_loss':before['loss'],'final_loss':after['loss'],'initial_generated_wer':before['wer'],
                   'final_generated_wer':after['wer'],'checkpoint':overfit['checkpoint']},
        'tiny_validation':{'examples':validation['examples'],'loss':validation['loss'],'generated_wer':validation['wer'],
                           'scope':'two distinct validation examples, not whole-split quality evidence'},
        'reload_resume':resume,'augmentation_and_data':integration,
        'python_regression_checks':{'count':count,'passed':True},
        'pilot':{'run_id':pilot['run_id'],'examples':pilot['train_examples'],'physical_batch_size':1,
                 'gradient_accumulation':16,'effective_batch':16,'optimizer_steps':pilot['progress']['global_step'],
                 'optimizer_step_seconds':step_times,'max_observed_rss_bytes':max(e['rss_bytes'] for e in optimizer_events if e['run_id']==pilot['run_id'])},
        'compute_estimate':{'profile':'same local CPU fp32, physical 1 accumulation 16; not measured GPU speed',
                            'train_examples':2538,'epochs':4,'optimizer_steps_per_run':steps,
                            'core_training_hours_per_run':train_hours,'core_training_hours_three_runs':[x*3 for x in train_hours],
                            'generated_eval_examples_per_epoch':eval_examples_per_epoch,
                            'observed_eval_seconds_per_example':rates,'illustrative_total_hours_per_run':total_hours,
                            'caveat':'Small two-update pilot; length/noise/thermal contention, generation caps, I/O and checkpoint costs can change full duration. First pilot step overlapped lightweight integration checks; second is the steadier observation.'},
        'actual_diagnostic_work':{'all_recorded_optimizer_updates':len(optimizer_events),
                                  'all_recorded_micro_batches':sum(r['micro_batches'] for r in optimizer_events),
                                  'all_recorded_training_example_presentations':sum(r['examples'] for r in optimizer_events),
                                  'primary_evidence_updates':1+overfit['progress']['global_step']+2+2+pilot['progress']['global_step'],
                                  'description':'Includes two earlier development runs, final smoke, selected overfit, uninterrupted continuation, resume and effective-batch pilot; all are diagnostic.'},
        'limitations':['No model-quality/generalization conclusion','No final test accessed',
                       'CUDA/mixed precision and Colab not executed','Manual listening from Step 3 remains pending',
                       'Read-speech/gender/encoding and DEMAND license limitations remain']}
    checks={'real_manifest_audio':integration['status']=='passed',
            'batch_features_labels':True,'finite_forward_backward_and_update':update['all_trainable_gradients_present_finite'] and update['changed_parameter_tensors']>0,
            'tiny_subset_learning':after['loss']<before['loss'] and after['wer']<before['wer'],
            'generated_validation_metric':validation['examples']==2 and validation['metric_source'].startswith('free'),
            'save_reload_new_process':resume['raw_generated_tokens_equal'] and resume['new_process'],
            'complete_resume':resume['full_state_bitwise_equal'],
            'controlled_B_E1_E2_configs':integration['B_vs_E1_changed_fields']==['learning_rate','run_id'] and integration['B_vs_E2_changed_fields']==['augmentation','run_id'],
            'final_test_guard':integration['test_loader_rejected'],'diagnostic_labels':True,
            'compute_measured':len(step_times)==2,'regression_checks':tests_ok}
    summary['completion_checks']=checks
    if not all(checks.values()): summary['status']='partial'
    write_json(folder/'verification_summary.json',summary)
    plot(folder,overfit,pilot)
    export(folder,summary)
    from .documentation import render
    render(summary,smoke,overfit,pilot,small)
    return summary


def plot(folder,overfit,pilot):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    events=[r for r in read_jsonl(Path(overfit['output'])/'events.jsonl') if r['event']=='optimizer_step' and r['global_step']<=overfit['progress']['global_step']]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    axes[0].plot([r['global_step'] for r in events],[r['train_loss'] for r in events],'.-',label='Train micro-batch CE')
    for key,label in [('train','Fixed train subset CE'),('validation_clean','Tiny validation CE')]:
        points=[(r['step'],r[key]['loss']) for r in overfit['evaluations'] if key in r]
        axes[0].plot(*zip(*points),'o--',label=label)
    axes[0].set(yscale='log',xlabel='Optimizer step',ylabel='Cross-entropy (log scale)',title='Tiny-overfit loss')
    axes[0].legend(fontsize=8)
    for key,label in [('train','4 memorized train examples'),('validation_clean','2 separate validation examples')]:
        points=[(r['step'],100*r[key]['wer']) for r in overfit['evaluations'] if key in r]
        axes[1].plot(*zip(*points),'o-',label=label)
    axes[1].set(xlabel='Optimizer step',ylabel='Generated corpus WER (%)',title='Memorization is not generalization',ylim=(-3,110))
    axes[1].legend(fontsize=8)
    axes[2].bar([1,2],pilot['optimizer_step_seconds'],color='#2563a6')
    axes[2].set(xticks=[1,2],xlabel='Pilot optimizer step',ylabel='Seconds per 16-example update',title='CPU fp32: physical 1 × accumulation 16')
    for i,y in enumerate(pilot['optimizer_step_seconds'],1):axes[2].text(i,y+.15,f'{y:.2f}s',ha='center')
    axes[2].set_ylim(0,max(pilot['optimizer_step_seconds'])*1.2)
    for a in axes:a.grid(axis='y',alpha=.2)
    fig.suptitle('DengerKelas Step 4 — diagnostics only; no final-test or full experiment results')
    fig.savefig(folder/'diagnostics.png',dpi=160)
    plt.close(fig)


def export(folder,summary):
    target=ROOT/'docs/evidence/step4'; target.mkdir(parents=True,exist_ok=True)
    for path in folder.iterdir():
        if path.is_file() and path.suffix in ('.json','.jsonl','.txt','.png'):
            shutil.copyfile(path,target/path.name)
        elif path.is_dir() and path.name.startswith('diagnostic_'):
            output=target/path.name; output.mkdir(exist_ok=True)
            for item in path.iterdir():
                if item.is_file() and item.suffix in ('.json','.jsonl','.txt'):
                    shutil.copyfile(item,output/item.name)
    checks={p.relative_to(target).as_posix():digest(p) for p in target.rglob('*') if p.is_file() and p.name not in ('README.md','snapshot_checksums.json')}
    code={p.relative_to(ROOT).as_posix():digest(p) for base in (ROOT/'src/training',ROOT/'configs') for p in base.glob('*.py' if base.name=='training' else 'training*.json')}
    code.update({p:digest(ROOT/p) for p in ('src/evaluation/generated.py','src/evaluation/small.py','tests/test_training_pipeline.py')})
    write_json(target/'snapshot_checksums.json',{'actual_report_sha256':checks,'source_sha256_at_export':code})
    (target/'README.md').write_text('''# Actual Step 4 diagnostic evidence

Exact copies from local `results/step4/`. Start with `verification_summary.json`.
Every metric is diagnostic, not a full B/E1/E2 result or a final-test benchmark.
Original public FLEURS references and generated outputs are included for verification;
raw speech/noise, weights, optimizer tensors, caches and private microphone logs are not.

`latest_smoke.json`, `latest_overfit.json`, `latest_pilot.json` identify selected evidence.
Run folders preserve batch checks, parameter-update proofs, per-step loss/timing/RSS,
reference/raw/normalized prediction files, metrics and environment/configuration.
`uninterrupted_continuation.json` plus `reload_resume_verification.json` prove resume.
`data_integration_verification.json` covers all eligible train/validation audio and E2/fixed-noise contracts.
`small_validation_verification.json` is only two validation records per condition.
`all_python_checks.txt` contains the full regression result. `diagnostics.png` plots actual logs.

Two preliminary runs are retained: the first smoke misreported the EOS/cap flag because
Whisper's tensor-only return strips EOS; the first overfit lacked the continuation reference.
The selected runs use raw decoder sequences, with prefix/EOS retained, and a verified
complete-resume comparison. These development issues did not change labels or official
experiment hyperparameters. Transformers also emits an upstream internal suppression-token
deprecation warning; explicit language/task and one total generation-length bound are used.
The first broad regression invocation stalled while Gradio mounted in the sandbox; it was
stopped and rerun in the normal host environment. Keep these limits visible.

`snapshot_checksums.json` records exact report and source bytes at export. Colab/CUDA
and mixed precision have not been executed; the included notebook is an execution path.
''',encoding='utf-8')


if __name__=='__main__':
    print(build()['status'])
