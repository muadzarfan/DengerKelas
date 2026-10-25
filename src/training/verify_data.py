"""Executed manifest/augmentation integration checks, without final-test audio."""
import numpy as np
import soundfile as sf
from src.data.common import ROOT,read_json,read_jsonl,write_json,digest
from src.data.preprocessing import read_audio
from src.data.augmentation import materialize
from .data import records,SpeechDataset
from .model import load_processor
from .config import load


def verify():
    proc=load_processor()
    rows={split:records(split) for split in ('train','validation')}
    counts={}
    for split, group in rows.items():
        for r in group:
            path=ROOT/r['audio_path']; info=sf.info(path)
            assert info.samplerate==16000 and info.channels==1 and info.frames==r['prepared_num_samples']
            assert digest(path)==r['audio_sha256']
        counts[split]=len(group)
    noisy_id=next(r['recording_id'] for r in read_jsonl(ROOT/'data/manifests/train_e2_epoch0_recipes.jsonl') if r['applied'])
    r=next(r for r in rows['train'] if r['recording_id']==noisy_id)
    clean=SpeechDataset([r],proc)
    original_features=clean[0]['input_features'].copy()
    source_hash=digest(ROOT/r['audio_path'])
    e2=SpeechDataset([r],proc,'E2')
    a=e2[0]; b=e2[0]
    assert a['recipe']['noise_environment']=='OOFFICE' and a['recipe']['applied']
    assert a['labels']==b['labels']==clean[0]['labels']
    assert np.array_equal(a['input_features'],b['input_features'])
    assert np.array_equal(original_features,clean[0]['input_features'])
    assert source_hash==digest(ROOT/r['audio_path'])
    mixed,meta=materialize(r,a['recipe'])
    assert len(mixed)==r['prepared_num_samples'] and np.isfinite(mixed).all() and np.max(np.abs(mixed))<=.990001
    assert abs(meta['measured_snr_db']-a['recipe']['snr_db'])<1e-6
    validation=[]
    r=rows['validation'][0]
    fixed_plans=read_jsonl(ROOT/'data/manifests/validation_noisy_recipes.jsonl')
    for snr in (15,5):
        ds=SpeechDataset([r],proc,'fixed_noise',f'{snr}dB')
        item=ds[0]
        expected=next(x for x in fixed_plans if x['recording_id']==r['recording_id'] and x['snr_db']==snr)
        assert item['recipe']==expected and item['recipe']['noise_environment']=='OHALLWAY'
        assert item['labels']==proc.tokenizer(r['training_target']).input_ids
        x,m=materialize(r,item['recipe'])
        assert len(x)==r['prepared_num_samples'] and abs(m['measured_snr_db']-snr)<1e-6
        validation.append({'snr_db':snr,'same_step3_recipe':True,'length_preserved':True,'target_unchanged':True})
    rejected=False
    try: records('test')
    except ValueError: rejected=True
    assert rejected
    configs={k:load(ROOT/f'configs/training_{k}.json') for k in ('B','E1','E2')}
    assert all(c['physical_batch_size']*c['gradient_accumulation']==16 for c in configs.values())
    diff1={k for k in configs['B'] if configs['B'][k]!=configs['E1'][k]}
    diff2={k for k in configs['B'] if configs['B'][k]!=configs['E2'][k]}
    assert diff1=={'run_id','learning_rate'} and diff2=={'run_id','augmentation'}
    result={'status':'passed','eligible_train_validation_audio_verified':counts,
            'step3_verification_status':read_json(ROOT/'results/step3/verification.json')['status'] if (ROOT/'results/step3/verification.json').exists() else 'see versioned Step 3 evidence',
            'E2_example':{'recording_id':noisy_id,'same_seed_features_equal':True,'target_unchanged':True,
                          'clean_features_and_source_hash_unchanged':True,'noise_environment':'OOFFICE',**meta},
            'fixed_validation':validation,'test_loader_rejected':True,'no_test_audio_or_features_opened':True,
            'B_vs_E1_changed_fields':sorted(diff1),'B_vs_E2_changed_fields':sorted(diff2)}
    write_json(ROOT/'results/step4/data_integration_verification.json',result)
    print(result)
    return result


if __name__=='__main__':
    verify()
