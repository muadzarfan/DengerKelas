"""Full manifest/file verification plus train/validation-only feature smoke."""
from collections import Counter
import json
import shutil
import importlib.metadata
import numpy as np
import soundfile as sf
from .common import ROOT,read_json,read_jsonl,write_json,digest,event
from .acquire import verify as verify_artifact
from .preprocessing import processor,prepare_example,padded_labels_for_smoke,read_audio
from .augmentation import recipe,materialize,noise_segment,mix_at_snr


def verify_all():
    cfg=read_json(ROOT/'configs/data_preparation.json')
    sources=read_json(ROOT/'configs/data_sources.json')
    checks=[]
    def check(name,condition,detail=None):
        result={'check':name,'passed':bool(condition),'detail':detail}
        checks.append(result)
        if not condition:
            write_json(ROOT/'results/step3/verification.json',{'status':'failed','checks':checks})
            raise AssertionError(f'{name}: {detail}')
    for spec in sources['files']:
        check('official_artifact_'+spec['name'],verify_artifact(ROOT/spec['path'],spec))
    records={s:read_jsonl(ROOT/f'data/manifests/fleurs_{s}.jsonl') for s in ('train','validation','test')}
    flat=[r for rows in records.values() for r in rows]
    check('recording_ids_unique',len({r['recording_id'] for r in flat})==len(flat),len(flat))
    for split,rows in records.items():
        for row in rows:
            path=ROOT/row['audio_path']
            check('file_exists_and_hash_'+row['recording_id'],path.is_file() and digest(path)==row['audio_sha256'])
            check('exclusion_has_reason_'+row['recording_id'],row['eligible'] or bool(row['exclusion_reasons']))
            if row['eligible']:
                info=sf.info(path)
                check('prepared_audio_'+row['recording_id'],info.samplerate==16000 and info.channels==1 and info.frames==row['prepared_num_samples'])
        source_split=sources['fleurs']['split_mapping'][split]
        from .preprocessing import metadata_rows
        check('official_split_count_'+split,len(rows)==len(list(metadata_rows(ROOT/f'data/raw/fleurs/{source_split}.tsv'))))
    noise={r['split']:r for r in read_jsonl(ROOT/'data/manifests/demand_noise.jsonl')}
    check('distinct_noise_environments',len({r['environment'] for r in noise.values()})==3)
    for split,row in noise.items():
        info=sf.info(ROOT/row['audio_path'])
        check('noise_valid_'+split,digest(ROOT/row['audio_path'])==row['audio_sha256'] and info.samplerate==16000 and info.channels==1 and info.frames==row['num_samples'])
        check('noise_role_'+split,row['environment']==sources['demand']['roles'][split] and row['channel']==1)
    eligible_hashes={s:{r['decoded_pcm_sha256'] for r in rows if r['eligible']} for s,rows in records.items()}
    check('no_exact_audio_leak_train_validation',not (eligible_hashes['train'] & eligible_hashes['validation']))
    check('no_exact_audio_leak_train_test',not (eligible_hashes['train'] & eligible_hashes['test']))
    check('no_exact_audio_leak_validation_test',not (eligible_hashes['validation'] & eligible_hashes['test']))
    recipe_counts={}
    for split in records:
        name='train_e2_epoch0_recipes.jsonl' if split=='train' else f'{split}_noisy_recipes.jsonl'
        plans=read_jsonl(ROOT/'data/manifests'/name)
        index={r['recording_id']:r for r in records[split] if r['eligible']}
        expected=len(index)*(1 if split=='train' else len(cfg['augmentation']['evaluation_snr_db']))
        check('recipe_count_'+split,len(plans)==expected)
        for plan in plans:
            row=index[plan['recording_id']]
            rebuilt=recipe(row,noise[split],cfg['augmentation'],plan['condition'],snr=plan['snr_db'] if split!='train' else None,epoch=plan['epoch'])
            check('recipe_reproducible_'+row['recording_id']+'_'+plan['condition'],plan==rebuilt)
            check('clean_source_retained_'+row['recording_id'],(ROOT/row['audio_path']).is_file())
            if plan['applied']:
                check('recipe_role_and_offset_'+row['recording_id']+'_'+plan['condition'],
                    plan['noise_environment']==noise[split]['environment'] and plan['noise_channel']==1 and
                    plan['num_samples']==row['prepared_num_samples'] and 0<=plan['noise_offset_samples']<=noise[split]['num_samples']-row['prepared_num_samples'])
        recipe_counts[split]=len(plans)
    lock=read_json(ROOT/'data/manifests/test_lock.json')
    check('test_lock_speech_hash',lock['speech_manifest_sha256']==digest(ROOT/'data/manifests/fleurs_test.jsonl'))
    check('test_lock_noise_hash',lock['fixed_noise_recipes_sha256']==digest(ROOT/'data/manifests/test_noisy_recipes.jsonl'))
    proc=processor()
    model_cfg=read_json(ROOT/'data/checkpoints/whisper-tiny-processor/config.json')
    prefixes=proc.tokenizer.convert_ids_to_tokens(proc.tokenizer.prefix_tokens)
    check('processor_indonesian_transcription_prefix',prefixes==['<|startoftranscript|>','<|id|>','<|transcribe|>','<|notimestamps|>'],prefixes)
    check('processor_sample_rate',proc.feature_extractor.sampling_rate==16000)
    check('processor_target_limit_matches',cfg['max_target_tokens']==model_cfg['max_target_positions'])
    feature_records=[]; prepared=[]; augmentation_checks=[]
    for split in ('train','validation'):  # Never inspect/test model features on final test data.
        samples=[r for r in records[split] if r['eligible']][:3]
        for row in samples:
            example=prepare_example(row,proc)
            features=example['input_features']
            check('feature_shape_'+row['recording_id'],features.shape==(80,3000) and features.dtype==np.float32 and np.isfinite(features).all())
            decoded=proc.tokenizer.decode(example['labels'],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            check('label_roundtrip_'+row['recording_id'],decoded==row['training_target'])
            prepared.append(example)
            feature_records.append({'recording_id':row['recording_id'],'split':split,'shape':list(features.shape),
                'dtype':str(features.dtype),'label_ids':example['labels'],'label_count':len(example['labels']),
                'special_token_prefix':prefixes,'finite':True,'roundtrip_equal':True})
        row=samples[0]
        speech,_=read_audio(ROOT/row['audio_path']); background,_=read_audio(ROOT/noise[split]['audio_path'])
        for target in (5.,15.):
            plan=recipe(row,noise[split],cfg['augmentation'],f'controlled_noise_{target:g}dB',snr=target)
            segment=noise_segment(background,plan['noise_offset_samples'],len(speech))
            mixed,clean_component,noise_component,metadata=mix_at_snr(speech,segment,target,cfg['augmentation']['peak_limit'])
            actual=10*np.log10(np.mean(clean_component**2)/np.mean(noise_component**2))
            second,meta2=materialize(row,plan)
            check('real_mix_snr_'+split+str(target),abs(actual-target)<1e-6)
            check('real_mix_reproducible_'+split+str(target),np.array_equal(mixed,second))
            check('real_mix_length_peak_'+split+str(target),len(mixed)==len(speech) and np.max(np.abs(mixed))<=.990001)
            noisy_features=prepare_example(row,proc,waveform=mixed)
            check('real_mix_target_unchanged_'+split+str(target),noisy_features['labels']==prepared[0 if split=='train' else 3]['labels'])
            augmentation_checks.append({'recording_id':row['recording_id'],'split':split,**metadata,
                'component_measured_snr_db':float(actual),'same_seed_bitwise_equal':True,'target_unchanged':True})
    padded=padded_labels_for_smoke(prepared,proc,model_cfg['decoder_start_token_id'])
    check('label_padding_minus100',bool(np.any(padded==-100)) and padded.dtype.kind in 'iu')
    check('decoder_start_stripped',not np.any(padded[:,0]==model_cfg['decoder_start_token_id']))
    check('valid_eos_preserved',any(model_cfg['eos_token_id'] in row for row in padded))
    write_json(ROOT/'results/step3/feature_smoke.json',{'samples':feature_records,'padding_shape':list(padded.shape),
        'padding_value':-100,'decoder_start_removed':True,'no_test_features_used':True,'framework':'NumPy; no training/model forward'})
    write_json(ROOT/'results/step3/augmentation_checks.json',augmentation_checks)
    storage={folder:sum(p.stat().st_size for p in (ROOT/folder).rglob('*') if p.is_file())
             for folder in ('data/raw','data/processed','data/manifests','data/checkpoints/whisper-tiny-processor')}
    result={'status':'passed','check_count':len(checks),'failed':0,'recipe_counts':recipe_counts,
        'storage_bytes':storage,'free_bytes_after':shutil.disk_usage(ROOT).free,
        'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','soundfile','transformers','tokenizers','matplotlib')},
        'listening_review':'Not performed: automated waveform audit and tokenizer roundtrip do not prove audio/transcript semantic alignment',
        'test_policy':'Integrity and fixed definitions only; no test inference, tuning, feature smoke, or listening selection'}
    write_json(ROOT/'results/step3/verification.json',result)
    # Store compact counts; detailed per-record exceptions are in flagged_recordings.jsonl.
    event('verification_passed',checks=len(checks),storage=storage)
    return result


if __name__=='__main__':
    verify_all()
