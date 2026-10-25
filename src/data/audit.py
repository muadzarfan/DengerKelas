"""Read every waveform; preserve official membership and audit exact overlap."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import unicodedata
import numpy as np
import soundfile as sf
from .common import ROOT, read_json, write_json, write_jsonl, digest, relative, event
from .preprocessing import metadata_rows, training_text, scoring_text, read_audio, processor, suspect_mojibake


def numeric_summary(values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x): return {'n':0}
    return {'n':len(x),'min':float(x.min()),'p05':float(np.percentile(x,5)),
        'median':float(np.median(x)),'p95':float(np.percentile(x,95)),
        'max':float(x.max()),'mean':float(x.mean()),'sum':float(x.sum())}


def analyze_waveform(path, config):
    info=sf.info(path)
    x,sr=sf.read(path,dtype='float32',always_2d=True)
    result={'sample_rate':sr,'channels':x.shape[1],'num_samples':len(x),'duration_seconds':len(x)/sr,
        'audio_subtype':info.subtype,'flags':[],'exclusion_reasons':[],'audio_sha256':digest(path)}
    pcm=x.astype('<f4',copy=False)
    result['decoded_pcm_sha256']=hashlib.sha256(f'{sr}|{x.shape[1]}|'.encode()+pcm.tobytes()).hexdigest()
    if not len(x):
        result['exclusion_reasons'].append('empty_audio')
        return result
    if not np.isfinite(x).all():
        result['exclusion_reasons'].append('nonfinite_audio')
        return result
    result.update({'rms':float(np.sqrt(np.mean(x.astype(np.float64)**2))),
        'peak':float(np.max(np.abs(x))), 'dc_mean':float(np.mean(x.astype(np.float64))),
        'clipping_fraction':float(np.mean(np.abs(x)>=config['clipping_absolute_threshold']))})
    if result['rms']<=config['near_zero_rms']: result['exclusion_reasons'].append('near_zero_energy')
    if result['peak']>1.001: result['exclusion_reasons'].append('waveform_out_of_range')
    if result['duration_seconds']>config['max_audio_seconds']: result['exclusion_reasons'].append('longer_than_single_window')
    if result['clipping_fraction']>=config['clipping_fraction_warning']: result['flags'].append('possible_clipping')
    if result['duration_seconds']<.25: result['flags'].append('very_short_audio')
    return result


def audit_speech():
    sources=read_json(ROOT/'configs/data_sources.json')
    cfg=read_json(ROOT/'configs/data_preparation.json')
    proc=processor()
    records=[]
    for split,source_split in sources['fleurs']['split_mapping'].items():
        for metadata in metadata_rows(ROOT/f'data/raw/fleurs/{source_split}.tsv'):
            metadata.pop('source_graphemes')  # available in immutable source TSV; not an ASR target
            name=metadata['source_filename']
            target=training_text(metadata['raw_transcription'])
            records.append({'recording_id':f'fleurs:id_id:{source_split}:{name[:-4]}',
                'dataset':'google/fleurs','revision':sources['fleurs']['revision'],'config':'id_id',
                'split':split,'source_split':source_split,'source_archive':f'data/id_id/audio/{source_split}.tar.gz',
                'archive_member':f'{source_split}/{name}','source_metadata_path':f'data/raw/fleurs/{source_split}.tsv',
                'audio_path':f'data/raw/fleurs/audio/{source_split}/{name}',
                **metadata,'training_target':target,'scoring_reference':scoring_text(target),
                'speaker_id':None,'speaker_independence_verified':False})
    encoded=proc.tokenizer([r['training_target'] for r in records],padding=False,truncation=False).input_ids
    for row,labels in zip(records,encoded):
        row['target_token_count']=len(labels)
        row['text_characters']=len(row['training_target'])
        row['text_words']=len(row['training_target'].split())
    def check(row):
        row=dict(row)
        try:
            measurements=analyze_waveform(ROOT/row['audio_path'],cfg)
            row.update(measurements)
            row['source_audio_sha256']=measurements['audio_sha256']
        except Exception as exc:
            row.update({'flags':[],'exclusion_reasons':['audio_decode_failure'],'decode_error':str(exc)})
            if (ROOT/row['audio_path']).is_file(): row['audio_sha256']=digest(ROOT/row['audio_path'])
        target=row['training_target']
        if not target: row['exclusion_reasons'].append('empty_transcript')
        if '\ufffd' in target or any(unicodedata.category(c) in ('Cc','Cs') for c in target):
            row['exclusion_reasons'].append('invalid_text_character')
        if suspect_mojibake(target):
            row['flags'].append('suspect_mojibake')
            row['exclusion_reasons'].append('suspect_mojibake_pending_review')
        if row['target_token_count']>cfg['max_target_tokens']: row['exclusion_reasons'].append('target_token_limit')
        if row.get('num_samples')!=row['metadata_num_samples']: row['flags'].append('metadata_sample_count_mismatch')
        if row.get('num_samples') and row['text_words']/(row['duration_seconds'] or 1)>6:
            row['flags'].append('high_words_per_second_review_only')
        if row.get('num_samples') and not set(row['exclusion_reasons']) & {'audio_decode_failure','empty_audio','nonfinite_audio'}:
            if row['sample_rate']!=16000 or row['channels']!=1:
                output=ROOT/f"data/processed/fleurs/{row['split']}/{row['source_filename']}"
                output.parent.mkdir(parents=True,exist_ok=True)
                x,_=read_audio(ROOT/row['audio_path'])
                if not output.exists(): sf.write(output,x,16000,subtype='FLOAT')
                row['original_audio_path']=row['audio_path']
                row['audio_path']=relative(output)
                row['audio_sha256']=digest(output)
                row['prepared_num_samples']=len(x)
                row['preprocessing_action']='mono/resample_poly to 16 kHz'
            else:
                row['prepared_num_samples']=row['num_samples']
                row['preprocessing_action']='reuse already-compatible source waveform; no denoising or normalization'
            row['prepared_sample_rate']=16000
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(check,records))
    records.sort(key=lambda r:r['recording_id'])
    groups=defaultdict(list)
    for row in records:
        if row.get('decoded_pcm_sha256'): groups[row['decoded_pcm_sha256']].append(row)
    overlap=[]
    rank={'train':0,'validation':1,'test':2}
    for checksum,group in groups.items():
        if len(group)<2: continue
        splits=sorted({r['split'] for r in group})
        overlap.append({'pcm_sha256':checksum,'splits':splits,'recording_ids':[r['recording_id'] for r in group]})
        maximum=max(rank[r['split']] for r in group)
        for row in group:
            row['flags'].append('exact_audio_duplicate')
            if len({r['scoring_reference'] for r in group})>1:
                row['flags'].append('conflicting_transcripts_for_identical_audio')
                if row['split']!='test': row['exclusion_reasons'].append('ambiguous_transcript_for_identical_audio')
            if rank[row['split']]<maximum:
                row['exclusion_reasons'].append('exact_audio_overlap_with_evaluation_split')
        train=[r for r in group if r['split']=='train']
        for row in train[1:]: row['exclusion_reasons'].append('exact_audio_duplicate_within_train')
    transcripts=defaultdict(list)
    sentence_ids=defaultdict(list)
    for row in records:
        key=hashlib.sha256(row['scoring_reference'].encode()).hexdigest()
        transcripts[key].append(row)
        sentence_ids[row['source_sentence_id']].append(row)
        row['exclusion_reasons']=sorted(set(row['exclusion_reasons']))
        row['flags']=sorted(set(row['flags']))
        row['eligible']=not row['exclusion_reasons']
        row['validation_status']='excluded' if not row['eligible'] else 'valid_with_flags' if row['flags'] else 'valid'
    text_groups=[{'scoring_text_sha256':key,'splits':sorted({r['split'] for r in group}),
                  'unique_audio_hashes':len({r.get('decoded_pcm_sha256') for r in group}),
                  'recording_ids':[r['recording_id'] for r in group]}
                 for key,group in transcripts.items() if len(group)>1]
    stats={}
    for split in ('train','validation','test'):
        rows=[r for r in records if r['split']==split]
        eligible=[r for r in rows if r['eligible']]
        stats[split]={'before_count':len(rows),'after_count':len(eligible),
            'before_duration_seconds':sum(r.get('duration_seconds',0) for r in rows),
            'after_duration_seconds':sum(r.get('duration_seconds',0) for r in eligible),
            'exclusion_reasons':dict(Counter(reason for r in rows for reason in r['exclusion_reasons'])),
            'flag_counts':dict(Counter(flag for r in rows for flag in r['flags'])),
            'duration_seconds':numeric_summary([r['duration_seconds'] for r in rows if 'duration_seconds' in r]),
            'target_tokens':numeric_summary([r['target_token_count'] for r in rows]),
            'text_characters':numeric_summary([r['text_characters'] for r in rows]),
            'text_words':numeric_summary([r['text_words'] for r in rows]),
            'sample_rates':dict(Counter(str(r.get('sample_rate')) for r in rows)),
            'channels':dict(Counter(str(r.get('channels')) for r in rows)),
            'source_gender':dict(Counter(r['source_gender'] for r in rows)),
            'unique_source_sentence_ids':len({r['source_sentence_id'] for r in rows}),
            'unique_source_filenames':len({r['source_filename'] for r in rows})}
        write_jsonl(ROOT/f'data/manifests/fleurs_{split}.jsonl',rows)
    leakage={'exact_audio_duplicate_groups':overlap,
        'exact_audio_cross_split_groups':[g for g in overlap if len(g['splits'])>1],
        'repeated_scoring_transcript_groups':text_groups,
        'repeated_scoring_transcript_cross_split_groups':[g for g in text_groups if len(g['splits'])>1],
        'source_sentence_ids_reused':sum(len(v)>1 for v in sentence_ids.values()),
        'source_id_meaning':'sentence/content ID, not unique recording or speaker; filenames disambiguate recordings',
        'policy':cfg['leakage'],'speaker_independence_verified':False}
    write_json(ROOT/'results/step3/speech_statistics.json',stats)
    write_json(ROOT/'results/step3/leakage_audit.json',leakage)
    write_jsonl(ROOT/'results/step3/flagged_recordings.jsonl',[
        {k:r.get(k) for k in ('recording_id','split','source_filename','flags','exclusion_reasons','duration_seconds','target_token_count')}
        for r in records if r['flags'] or r['exclusion_reasons']])
    event('speech_audit_finished',counts={k:(v['before_count'],v['after_count']) for k,v in stats.items()},
          exact_cross_split_groups=len(leakage['exact_audio_cross_split_groups']),
          repeated_text_cross_split_groups=len(leakage['repeated_scoring_transcript_cross_split_groups']))
    return records,stats,leakage


def audit_noise():
    extracted=read_json(ROOT/'results/step3/extraction.json')['noise']
    config=read_json(ROOT/'configs/data_preparation.json')
    # Noise is not restricted to the 30-second speech input window.
    noise_config={**config,'max_audio_seconds':float('inf')}
    result=[]
    for record in extracted:
        audit=analyze_waveform(ROOT/record['audio_path'],noise_config)
        if audit['sample_rate']!=16000 or audit['channels']!=1 or audit['exclusion_reasons']:
            raise ValueError(f"Invalid selected noise: {record['environment']}: {audit}")
        result.append({**record,**audit})
    write_jsonl(ROOT/'data/manifests/demand_noise.jsonl',result)
    event('noise_audit_finished',environments=[{'environment':r['environment'],'seconds':r['duration_seconds'],
          'sample_rate':r['sample_rate'],'samples':r['num_samples']} for r in result])
    return result


if __name__=='__main__':
    audit_speech()
    audit_noise()
