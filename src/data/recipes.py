from .common import ROOT,read_json,read_jsonl,write_json,write_jsonl,digest,event
from .augmentation import recipe


def build_recipes():
    cfg=read_json(ROOT/'configs/data_preparation.json')['augmentation']
    noise={r['split']:r for r in read_jsonl(ROOT/'data/manifests/demand_noise.jsonl')}
    counts={}
    for split in ('train','validation','test'):
        records=[r for r in read_jsonl(ROOT/f'data/manifests/fleurs_{split}.jsonl') if r['eligible']]
        plans=[]
        for record in records:
            if split=='train':
                plans.append(recipe(record,noise[split],cfg,'E2_training',epoch=0))
            else:
                for snr in cfg['evaluation_snr_db']:
                    plans.append(recipe(record,noise[split],cfg,f'controlled_noise_{snr:g}dB',snr=snr))
        filename='train_e2_epoch0_recipes.jsonl' if split=='train' else f'{split}_noisy_recipes.jsonl'
        write_jsonl(ROOT/'data/manifests'/filename,plans)
        counts[split]={'recipes':len(plans),'noisy':sum(r['applied'] for r in plans),
                       'clean':sum(not r['applied'] for r in plans),'environment':noise[split]['environment']}
    lock={'policy':'Final test: integrity audit and fixed recipe generation only; no ASR, tuning, listening inspection or checkpoint selection',
        'dataset_revision':read_json(ROOT/'configs/data_sources.json')['fleurs']['revision'],
        'speech_manifest_sha256':digest(ROOT/'data/manifests/fleurs_test.jsonl'),
        'fixed_noise_recipes_sha256':digest(ROOT/'data/manifests/test_noisy_recipes.jsonl'),
        'preprocessing_config_sha256':digest(ROOT/'configs/data_preparation.json'),
        'noise_environment':'PCAFETER','noise_channel':1,'snr_db':cfg['evaluation_snr_db']}
    write_json(ROOT/'data/manifests/test_lock.json',lock)
    write_json(ROOT/'results/step3/recipe_statistics.json',counts)
    event('noise_recipes_ready',**counts)
    return counts


if __name__=='__main__':
    build_recipes()
