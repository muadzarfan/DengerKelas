"""Standalone tiny generated-transcript evaluation; never accepts final test."""
import argparse
import time
from pathlib import Path
from src.data.common import ROOT,read_json,write_json
from src.training.config import setup
from src.training.data import records,SpeechDataset,SpeechCollator
from src.training.model import load_model,load_processor
from .generated import evaluate


def run(checkpoint=None):
    if checkpoint is None:
        checkpoint=Path(read_json(ROOT/'results/step4/latest_overfit.json')['checkpoint'])
    else:
        checkpoint=Path(checkpoint)
    cfg=setup(read_json(checkpoint/'run_config.json'))
    cfg['run_id']='diagnostic_small_validation_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    p=load_processor(checkpoint); model=load_model(checkpoint).to(cfg['resolved_device'])
    collator=SpeechCollator(p,model.config.decoder_start_token_id)
    index={r['recording_id']:r for r in records('validation')}
    rows=[index[i] for i in cfg['validation_recording_ids']]
    output=ROOT/'results/step4'/cfg['run_id']; output.mkdir(parents=True,exist_ok=False)
    result={'scope':'tiny validation wiring diagnostic; not a clean/noisy project benchmark',
            'checkpoint':str(checkpoint),'examples_per_condition':len(rows),'conditions':{}}
    for condition in ('clean','15dB','5dB'):
        dataset=SpeechDataset(rows,p,'clean' if condition=='clean' else 'fixed_noise',condition)
        result['conditions'][condition]=evaluate(model,dataset,collator,cfg,output/f'{condition}_predictions.jsonl',str(checkpoint),'smoke_validation_'+condition)
    write_json(output/'summary.json',result)
    write_json(ROOT/'results/step4/small_validation_verification.json',result)
    print(result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint')
    run(parser.parse_args().checkpoint)
