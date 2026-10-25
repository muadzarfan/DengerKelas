"""Corpus WER from free generation, with symmetric Step 3 normalization."""
import time
import torch
from src.data.common import write_jsonl
from src.data.preprocessing import scoring_text
from src.training.model import generate


def word_errors(reference, prediction):
    a, b = reference.split(), prediction.split()
    distance = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): distance[i][0] = i
    for j in range(len(b)+1): distance[0][j] = j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            distance[i][j] = min(distance[i-1][j]+1, distance[i][j-1]+1,
                                 distance[i-1][j-1]+(a[i-1]!=b[j-1]))
    i, j = len(a), len(b)
    substitutions = deletions = insertions = 0
    while i or j:
        if i and j and a[i-1] == b[j-1] and distance[i][j] == distance[i-1][j-1]:
            i-=1; j-=1
        elif i and j and distance[i][j] == distance[i-1][j-1]+1:
            substitutions+=1; i-=1; j-=1
        elif i and distance[i][j] == distance[i-1][j]+1:
            deletions+=1; i-=1
        else:
            insertions+=1; j-=1
    return {'substitutions':substitutions,'deletions':deletions,'insertions':insertions,
            'reference_words':len(a),'errors':distance[-1][-1]}


def corpus_wer(pairs):
    counts = [word_errors(scoring_text(r), scoring_text(p)) for r,p in pairs]
    total = {k:sum(x[k] for x in counts) for k in ('substitutions','deletions','insertions','reference_words','errors')}
    if total['reference_words'] == 0:
        raise ValueError('Corpus WER needs at least one reference word')
    return {**total,'wer':total['errors']/total['reference_words']}


@torch.inference_mode()
def evaluate(model, dataset, collator, cfg, output_path, checkpoint, scope):
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    rows = []; total_loss = 0.; tokens = 0
    for i, record in enumerate(dataset.rows):
        item = dataset[i]
        batch = {k:v.to(cfg['resolved_device']) for k,v in collator([item]).items()}
        with torch.autocast(device_type=cfg['resolved_device'],dtype=torch.bfloat16 if cfg['resolved_precision']=='bf16' else torch.float16,
                            enabled=cfg['resolved_precision']!='fp32'):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite evaluation loss')
        n = int(batch['labels'].ne(-100).sum())
        total_loss += float(loss)*n; tokens += n
        predicted = generate(model,batch,cfg['generation_max_new_tokens'])
        text = dataset.proc.tokenizer.batch_decode(predicted, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        ids = predicted[0].tolist()
        if len(ids) > min(448,4+cfg['generation_max_new_tokens']):
            raise AssertionError('Generation exceeded the verified decoder-token budget')
        row = {'recording_id':record['recording_id'],'split':record['split'],'condition':dataset.condition,
               'scope':scope,'run_id':cfg['run_id'],'checkpoint':str(checkpoint),
               'reference':record['training_target'],'raw_prediction':text,
               'normalized_reference':scoring_text(record['training_target']),
               'normalized_prediction':scoring_text(text),'generated_token_count':len(ids),
               'generated_token_ids':ids,
               'generation_cap_without_eos':ids[-1] != model.config.eos_token_id,
               'teacher_forced_loss':float(loss),'target_tokens_without_decoder_start':n}
        row.update(word_errors(row['normalized_reference'],row['normalized_prediction']))
        rows.append(row)
    write_jsonl(output_path,rows)
    metric = corpus_wer([(r['reference'],r['raw_prediction']) for r in rows])
    result = {**metric,'loss':total_loss/tokens,'target_tokens':tokens,'examples':len(rows),
              'scope':scope,'split':dataset.rows[0]['split'],'condition':dataset.condition,
              'prediction_file':str(output_path),'seconds':time.perf_counter()-started,
              'generation_capped_examples':sum(r['generation_cap_without_eos'] for r in rows),
              'metric_source':'free model.generate; never teacher-forced argmax'}
    model.train(was_training)
    return result
