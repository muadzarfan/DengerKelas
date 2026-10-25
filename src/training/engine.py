"""Single-device full-model training with token-weighted accumulation and resume."""
import math
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from src.data.common import ROOT, write_json
from src.evaluation.generated import evaluate
from . import checkpoint
from .config import environment, total_steps
from .model import load_model
from .data import SpeechCollator


class Engine:
    def __init__(self, cfg, processor, train, output, resume=None):
        self.cfg, self.proc, self.train, self.output = cfg,processor,train,Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.env = environment()
        self.identity = checkpoint.identity()
        cfg['scheduler_total_steps'] = cfg.get('scheduler_total_steps') or total_steps(cfg,len(train))
        self.model = load_model(resume).to(cfg['resolved_device'])
        if cfg['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.collator = SpeechCollator(processor,self.model.config.decoder_start_token_id)
        decay, no_decay = [], []
        for p in self.model.parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
        self.optimizer = torch.optim.AdamW([{'params':decay,'weight_decay':cfg['weight_decay']},
                                           {'params':no_decay,'weight_decay':0.}],lr=cfg['learning_rate'],eps=1e-8)
        warmup = math.ceil(cfg['warmup_ratio']*cfg['scheduler_total_steps'])
        def lr_scale(step):
            if cfg['scheduler'] == 'constant': return 1.
            if step < warmup: return max(1,step)/max(1,warmup)
            return max(0.,(cfg['scheduler_total_steps']-step)/max(1,cfg['scheduler_total_steps']-warmup))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_scale)
        self.scaler = torch.amp.GradScaler('cuda',enabled=cfg['resolved_precision']=='fp16')
        self.progress = {'epoch':0,'next_micro_batch':0,'global_step':0,'micro_step':0,
                         'best_validation_wer':None,'best_checkpoint':None}
        if resume:
            self.progress = checkpoint.restore(Path(resume),self.optimizer,self.scheduler,self.scaler,cfg,self.identity)
        write_json(self.output/'run_config.json',cfg)
        write_json(self.output/'environment.json',self.env)
        write_json(self.output/'data_identity.json',self.identity)
        params = {'total':sum(p.numel() for p in self.model.parameters()),
                  'trainable':sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                  'frozen_names':[n for n,p in self.model.named_parameters() if not p.requires_grad],
                  'encoder_layers':self.model.config.encoder_layers,'decoder_layers':self.model.config.decoder_layers,
                  'd_model':self.model.config.d_model,'encoder_attention_heads':self.model.config.encoder_attention_heads,
                  'decoder_attention_heads':self.model.config.decoder_attention_heads,
                  'generation_language':self.model.generation_config.language,'task':self.model.generation_config.task,
                  'forced_decoder_ids':self.model.generation_config.forced_decoder_ids,
                  'prefix_ids':self.proc.tokenizer.prefix_tokens}
        write_json(self.output/'model_contract.json',params)
        self.log('run_started',resume_from=str(resume) if resume else None,progress=self.progress,**params)

    def log(self,event,**values):
        import json
        row = {'event':event,'run_id':self.cfg['run_id'],'kind':self.cfg['kind'],
               'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values}
        with (self.output/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)

    def autocast(self):
        precision = self.cfg['resolved_precision']
        return torch.autocast(device_type=self.cfg['resolved_device'],
                              dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision!='fp32')

    def batch(self, indices):
        return {k:v.to(self.cfg['resolved_device']) for k,v in self.collator([self.train[i] for i in indices]).items()}

    def inspect_batch(self, indices):
        items = [self.train[i] for i in indices]
        batch = self.collator(items)
        result = {'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(torch.isfinite(v).all())} for k,v in batch.items()},
                  'masked_tokens':int(batch['labels'].eq(-100).sum()),'labels':[]}
        for i,item in enumerate(items):
            labels = batch['labels'][i][batch['labels'][i].ne(-100)].tolist()
            decoded = self.proc.tokenizer.decode(labels,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            assert decoded == item['target'] and labels[-1] == self.model.config.eos_token_id
            assert labels[0] == 50275 and self.model.config.decoder_start_token_id not in labels
            result['labels'].append({'recording_id':item['recording_id'],'duration_seconds':item['duration_seconds'],
                                    'valid_tokens':len(labels),'decoded_label':decoded,'target_equal':True,
                                    'eos_preserved':True,'decoder_start_removed_once':True})
        write_json(self.output/'batch_check.json',result)
        self.log('batch_verified',tensors=result['tensors'],masked_tokens=result['masked_tokens'])
        return result

    def train_to(self, target_step, verify_update=False):
        timings = []; updates = []
        self.model.train()
        while self.progress['global_step'] < target_step:
            epoch = self.progress['epoch']
            self.train.epoch = epoch
            order = np.random.default_rng(self.cfg['seed']+epoch).permutation(len(self.train)).tolist()
            size = self.cfg['physical_batch_size']
            micros = [order[i:i+size] for i in range(0,len(order),size)]
            cursor = self.progress['next_micro_batch']
            group = micros[cursor:cursor+self.cfg['gradient_accumulation']]
            if not group:
                self.progress.update(epoch=epoch+1,next_micro_batch=0)
                continue
            start = time.perf_counter()
            expected_tokens = sum(self.train.rows[i]['target_token_count']-1 for ids in group for i in ids)
            before = {n:p.detach().cpu().clone() for n,p in self.model.named_parameters()} if verify_update else None
            self.optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.; actual_tokens = 0
            for ids in group:
                batch = self.batch(ids)
                n = int(batch['labels'].ne(-100).sum())
                with self.autocast():
                    loss = self.model(**batch).loss
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                # Every valid token has the same weight, including partial final groups.
                self.scaler.scale(loss*n/expected_tokens).backward()
                mean_loss += float(loss.detach())*n/expected_tokens
                actual_tokens += n
                self.progress['micro_step'] += 1
            if actual_tokens != expected_tokens: raise AssertionError('Loss accumulation denominator mismatch')
            self.scaler.unscale_(self.optimizer)
            gradient = {n:p.grad for n,p in self.model.named_parameters() if p.requires_grad}
            missing = [n for n,g in gradient.items() if g is None]
            nonfinite = [n for n,g in gradient.items() if g is not None and not torch.isfinite(g).all()]
            if missing or nonfinite:
                raise FloatingPointError(f'Missing/nonfinite gradients: {missing}, {nonfinite}')
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg['max_grad_norm'],error_if_nonfinite=True)
            used_lr = self.optimizer.param_groups[0]['lr']
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer); self.scaler.update()
            if self.scaler.get_scale() < old_scale:
                raise FloatingPointError('AMP skipped an update; stopping rather than recording a false optimizer step')
            self.scheduler.step()
            self.progress['global_step'] += 1
            self.progress['next_micro_batch'] += len(group)
            if self.progress['next_micro_batch'] == len(micros):
                self.progress.update(epoch=epoch+1,next_micro_batch=0)
            seconds = time.perf_counter()-start
            timings.append(seconds)
            row = {'global_step':self.progress['global_step'],'micro_step':self.progress['micro_step'],
                   'epoch':epoch+(cursor+len(group))/len(micros),'train_loss':mean_loss,'learning_rate':used_lr,
                   'gradient_norm_before_clip':float(norm),'micro_batches':len(group),
                   'examples':sum(len(g) for g in group),'valid_tokens':actual_tokens,'seconds':seconds,
                   'rss_bytes':psutil.Process().memory_info().rss,
                   'cuda_peak_bytes':torch.cuda.max_memory_allocated() if self.cfg['resolved_device']=='cuda' else None}
            if before is not None:
                changed = [n for n,p in self.model.named_parameters() if not torch.equal(p.detach().cpu(),before[n])]
                delta = {prefix:max(float((p.detach().cpu()-before[n]).abs().max()) for n,p in self.model.named_parameters() if n.startswith(prefix))
                         for prefix in ('model.encoder','model.decoder')}
                assert len(changed)>0 and all(x>0 for x in delta.values())
                update = {'loss_finite':True,'all_trainable_gradients_present_finite':True,
                          'gradient_tensors_checked':len(gradient),'changed_parameter_tensors':len(changed),
                          'total_parameter_tensors':len(before),'max_abs_update_by_component':delta,
                          'optimizer_step':self.progress['global_step'],'micro_batches_accumulated':len(group),
                          'accumulation':'sum valid-token losses / valid tokens across group'}
                updates.append(update); write_json(self.output/'parameter_update.json',update)
                verify_update = False
                del before
            self.log('optimizer_step',**row)
        return {'steps':len(timings),'step_seconds':timings,'parameter_update_checks':updates}

    def evaluate(self,dataset,name,checkpoint_name=None):
        result = evaluate(self.model,dataset,self.collator,self.cfg,self.output/f'{name}_predictions.jsonl',
                          checkpoint_name or f"step-{self.progress['global_step']}",name)
        write_json(self.output/f'{name}_metrics.json',result)
        self.log('evaluation',global_step=self.progress['global_step'],**result)
        return result

    def save(self, label=None):
        folder = ROOT/'checkpoints'/self.cfg['run_id']/(label or f"checkpoint-{self.progress['global_step']:06d}")
        started = time.perf_counter()
        checkpoint.save(folder,self.model,self.proc,self.optimizer,self.scheduler,self.scaler,
                        self.progress,self.cfg,self.identity,self.env)
        self.log('checkpoint_saved',path=str(folder),seconds=time.perf_counter()-started,progress=self.progress)
        return folder
