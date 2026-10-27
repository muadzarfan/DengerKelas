"""Regression checks from the 2026-10-27 plan/code audit."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from src.data.common import ROOT
from src.data.common import digest,write_json
from src.training.config import load,next_evaluation_step
from src.training.checkpoint import signature,validate_resume_config,state_fingerprint
from src.training.integrity import verify_noise,verify_pretrained,verify_checkpoint
from src.training.data import SpeechDataset
from src.training.model import load_model


def resume_config():
    cfg=load(ROOT/'configs/training_B.json')
    cfg.update(resolved_precision='fp32',resolved_device='cpu',
               train_recording_ids=['train-a','train-b'],validation_recording_ids=['val-a'])
    return cfg


class AuditRegressions(unittest.TestCase):
    def test_resume_signature_locks_ordered_examples(self):
        before=resume_config()
        after=copy.deepcopy(before)
        after['train_recording_ids'].reverse()
        self.assertNotEqual(signature(before),signature(after))

    def test_resume_signature_locks_evaluation_policy(self):
        before=resume_config()
        after=copy.deepcopy(before)
        after['generation_max_new_tokens']=96
        self.assertNotEqual(signature(before),signature(after))

    def test_resume_rejects_changed_scope_and_validation(self):
        before=resume_config()
        validate_resume_config(before,copy.deepcopy(before))
        for key,value in [('kind','diagnostic'),('validation_recording_ids',['other-val']),
                          ('validation_conditions',['clean']),('selection_metric','loss')]:
            with self.subTest(key=key):
                after=copy.deepcopy(before); after[key]=value
                with self.assertRaisesRegex(ValueError,'Resume data/order'):
                    validate_resume_config(before,after)

    def test_resume_signature_is_an_immutable_snapshot(self):
        cfg=resume_config(); recorded=signature(cfg)
        cfg['train_recording_ids'].append('later-change')
        self.assertEqual(recorded['train_recording_ids'],['train-a','train-b'])

    def test_evaluation_stays_on_original_step_grid_after_resume(self):
        self.assertEqual(next_evaluation_step(160,159,636),318)
        self.assertEqual(next_evaluation_step(159,159,636),318)
        self.assertEqual(next_evaluation_step(630,159,636),636)

    def test_noise_corruption_and_wrong_role_rejected(self):
        scratch=ROOT/'results/audit_20261027/test_tmp'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path=Path(directory)/'noise.wav'
            sf.write(path,np.zeros(100,np.float32),16000,subtype='FLOAT')
            row={'split':'train','environment':'OOFFICE','channel':1,'num_samples':100,
                 'audio_path':path.relative_to(ROOT).as_posix(),'audio_sha256':digest(path)}
            verify_noise(row,'train','OOFFICE')
            with self.assertRaisesRegex(ValueError,'split/environment'):
                verify_noise(row,'validation','OHALLWAY')
            sf.write(path,np.ones(100,np.float32)*.1,16000,subtype='FLOAT')
            with self.assertRaisesRegex(ValueError,'Noise/manifest'):
                verify_noise(row,'train','OOFFICE')

    def test_E2_loader_invokes_noise_integrity_gate(self):
        with patch('src.training.integrity.verify_noise',side_effect=ValueError('changed noise')) as gate:
            with self.assertRaisesRegex(ValueError,'changed noise'):
                SpeechDataset([{'split':'train','eligible':True}],None,'E2')
            gate.assert_called_once()

    def test_model_integrity_checked_before_loading(self):
        with patch('src.training.integrity.verify_pretrained',side_effect=ValueError('changed weights')):
            with patch('src.training.model.WhisperForConditionalGeneration.from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError,'changed weights'):
                    load_model()
                loader.assert_not_called()

    def test_checkpoint_integrity_accepts_combined_processor_layout(self):
        scratch=ROOT/'results/audit_20261027/test_tmp'; scratch.mkdir(parents=True,exist_ok=True)
        names=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
               'run_config.json','data_identity.json','progress.json','training_state.pt','processor_config.json']
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder=Path(directory)
            for name in names:(folder/name).write_bytes(b'fixture-only')
            write_json(folder/'artifact_checksums.json',{name:digest(folder/name) for name in names})
            verify_checkpoint(folder)
            (folder/'config.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Checkpoint integrity mismatch'):
                verify_checkpoint(folder)

    def test_numpy_rng_position_is_part_of_full_fingerprint(self):
        original=np.random.get_state()
        try:
            model=torch.nn.Linear(1,1); optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
            a=state_fingerprint(model,optimizer,scheduler,{})
            state=np.random.get_state()
            np.random.set_state((state[0],state[1],(state[2]+1)%624,state[3],state[4]))
            b=state_fingerprint(model,optimizer,scheduler,{})
            self.assertNotEqual(a['numpy_rng'],b['numpy_rng'])
            self.assertEqual(a['schema_version'],2)
        finally:
            np.random.set_state(original)


if __name__=='__main__':
    unittest.main(verbosity=2)
