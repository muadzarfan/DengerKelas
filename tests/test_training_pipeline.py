import unittest
import torch
from src.data.common import ROOT
from src.evaluation.generated import word_errors,corpus_wer
from src.training.config import load,total_steps
from src.training.data import records,SpeechDataset,SpeechCollator,short_subset
from src.training.model import load_processor


class TrainingChecks(unittest.TestCase):
    def test_wer_counts_and_symmetric_normalization(self):
        self.assertEqual(word_errors('a b c','a x c d'),{'substitutions':1,'deletions':0,'insertions':1,'reference_words':3,'errors':2})
        self.assertEqual(word_errors('a b','')['deletions'],2)
        self.assertEqual(word_errors('','a')['insertions'],1)
        self.assertEqual(corpus_wer([('AI, 2026!','ai 2026')])['wer'],0.)
        # Corpus aggregation weights reference words, not an average of per-row WER.
        self.assertAlmostEqual(corpus_wer([('a','x'),('a b c','a b c')])['wer'],.25)

    def test_configs_effective_batch_and_controlled_variables(self):
        b=load(ROOT/'configs/training_B.json'); e1=load(ROOT/'configs/training_E1.json'); e2=load(ROOT/'configs/training_E2.json')
        self.assertEqual(total_steps(b,2538),636)
        self.assertEqual(b['physical_batch_size']*b['gradient_accumulation'],16)
        self.assertEqual({k for k in b if b[k]!=e1[k]},{'run_id','learning_rate'})
        self.assertEqual({k for k in b if b[k]!=e2[k]},{'run_id','augmentation'})

    def test_final_test_and_excluded_rows_rejected(self):
        with self.assertRaisesRegex(ValueError,'final test'):
            records('test')
        with self.assertRaisesRegex(ValueError,'final test'):
            SpeechDataset([{'split':'test','eligible':True}],None)
        with self.assertRaisesRegex(ValueError,'Excluded'):
            SpeechDataset([{'split':'train','eligible':False}],None)
        with self.assertRaisesRegex(ValueError,'train-only'):
            SpeechDataset([{'split':'validation','eligible':True}],None,'E2')

    def test_real_collator_masks_padding_preserves_eos(self):
        p=load_processor(); ds=SpeechDataset(short_subset(records('train'),2),p)
        batch=SpeechCollator(p,50258)([ds[0],ds[1]])
        self.assertEqual(tuple(batch['input_features'].shape),(2,80,3000))
        self.assertEqual(tuple(batch['attention_mask'].shape),(2,3000))
        self.assertTrue(torch.isfinite(batch['input_features']).all())
        self.assertTrue(batch['labels'].eq(-100).any())
        self.assertTrue(torch.all(batch['labels'][:,0].eq(50275)))
        for labels in batch['labels']:
            self.assertEqual(labels[labels.ne(-100)][-1],50257)
            self.assertFalse(labels.eq(50258).any())


if __name__=='__main__':
    unittest.main(verbosity=2)
