import csv
import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from src.data.common import ROOT
from src.data.preprocessing import training_text,scoring_text,read_audio,metadata_rows,suspect_mojibake,prepare_example
from src.data.augmentation import mix_at_snr,noise_segment,recipe,stable_seed


class DataChecks(unittest.TestCase):
    def test_snr_and_shared_gain_prevent_clipping(self):
        rng=np.random.default_rng(5)
        speech=(rng.normal(size=16000)*.4).astype(np.float32)
        noise=(rng.normal(size=16000)*.3).astype(np.float32)
        for target in (5,15,20):
            mixed,s,n,meta=mix_at_snr(speech,noise,target)
            self.assertAlmostEqual(10*np.log10(np.mean(s*s)/np.mean(n*n)),target,places=9)
            np.testing.assert_allclose(mixed,s+n,atol=1e-7)
            self.assertLessEqual(np.max(np.abs(mixed)),.990001)
            self.assertLess(meta['common_gain'],1)
            self.assertEqual(mixed.shape,speech.shape)

    def test_zero_energy_and_wrap(self):
        x=np.ones(20,np.float32)*.1
        with self.assertRaisesRegex(ValueError,'Zero-energy noise'):
            mix_at_snr(x,np.zeros_like(x),5)
        mixed,_,_,meta=mix_at_snr(np.zeros_like(x),x,5)
        self.assertFalse(meta['applied']); self.assertFalse(mixed.any())
        np.testing.assert_array_equal(noise_segment(np.array([1,2,3]),2,5),[3,1,2,3,1])

    def test_seed_role_and_target_preservation(self):
        config={'seed':42,'noise_probability':.5,'train_snr_db_min':5,'train_snr_db_max':20,'peak_limit':.99,'snr_definition':'whole RMS'}
        record={'recording_id':'abc','split':'train','prepared_num_samples':1234,'audio_path':'clean.wav','audio_sha256':'a','training_target':'Belajar, belajar!'}
        noise={'split':'train','environment':'OOFFICE','num_samples':10000,'audio_path':'n.wav','audio_sha256':'b','channel':1}
        before=record.copy()
        a=recipe(record,noise,config,'E2'); b=recipe(record,noise,config,'E2')
        self.assertEqual(a,b); self.assertEqual(record,before)
        self.assertNotEqual(stable_seed('abc','E2',0),stable_seed('abc','E2',1))
        with self.assertRaisesRegex(ValueError,'role'):
            recipe(record,{**noise,'split':'test'},config,'E2')
        draws=[recipe({**record,'recording_id':str(i)},noise,config,'E2') for i in range(1000)]
        self.assertTrue(.43<np.mean([r['applied'] for r in draws])<.57)
        self.assertTrue(all(5<=r['snr_db']<=20 for r in draws if r['applied']))

    def test_minimal_training_vs_symmetric_scoring(self):
        original='  Belajar, belajar!  AI 2026 café  '
        self.assertEqual(training_text(original),'Belajar, belajar! AI 2026 café')
        self.assertEqual(scoring_text(original),'belajar belajar ai 2026 café')
        self.assertEqual(scoring_text('AI, 2026'),scoring_text('ai 2026'))
        self.assertTrue(suspect_mojibake('S\u00c3\u00a3o Paulo'))
        self.assertFalse(suspect_mojibake('São Paulo, café dan AI'))
        # Access guards must fire before any file read or processor call.
        with self.assertRaisesRegex(ValueError,'Final test'):
            prepare_example({'split':'test','eligible':True},None)
        with self.assertRaisesRegex(ValueError,'Excluded record'):
            prepare_example({'split':'train','eligible':False},None)

    def test_tsv_literal_quotes_and_real_resampling(self):
        scratch=ROOT/'results/step3/test_tmp'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            p=Path(folder)/'source.tsv'
            p.write_text('7\ta.wav\t"A", lalu B.\ta lalu b\ta | b\t48000\tMALE\n',encoding='utf-8')
            row=list(metadata_rows(p))[0]
            self.assertEqual(row['raw_transcription'],'"A", lalu B.')
            t=np.arange(48000)/48000
            stereo=np.column_stack([.2*np.sin(2*np.pi*400*t)]*2)
            wav=Path(folder)/'stereo.wav'; sf.write(wav,stereo,48000,subtype='PCM_16')
            mono,sr=read_audio(wav)
            self.assertEqual(sr,16000); self.assertEqual(len(mono),16000)
            self.assertEqual(mono.ndim,1); self.assertTrue(np.isfinite(mono).all())


if __name__=='__main__':
    unittest.main(verbosity=2)
