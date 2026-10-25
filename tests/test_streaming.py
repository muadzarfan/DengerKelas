"""Focused plumbing checks; synthetic signals/fake ASR do not test quality."""
import tempfile
import threading
import time
import unittest
import numpy as np
from src.streaming.audio import normalize_mono, StreamResampler
from src.streaming.config import Settings, ROOT
from src.streaming.session import Session
from src.streaming.transcript import Transcript, Word


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Timed out waiting for worker")


class FakeASR:
    def __init__(self):
        self.calls = []
    def decode(self, audio, offset):
        self.calls.append((offset, len(audio)))
        return [Word(offset, offset+len(audio)/16000, " ulang")]


class AudioTests(unittest.TestCase):
    def test_integer_and_stereo(self):
        x = np.array([[-32768, 32767], [16384, 16384]], dtype=np.int16)
        out = normalize_mono(x)
        self.assertAlmostEqual(out[0], -1/65536)
        self.assertAlmostEqual(out[1], .5)
        np.testing.assert_allclose(normalize_mono(np.array([0,128,255],np.uint8)), [-1,0,127/128])

    def test_continuous_resampling_matches_whole_signal(self):
        for rate in (44100, 48000, 16000):
            t = np.arange(rate*2)/rate
            x = (.2*np.sin(2*np.pi*440*t)).astype(np.float32)
            stream = StreamResampler(rate)
            parts = [stream.push(x[i:i+997]) for i in range(0,len(x),997)]
            parts.append(stream.flush())
            actual = np.concatenate(parts)
            whole = StreamResampler(rate)
            expected = np.concatenate([whole.push(x), whole.flush()])
            self.assertLessEqual(abs(len(actual)-32000), 1)
            np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_invalid_audio(self):
        for bad in (np.array([np.nan]),np.ones((2,20)),np.array([2.0])):
            with self.assertRaises(ValueError):
                normalize_mono(bad)


class TranscriptTests(unittest.TestCase):
    def test_repetition_and_overlapping_replacement(self):
        t = Transcript()
        t.update([Word(0,.4," ya"),Word(.5,.9," ya"),Word(1,1.4," lagi")],0)
        t.advance(1)
        t.update([Word(.5,.9," ya"),Word(1,1.4," lagi"),Word(1.5,1.9," lagi")],1)
        t.finish()
        self.assertEqual(t.snapshot()["history"], "ya ya lagi lagi")
        self.assertEqual(t.snapshot()["partial"], "")


class SessionTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "results/step2/test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.settings = Settings(initial_seconds=.2, update_seconds=.1, window_seconds=.6,
                                 overlap_seconds=.2,max_audio_seconds=1.2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rollover_stop_flush_and_bounded_storage(self):
        backend = FakeASR()
        s = Session(backend,self.settings,16000,self.tmp.name,"synthetic_test")
        for i in range(30):
            s.push(np.ones(1600,np.float32)*.1)
            if i:
                wait_for(lambda:s.decoded>=s.total)
        s.push(np.ones(480,np.float32)*.1)  # less than the update interval
        s.close()
        self.assertEqual(s.decoded, s.total)
        self.assertTrue(s.done)
        self.assertLessEqual(s.max_buffer, int(self.settings.max_audio_seconds*16000))
        self.assertGreater(s.base,0)
        self.assertTrue(any(e["event"]=="final_flush" for e in s.events))
        self.assertTrue(s.snapshot()["history"])

    def test_capture_continues_during_decode_and_overflow_is_explicit(self):
        entered, release = threading.Event(), threading.Event()
        class Blocking:
            def decode(self,audio,offset):
                entered.set(); release.wait(2); return []
        s=Session(Blocking(),self.settings,16000,self.tmp.name,"synthetic_test")
        s.push(np.ones(3200,np.float32)*.1)
        self.assertTrue(entered.wait(2))
        before=time.perf_counter()
        s.push(np.ones(3200,np.float32)*.1)
        self.assertLess(time.perf_counter()-before,.1)
        with self.assertRaisesRegex(RuntimeError,"No speech was silently discarded"):
            s.push(np.ones(16000,np.float32)*.1)
        s.fail("Expected overload test")
        release.set(); s.close(cancel=True)

    def test_cancel_then_new_session_is_empty(self):
        backend=FakeASR()
        a=Session(backend,self.settings,16000,self.tmp.name,"synthetic_test")
        a.push(np.ones(3200,np.float32)*.1)
        wait_for(lambda:a.decoded>0)
        a.close(cancel=True)
        b=Session(backend,self.settings,16000,self.tmp.name,"synthetic_test")
        self.assertNotEqual(a.id,b.id)
        self.assertEqual(b.total,0)
        self.assertEqual(b.snapshot()["history"],"")
        self.assertEqual(b.snapshot()["partial"],"")
        b.close()

    def test_silence_gate_and_failure_cleanup(self):
        backend=FakeASR()
        s=Session(backend,self.settings,16000,self.tmp.name,"synthetic_silence")
        s.push(np.zeros(4000,np.float32)); s.close()
        self.assertEqual(backend.calls,[])
        self.assertEqual(s.snapshot()["history"],"")
        class Broken:
            def decode(self,audio,offset): raise RuntimeError("deliberate failure")
        b=Session(Broken(),self.settings,16000,self.tmp.name,"synthetic_test")
        b.push(np.ones(4000,np.float32)*.1)
        wait_for(lambda:b.done)
        self.assertIn("deliberate failure",b.error)
        b.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
