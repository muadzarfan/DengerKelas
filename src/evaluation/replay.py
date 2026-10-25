"""Real-rate replay through the same preprocessing/session/worker as the app."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from ..streaming.config import ROOT, load_settings
from ..streaming.runtime import LocalWhisper
from ..streaming.session import Session
from .measurements import summarize


def waveform(path, seconds=None):
    audio, rate = sf.read(path, dtype="float32")
    if seconds is not None:
        # Repeated real speech plus 1 s silence is explicitly a stress fixture,
        # not a continuous lecture or independently recorded speech.
        unit = np.concatenate([audio, np.zeros((rate,) + audio.shape[1:], np.float32)])
        audio = np.concatenate([unit] * (int(seconds * rate / len(unit)) + 1))[:int(seconds*rate)]
    return audio, rate


def paced(backend, settings, path, seconds=None, stop_at=None, source="paced_public_fixture"):
    audio, rate = waveform(path, seconds)
    if stop_at:
        audio = audio[:int(stop_at*rate)]
    session = Session(backend, settings, rate, source=source)
    block = round(rate * .25)
    begin = time.perf_counter()
    for start in range(0, len(audio), block):
        chunk = audio[start:start+block]
        target = begin + (start + len(chunk))/rate
        time.sleep(max(0, target-time.perf_counter()))
        session.push(chunk)
        if session.error:
            break
    session.close()
    summary = summarize(session.log_path)
    target = ROOT / "results/step2" / (session.id + "_summary.json")
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="data/fixtures/fleurs_id_sample.wav")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--stop-at", type=float)
    parser.add_argument("--source", default="paced_public_fixture")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    settings = load_settings()
    backend = LocalWhisper(settings)
    print(json.dumps(backend.metadata()), flush=True)
    paced(backend, settings, Path(args.audio), args.seconds, args.stop_at, args.source)
