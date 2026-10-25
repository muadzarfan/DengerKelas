"""Stateful PyAV resampling; chunks are incremental, never cumulative."""
from fractions import Fraction
import av
import numpy as np


def normalize_mono(samples):
    x = np.asarray(samples)
    if x.ndim not in (1, 2) or (x.ndim == 2 and not 1 <= x.shape[1] <= 8):
        raise ValueError("Audio must have shape (samples,) or (samples, channels<=8)")
    if x.dtype.kind in "iu":
        info = np.iinfo(x.dtype)
        center = (info.max + 1) / 2 if x.dtype.kind == "u" else 0
        scale = (info.max + 1) / 2 if x.dtype.kind == "u" else max(abs(info.min), info.max)
        x = (x.astype(np.float32) - center) / scale
    elif x.dtype.kind == "f":
        x = x.astype(np.float32)
    else:
        raise ValueError("Unsupported audio sample representation")
    if not np.isfinite(x).all() or (x.size and np.max(np.abs(x)) > 1.001):
        raise ValueError("Audio contains nonfinite or out-of-range samples")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


class StreamResampler:
    def __init__(self, sample_rate):
        if not 8000 <= sample_rate <= 192000:
            raise ValueError("Unsupported input sample rate")
        self.input_rate = int(sample_rate)
        self.input_samples = 0
        self.output_samples = 0
        self.closed = False
        self.resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

    def _collect(self, frames):
        parts = [frame.to_ndarray().reshape(-1) for frame in frames]
        result = np.concatenate(parts) if parts else np.empty(0, np.float32)
        self.output_samples += len(result)
        return result.astype(np.float32, copy=False)

    def push(self, samples):
        if self.closed:
            raise ValueError("Resampler is already closed")
        x = normalize_mono(samples)
        if not len(x):
            return x
        frame = av.AudioFrame.from_ndarray(x[None, :], format="fltp", layout="mono")
        frame.sample_rate = self.input_rate
        frame.pts = self.input_samples
        frame.time_base = Fraction(1, self.input_rate)
        self.input_samples += len(x)
        return self._collect(self.resampler.resample(frame))

    def flush(self):
        if self.closed:
            return np.empty(0, np.float32)
        self.closed = True
        return self._collect(self.resampler.resample(None))
