"""One decoder thread; bounded raw audio; no queue of inference jobs."""
from collections import deque
from pathlib import Path
import json
import threading
import time
import uuid
import numpy as np
import psutil
from .audio import StreamResampler
from .config import ROOT
from .transcript import Transcript


class Session:
    def __init__(self, backend, settings, input_rate, log_dir=None, source="microphone"):
        self.id = uuid.uuid4().hex
        self.settings, self.backend = settings, backend
        self.resampler = StreamResampler(input_rate)
        self.condition = threading.Condition(threading.RLock())
        self.log_lock = threading.Lock()
        self.parts = deque()
        self.base = self.total = self.cursor = self.decoded = 0
        self.max_buffer = 0
        self.transcript = Transcript()
        self.capturing, self.processing = True, False
        self.stop_requested = self.cancelled = self.done = False
        self.error = None
        self.started = time.perf_counter()
        self.first_voice = self.first_text = None
        self.decode_total = 0.0
        self.update_id = 0
        self.last_decode_seconds = None
        self.events = deque(maxlen=2000)
        self.process = psutil.Process()
        self.log_dir = Path(log_dir or ROOT / "results/step2")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{self.id}.jsonl"
        self.log_handle = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.log("session_start", source=source, input_rate=input_rate, config=settings.to_dict())
        self.worker = threading.Thread(target=self._run, name="DengerKelas-ASR", daemon=True)
        self.worker.start()

    def log(self, event, **values):
        row = {"event": event, "session": self.id, "t": time.perf_counter() - self.started, **values}
        with self.log_lock:
            self.events.append(row)
            if not self.log_handle.closed:
                self.log_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def push(self, samples):
        with self.condition:
            if not self.capturing or self.done:
                raise RuntimeError(self.error or "Session is not capturing audio")
            x = self.resampler.push(samples)
            self._append(x)
            self.log("chunk_arrival", input_samples=len(samples), output_samples=len(x),
                     audio_end=self.total / 16000, buffered_samples=self.total - self.base)
            self.condition.notify_all()

    def _append(self, x):
        if not len(x):
            return
        if self.total + len(x) > int(self.settings.max_session_seconds * 16000):
            raise RuntimeError("Session limit reached. Stop and reset before a new session.")
        if self.total - self.base + len(x) > int(self.settings.max_audio_seconds * 16000):
            raise RuntimeError("Audio buffer is full: recognition cannot keep up. No speech was silently discarded. Stop and review the log.")
        self.parts.append(x.copy())
        self.total += len(x)
        self.max_buffer = max(self.max_buffer, self.total - self.base)
        rms = float(np.sqrt(np.mean(x * x)))
        if self.first_voice is None and rms >= self.settings.energy_threshold:
            self.first_voice = time.perf_counter() - self.started
            self.log("first_energy_onset", rms=rms, note="RMS heuristic, not trained VAD")

    def stop(self):
        with self.condition:
            if not self.stop_requested and not self.done:
                self._append(self.resampler.flush())
                self.capturing = False
                self.stop_requested = True
                self.log("stop_received", audio_end=self.total / 16000)
                self.condition.notify_all()

    def fail(self, message):
        with self.condition:
            self.error = str(message)
            self.capturing = False
            self.cancelled = True
            self.log("error", message=self.error)
            self.condition.notify_all()

    def close(self, cancel=False, timeout=30):
        if cancel:
            with self.condition:
                self.capturing = False
                self.cancelled = True
                self.log("cancel_requested")
                self.condition.notify_all()
        else:
            self.stop()
        self.worker.join(timeout)
        if self.worker.is_alive():
            raise RuntimeError("Previous decoder is still finishing; wait before starting a new session")
        with self.log_lock:
            self.log_handle.close()

    def _trim(self, new_base):
        remove = new_base - self.base
        while remove > 0 and self.parts:
            part = self.parts.popleft()
            if len(part) > remove:
                self.parts.appendleft(part[remove:].copy())
                remove = 0
            else:
                remove -= len(part)
        self.base = new_base

    def _run(self):
        try:
            while True:
                with self.condition:
                    if self.cancelled:
                        break
                    pending = self.total - self.decoded
                    minimum = self.settings.initial_seconds if self.decoded == 0 else self.settings.update_seconds
                    ready = pending >= int(minimum * 16000)
                    if self.stop_requested and pending <= 0:
                        self.transcript.finish()
                        break
                    if not ready and not self.stop_requested:
                        self.condition.wait(0.1)
                        continue
                    start = self.cursor
                    end = min(self.total, start + int(self.settings.window_seconds * 16000))
                    if end <= start:
                        break
                    audio = np.concatenate(list(self.parts))[:end - self.base].copy()
                    if start != self.base:
                        raise RuntimeError("Audio timeline invariant failed")
                    final = self.stop_requested and end == self.total
                    self.processing = True
                    self.log("inference_start", audio_start=start / 16000, audio_end=end / 16000, final=final)
                begin = time.perf_counter()
                energy = float(np.sqrt(np.mean(audio * audio)))
                words = self.backend.decode(audio, start / 16000) if energy >= self.settings.energy_threshold else None
                elapsed = time.perf_counter() - begin
                with self.condition:
                    if self.cancelled:
                        break
                    self.processing = False
                    self.decoded = end
                    self.decode_total += elapsed
                    self.last_decode_seconds = elapsed
                    self.log("inference_finish", duration=elapsed, audio_start=start / 16000,
                             audio_end=end / 16000, lag=(self.total - end) / 16000,
                             rms=energy, gated=words is None, rss=self.process.memory_info().rss)
                    if words is not None:
                        self.transcript.update(words, start / 16000)
                    if end - start >= int(self.settings.window_seconds * 16000):
                        self.cursor = end - int(self.settings.overlap_seconds * 16000)
                        self.transcript.advance(self.cursor / 16000)
                        self._trim(self.cursor)
                        self.log("buffer_rollover", new_start=self.cursor / 16000, audio_end=self.total / 16000)
                    self.update_id += 1
                    text = self.transcript.snapshot()
                    if self.first_text is None and (text["history"] or text["partial"]):
                        self.first_text = time.perf_counter() - self.started
                    self.log("transcript_emitted", update_id=self.update_id, **text)
                    if final:
                        self.transcript.finish()
                        self.log("final_flush", audio_end=end / 16000)
                        break
        except Exception as exc:
            self.error = f"Local recognition failed: {exc}"
            self.log("error", message=self.error)
        finally:
            with self.condition:
                self.done, self.processing, self.capturing = True, False, False
                self.log("session_finished", audio_seconds=self.total / 16000,
                         decode_seconds=self.decode_total, max_buffer_samples=self.max_buffer,
                         error=self.error, cancelled=self.cancelled)
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            if self.error:
                status = "Error: " + self.error
            elif self.done:
                status = "Sesi berhenti"
            elif self.stop_requested:
                status = "Menyelesaikan audio terakhir"
            elif self.processing:
                status = "Audio masuk · model sedang memproses"
            elif self.decoded == 0:
                status = "Audio masuk · menunggu cukup audio"
            else:
                status = "Audio masuk · menunggu pembaruan berikutnya"
            return {"session": self.id, "status": status, "capturing": self.capturing,
                    "processing": self.processing, "done": self.done, "error": self.error,
                    "update_id": self.update_id, "audio_seconds": self.total / 16000,
                    "lag_seconds": (self.total - self.decoded) / 16000,
                    "processing_seconds": self.last_decode_seconds,
                    "buffer_seconds": (self.total - self.base) / 16000,
                    **self.transcript.snapshot()}
