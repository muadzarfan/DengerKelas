from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_path: str = "data/checkpoints/whisper-tiny-ct2"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = 1
    language: str = "id"
    task: str = "transcribe"
    initial_seconds: float = 2.5
    update_seconds: float = 1.0
    window_seconds: float = 18.0
    overlap_seconds: float = 3.0
    max_audio_seconds: float = 24.0
    max_session_seconds: float = 300.0
    energy_threshold: float = 0.003
    sample_rate: int = 16000

    def __post_init__(self):
        if not (0 < self.overlap_seconds < self.window_seconds < self.max_audio_seconds):
            raise ValueError("Require 0 < overlap < window < maximum audio storage")
        if not (0 < self.initial_seconds <= self.window_seconds and self.update_seconds > 0):
            raise ValueError("Invalid inference cadence")
        if self.sample_rate != 16000 or self.language != "id" or self.task != "transcribe":
            raise ValueError("Step 2 uses 16 kHz Indonesian transcription")
        if self.max_session_seconds <= 0 or self.energy_threshold < 0:
            raise ValueError("Invalid session limit or energy threshold")

    @property
    def local_model_path(self):
        return str((ROOT / self.model_path).resolve())

    def to_dict(self):
        return asdict(self)


def load_settings(path=None):
    path = Path(path or os.environ.get("DENGERKELAS_CONFIG", ROOT / "configs/streaming.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_path", "device", "compute_type"):
        value = os.environ.get("DENGERKELAS_" + key.upper())
        if value:
            data[key] = value
    return Settings(**data)
