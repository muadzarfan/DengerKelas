from pathlib import Path
from importlib.metadata import version
from faster_whisper import WhisperModel
import ctranslate2
from .transcript import Word


class LocalWhisper:
    def __init__(self, settings):
        self.settings = settings
        path = Path(settings.local_model_path)
        for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"Missing local model file {name}; run python -m src.evaluation.download_assets --model")
        self.model = WhisperModel(
            str(path), device=settings.device, compute_type=settings.compute_type,
            cpu_threads=settings.cpu_threads, num_workers=1, local_files_only=True,
        )

    def decode(self, audio, offset):
        segments, _ = self.model.transcribe(
            audio, language=self.settings.language, task=self.settings.task,
            beam_size=self.settings.beam_size, best_of=1, temperature=0,
            word_timestamps=True, condition_on_previous_text=False,
            vad_filter=False, no_speech_threshold=0.6,
        )
        # Iterating the generator is part of inference and must be timed.
        return [Word(offset + word.start, offset + word.end, word.word)
                for segment in segments for word in (segment.words or [])]

    def metadata(self):
        return {"faster_whisper": version("faster-whisper"),
                "ctranslate2": version("ctranslate2"),
                "cuda_devices": ctranslate2.get_cuda_device_count(),
                "model_path": self.settings.local_model_path,
                "device": self.settings.device, "compute_type": self.settings.compute_type}
