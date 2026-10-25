"""Download only a converted Tiny checkpoint and one public Indonesian clip."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import tarfile
import csv
import io
from ..streaming.config import ROOT


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as out:
        while block := source.read(1024 * 1024):
            out.write(block)
    partial.replace(destination)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def model():
    repo = "Systran/faster-whisper-tiny"
    info = fetch_json(f"https://huggingface.co/api/models/{repo}")
    revision = info["sha"]
    available = {f["rfilename"] for f in info["siblings"]}
    folder = ROOT / "data/checkpoints/whisper-tiny-ct2"
    target = ROOT / "data/manifests/step2_model.json"
    old = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    names = [n for n in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt", "preprocessor_config.json") if n in available]
    for name in names:
        path = folder / name
        verified = (path.is_file() and old.get("revision") == revision and
                    old.get("files", {}).get(name, {}).get("sha256") == sha256(path))
        if not verified:
            print("Downloading", name, flush=True)
            download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", path)
    record = {"source_model": "openai/whisper-tiny", "artifact_repository": repo,
              "revision": revision, "training": "No project fine-tuning; upstream CTranslate2 conversion",
              "license": "MIT", "license_url": "https://huggingface.co/Systran/faster-whisper-tiny",
              "original_model_card_license": "Apache-2.0", "original_model_card": "https://huggingface.co/openai/whisper-tiny",
              "files": {name: {"bytes": (folder / name).stat().st_size, "sha256": sha256(folder / name)} for name in names}}
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(target)


def fixture():
    # Dataset viewer returned HTTP 500/501 here. Stream only the beginning of
    # the official TRAIN archive, stop after its first WAV, never save the tar.
    # A hard compressed-byte cap prevents accidentally acquiring the full split.
    info = fetch_json("https://huggingface.co/api/datasets/google/fleurs")
    revision = info["sha"]
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{revision}/data/id_id"
    url = base + "/audio/train.tar.gz"
    destination = ROOT / "data/fixtures/fleurs_id_sample.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)

    class LimitedReader:
        def __init__(self, response):
            self.response, self.count = response, 0
        def read(self, size=-1):
            if size < 0 or self.count + size > 8 * 1024 * 1024:
                raise RuntimeError("Single-sample archive download exceeded 8 MiB limit")
            data = self.response.read(size)
            self.count += len(data)
            return data

    with urllib.request.urlopen(url, timeout=120) as response:
        limited = LimitedReader(response)
        with tarfile.open(fileobj=limited, mode="r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".wav"):
                    if member.size > 2 * 1024 * 1024:
                        raise RuntimeError("Unexpectedly large single WAV fixture")
                    destination.write_bytes(archive.extractfile(member).read())
                    member_name = member.name
                    break
            else:
                raise RuntimeError("No WAV found in the allowed archive prefix")
    with urllib.request.urlopen(base + "/train.tsv", timeout=60) as response:
        metadata = response.read(2 * 1024 * 1024).decode("utf-8")
    row = next(r for r in csv.reader(io.StringIO(metadata), delimiter="\t")
               if Path(r[1]).name == Path(member_name).name)
    import soundfile as sf
    info = sf.info(destination)
    record = {"dataset": "google/fleurs", "revision": revision, "config": "id_id", "split": "train",
              "row_id": row[0], "archive_member": member_name, "compressed_bytes_read": limited.count,
              "purpose": "Single public paced replay fixture, not a recognition benchmark or independent test set",
              "license": "CC BY 4.0", "license_url": "https://huggingface.co/datasets/google/fleurs",
              "source": url, "reference": row[2],
              "path": "data/fixtures/fleurs_id_sample.wav", "samplerate": info.samplerate,
              "frames": info.frames, "duration": info.duration, "channels": info.channels,
              "sha256": sha256(destination)}
    target = ROOT / "data/manifests/step2_fixture.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.model:
        model()
    if args.fixture:
        fixture()
