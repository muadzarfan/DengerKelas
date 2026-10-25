from pathlib import Path
import hashlib
import json
import time
import threading

ROOT = Path(__file__).resolve().parents[2]
_EVENT_LOCK=threading.Lock()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def event(name, **values):
    row = {'event': name, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}
    path = ROOT / 'results/step3/acquisition.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with _EVENT_LOCK:
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
