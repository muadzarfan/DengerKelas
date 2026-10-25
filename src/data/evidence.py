"""Export a small reviewable snapshot from actual local Step 3 reports."""
import shutil
from .common import ROOT, read_json, write_json, digest


def export_evidence():
    source = ROOT / 'results/step3'
    destination = ROOT / 'docs/evidence/step3'
    destination.mkdir(parents=True, exist_ok=True)
    # These reports contain public dataset identities/statistics, never user audio
    # or private microphone events. Raw archives and model artifacts stay local.
    names = [
        'speech_statistics.json', 'leakage_audit.json', 'flagged_recordings.jsonl',
        'recipe_statistics.json', 'verification.json', 'rerun_verification.json',
        'feature_smoke.json', 'augmentation_checks.json', 'extraction.json',
        'focused_checks.txt', 'distributions.png', 'acquisition.jsonl',
    ]
    for name in names:
        shutil.copyfile(source / name, destination / name)
    initial = destination / 'initial_v1'
    initial.mkdir(exist_ok=True)
    for name in ('speech_statistics.json', 'verification.json', 'test_lock.json'):
        historical = source / 'initial_v1' / name
        if historical.is_file():
            shutil.copyfile(historical, initial / name)
    manifests = ROOT / 'data/manifests'
    rerun = read_json(source / 'rerun_verification.json')
    current = {name: digest(manifests / name) for name in rerun['current_hashes']}
    if current != rerun['current_hashes']:
        raise ValueError('Manifest changed since the recorded verification run')
    lock = read_json(manifests / 'test_lock.json')
    if lock['preprocessing_config_sha256'] != digest(ROOT / 'configs/data_preparation.json'):
        raise ValueError('Preprocessing config changed since test lock creation')
    files = [p for p in sorted(destination.rglob('*'))
             if p.is_file() and p.name not in ('snapshot_checksums.json', 'README.md')]
    write_json(destination / 'snapshot_checksums.json', {
        'meaning': 'Exact copies of actual local reports; hashes make the snapshot reviewable',
        'files': {p.relative_to(destination).as_posix(): digest(p) for p in files},
        'current_manifest_sha256': current,
        'data_source_config_sha256': digest(ROOT / 'configs/data_sources.json'),
        'preprocessing_config_sha256': digest(ROOT / 'configs/data_preparation.json'),
        'test_lock_sha256': digest(manifests / 'test_lock.json'),
    })
    (destination / 'README.md').write_text('''# Step 3 evidence snapshot

These are exact copies of executed local reports from `results/step3/`, exported
by `python -m src.data.documentation`. They are review evidence, not fabricated
examples. The source of factual counts remains `results/step3/speech_statistics.json`;
this snapshot preserves the same bytes for source review. Configs/manifests are
stored in their normal project folders.

- `speech_statistics.json`: source/eligible counts, durations, token/text distributions, flags and source metadata.
- `leakage_audit.json`: exact PCM, normalized transcript and repeated source-ID audit, with recording IDs.
- `flagged_recordings.jsonl`: quality flags/exclusions; originals remain available locally.
- `verification.json`: actual integrity/recipe/feature check result, versions, data storage and limitations.
- `rerun_verification.json`: previous/current JSONL hashes, elapsed time and equality.
- `feature_smoke.json`, `augmentation_checks.json`: train/validation processor-label and real RMS-SNR checks only.
- `recipe_statistics.json`, `extraction.json`: actual recipe counts and source/channel inventory.
- `focused_checks.txt`: output of the five focused preparation unit checks.
- `acquisition.jsonl`: original download/retry/reuse and pipeline events; failures remain visible.
- `distributions.png`: all-official-row duration and target-token histograms, visually inspected.
- `initial_v1/`: superseded pre-mojibake-filter statistics/verification/lock retained to explain the v2 quality decision. Use current evidence for final counts.
- `snapshot_checksums.json`: report, manifest, config and test-lock checksums at export.

No raw audio, model weights, padded-feature collection, test model output,
private microphone events or participant records are included. Listening review
has not been performed. No training, ASR WER/CER result or five-user study is claimed.
''', encoding='utf-8')


if __name__ == '__main__':
    export_evidence()
