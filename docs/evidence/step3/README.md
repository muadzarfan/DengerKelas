# Step 3 evidence snapshot

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
