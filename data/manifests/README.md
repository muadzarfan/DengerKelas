# Data manifests

Public data provenance and Step 3 manifests live here. No audio, model weights, private microphone transcripts, user-study recordings or consent forms belong in this directory. FLEURS public reference transcripts are included with source/revision/attribution; source terms are documented in `docs/DATA_SOURCES_AND_LICENSES.md`.

- `fleurs_{train,validation,test}.jsonl`: every official source row, stable recording ID, reusable project-relative audio path, raw/target text, source metadata, file/PCM hashes, numeric audit, eligibility and exclusion reasons. Filter `eligible == true` for compatible single-window training/evaluation. Do not silently drop rows or change official splits.
- `demand_noise.jsonl`: source environment, exclusive split role, channel 1, actual sample count/duration, source archive and audio hashes, and the available channel inventory.
- `train_e2_epoch0_recipes.jsonl`: reproducible E2 epoch-0 examples; call `src.data.augmentation.recipe` with the new epoch for later training epochs.
- `{validation,test}_noisy_recipes.jsonl`: fixed 15/5 dB recipes with seeds, offsets, channel, component paths/hashes and SNR policy. Clean audio remains the speech manifest's original path. These are controlled mixtures, not classroom recordings.
- `test_lock.json`: final test speech/recipe/config hashes and policy. Test model feature access is denied by default in `prepare_example`.
- `step3_downloads.json` and `step3_metadata_downloads.json`: official artifact URLs, checksums and verified local hashes. Local raw archives/metadata snapshots are ignored by Git.

Do not edit generated JSONL by hand. Regenerate with `python -m src.data.prepare --stage all`; known verified raw artifacts are reused. Actual counts and limitations are centralized in `docs/DATASET_CARD.md`; rerun evidence is in `docs/evidence/step3/`. All dataset paths are relative to the DengerKelas project root, not temporary paths.
