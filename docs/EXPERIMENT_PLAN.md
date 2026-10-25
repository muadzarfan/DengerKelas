# Experiment and evaluation plan

This is a preregistered planning outline, not a record of completed runs. No performance target is imposed by the supplied assignment summary. The 15/5 dB noise levels, hyperparameters, and approximate user-data volume below are project proposals and must be checked for feasibility.

## Conditions

| Run | Initialization/model | Learning rate | Training audio | Role |
|---|---|---:|---|---|
| P0 | Original multilingual `openai/whisper-tiny` | — | No fine-tuning | Additional pretrained reference; does not satisfy DL construction alone |
| B | Same pretrained initialization | 1e-5 | Clean | Trained baseline |
| E1 | Same pretrained initialization | 5e-6 | Clean | Learning-rate comparison |
| E2 | Same pretrained initialization | 1e-5 | Clean + noise augmentation | Augmentation comparison |

B/E1/E2 are the baseline plus two changed conditions for Deep Learning. Proposed common settings: full-model supervised fine-tuning; teacher-forced token cross-entropy with padding ignored; AdamW; physical batch size 4; gradient accumulation 4; maximum 4 epochs; compatible mixed precision; same initialization and controlled settings. These are starting choices, not validated optimal values. Record actual versions and any deviations.

## Data and leakage controls

- Primary speech corpus: `google/fleurs`, Indonesian `id_id`; preserve official train/validation/test splits. Verify version, counts, labels/transcripts, sample rates, duration distribution, missing/corrupt examples, and license when acquired.
- Noise: official DEMAND recordings, with proposed environment split OOFFICE for training, OHALLWAY for validation, PCAFETER for final test. Verify names, channel selection, files, and terms before use. Avoid noise recording overlap across splits.
- Use a consistent microphone channel and document mixing, clipping checks, and SNR calculation. Proposed test SNRs: 15 dB and 5 dB.
- Keep final noisy/clean test material isolated from training, hyperparameter selection, and checkpoint selection. Later collect approximately 15–25 minutes of consented Indonesian mini-explanations for independent practical evaluation; this is a proposed target, not collected data.
- Record source, license/attribution, manifest hashes, preprocessing, exclusions, and split membership. Keep raw audio and personal data out of Git.

## Training and checkpoint selection

- Use validation only for checkpoint selection and early-stage decisions; define selection metric/policy before runs. Never tune on the test split.
- Log epoch/step, training and validation loss, generated validation WER (and optionally CER), learning rate, throughput, elapsed training time, peak memory, hardware, seed, config, and checkpoint identity.
- Plot training/validation loss and metric for each trained condition. Inspect divergence, overfit, instability, and decoding errors. Preserve full run metadata.
- If validation WER selection is used, state decoding settings and keep them fixed across comparisons.

## Technical evaluation

- Score P0/B/E1/E2 on the same clean test and controlled noisy versions; report WER and supplementary CER, corpus-level method, normalization, and sample counts after verifying data metadata.
- Compare clean regression versus noisy robustness. Add representative transcription/reference pairs and categorize substitutions, deletions, insertions, proper nouns, code-switching, and noise-related errors where supported by evidence.
- Measure local first-display latency from speech/audio availability to first visible partial output; settled-text delay from word/audio endpoint to settled display; processing delay/real-time factor; sustained-stream backlog and buffer behavior. Define clocks and aggregation (median and tail percentile proposed) before collecting measurements.
- Measure CPU/GPU, RAM/VRAM, model size, startup time, and sustained inference throughput on the intended demonstration machine. Record machine/software conditions. No hardware or target threshold is assumed yet.
- Stabilization ablation: compare raw/unstabilized hypothesis display with agreement-based stabilization on matched audio and inference settings. Measure WER/CER, provisional revisions, settled delay, and backlog. Assess reading experience in the user study; do not infer benefit from stability alone.
- Capture screenshots, architecture diagram, event logs, and examples only from actual runs.

## User evaluation

Recruit at least five real Indonesian-speaking BINUS students from the target group. Obtain consent and use realistic short single-speaker explanation-following tasks. Observe actual use, collect questionnaire/interview/structured-form responses, and connect usability/reading feedback to measured stability and delay. Report anonymized evidence and recruitment limitations. AI-generated feedback is not evidence. Obtain lecturer guidance if fewer than five participants become necessary.

## Provisional engineering targets

No numeric performance/latency target is specified in the supplied requirements. Set provisional engineering thresholds only after the demonstration computer is identified and a baseline streaming prototype establishes feasible operating conditions; label them as project targets, not assignment requirements.
