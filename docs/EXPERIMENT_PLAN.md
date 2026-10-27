# Experiment and evaluation plan

This is the plan for full experiments, not a record of completed B/E1/E2 runs. Step 3 verified data/noise definitions; Step 4 verified the training pipeline using short diagnostics. No performance target is imposed by the supplied assignment summary. Independent user-data volume remains proposed, not collected. Actual diagnostic results and compute estimates are centralized in `STEP4_TRAINING_PIPELINE.md` and `evidence/step4/verification_summary.json`.

## Conditions

| Run | Initialization/model | Learning rate | Training audio | Role |
|---|---|---:|---|---|
| P0 | Original multilingual `openai/whisper-tiny` | — | No fine-tuning | Additional pretrained reference; does not satisfy DL construction alone |
| B | Same pretrained initialization | 1e-5 | Clean | Trained baseline |
| E1 | Same pretrained initialization | 5e-6 | Clean | Learning-rate comparison |
| E2 | Same pretrained initialization | 1e-5 | Clean + noise augmentation | Augmentation comparison |

B/E1/E2 are the baseline plus two changed conditions for Deep Learning. Common settings in `configs/training_B/E1/E2.json`: full-model supervised fine-tuning; teacher-forced token cross-entropy with padding ignored; AdamW; physical batch size 4; gradient accumulation 4; maximum 4 epochs; compatible mixed precision; same pinned pretrained initialization. These are controlled starting choices, not validated optimal values. Step 4 CPU profiles use physical 1 and accumulation 16, preserving effective 16. Choose one hardware/precision/profile for all comparisons. Never initialize E1/E2 from B.

## Step 4 fixed procedure for Step 5

Audit 27 October 2026 preserved this design. Resume now checks ordered example lists and decoding/evaluation settings, preserves the original evaluation grid and scheduler horizon, and verifies artifact/noise hashes at use time. Diagnostics cannot select a best project model or silently expand into a full-run budget. These safeguards do not change the planned experiment variables. See [current status](../PROJECT_STATUS.md).

- Seed 20261025; AdamW eps1e-8; weight decay0.01 on matrices, zero on biases/normalization vectors; linear scheduler with 5% warmup; max gradient norm1.0. These new shared choices control regularization/schedule and keep the procedure simple.
- Evaluate every159 optimizer steps (one current full epoch). Select lowest **clean validation generated corpus WER**, earliest checkpoint on ties. No test-driven choices. Retain best and latest checkpoints, maximum2; maximum4epochs, no experiment early stopping.
- Log token-weighted training CE, validation CE and free-generation WER. Training metric uses a fixed32-example hash-selected train subset; label it as a subset. Evaluate all eligible validation records on clean/OHALLWAY15/5dB. Scoring normalization is the Step 3 function applied symmetrically. Keep raw/normalized predictions and references.
- Every new full condition starts from the exact same verified official multilingual safetensors revision as the Step 3 processor. Run output/config/checkpoints are separate. Complete resume is for interruption recovery within the same configuration, not initialization of another condition.
- E2 reuses Step 3 recipe/materialize with record/epoch seeds and OOFFICE only. Validation fixed recipes remain unchanged; test stays locked.
- Step 4's tiny-overfit learning and tiny validation/noise metrics are diagnostics, not B/E1/E2 or P0 benchmark results. CPU pilot supports planning only; generation, I/O, checkpoint cost and thermal/runtime changes can alter extrapolated duration. GPU/Colab requires its own verification.

## Data and leakage controls

- Primary speech corpus: verified `google/fleurs`, Indonesian `id_id`, official splits. Version/counts/quality rules and limitations are in `DATASET_CARD.md`; source/attribution in `DATA_SOURCES_AND_LICENSES.md`. Preserve excluded source rows and keep manual-listening limitation visible.
- Noise: verified official DEMAND OOFFICE training, OHALLWAY validation, PCAFETER final test; channel1, with all channels of an environment assigned to its single role. Preserve the documented license-notice discrepancy and avoid environment overlap.
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
