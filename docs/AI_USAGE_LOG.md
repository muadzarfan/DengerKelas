# AI usage log

This log begins with the Step 1 planning/setup assistance. Update it during development and carry it into both course reports. The student must independently verify outputs and understand submitted work.

| Date | Tool | Purpose | Prompt/instruction summary | Output used | Student verification | Changes/decisions after review |
|---|---|---|---|---|---|---|
| 2026-10-25 | OpenAI Codex (GPT-6) | Scope finalization, assignment mapping, workspace/environment foundation | Execute Step 1 only for DengerKelas using supplied pasted project specification; inspect workspace, create brief/matrix/plan/decisions/environment/log/scaffold, and verify environment without fabricating evidence | Draft planning documents and repository skeleton; environment observations | User/student must review all project and assignment decisions. Workspace search found no official forms; environment checks and failure are recorded in `docs/ENVIRONMENT.md`. No research result or user feedback was generated. | Kept project concept and technical direction; renamed project to DengerKelas. Marked assignment source sections as supplied-summary references pending original-form review; left approvals, installs, data, and experiments open. |

## Step 2 assistance — 2026-10-25

| Field | Record |
|---|---|
| Tool | OpenAI Codex; local Python/Git tools; official web documentation; browser automation |
| Purpose | Implement and verify the smallest local Indonesian streaming feasibility prototype |
| Instruction summary | Continue the existing project; repair environment, use multilingual pretrained Tiny locally, implement continuous microphone streaming, bounded storage/worker, timestamp history, Stop/Reset, genuine checks and timing; defer training/full data/stabilization/user study |
| Outputs used | `app/`, `src/streaming/`, `src/evaluation/`, config, dependency lock, focused tests, provenance manifests, documentation and genuine replay logs |
| Agent verification | Installed-version/API inspection; pip check; 11 Python and two browser-controller checks; local socket-denied model load/inference; CLI/browser paced speech, 125-second sustained replay, silence, mid-speech Stop, Reset and consecutive-session regression; final-source browser smoke and output bundle exclusion checks |
| Student verification | Student replied “Passed on the intended demonstration computer” to the explicit microphone Start/live text/Stop/Reset checklist. Multiple actual microphone sessions appear in local event logs. Student review/understanding of code and academic claims is still required. |
| Decisions after evidence | Replaced blocked pip-less runtime with isolated Python 3.12; retained old environment. Used explicit PCM bridge for chunk/Stop ordering; kept CPU INT8/Tiny. Fixed a rapid-restart AudioContext race found in real microphone logs. Kept failed checks documented. |
| Scope and integrity | No fine-tuning, formal dataset acquisition, invented WER/noise results, generated user feedback, or five-user study. Microphone text is excluded from public evidence. Pass is limited to observed feasibility conditions. |

## Step 3 assistance — 2026-10-25

| Field | Record |
|---|---|
| Tool | OpenAI Codex; official Hugging Face/Zenodo web/API metadata; local Python/Git tools; pypdf for the source DEMAND document; Matplotlib |
| Purpose | Execute actual dataset acquisition, audit, preprocessing and reproducible noise preparation in the existing workspace |
| Instruction summary | Use official FLEURS Indonesian splits and OOFFICE/OHALLWAY/PCAFETER only; acquire automatically with integrity/reuse; audit all data, preserve source labels/splits; prepare Whisper features/labels and fixed noise definitions; verify and document without training or test tuning |
| Outputs used | `src/data/`, `configs/data_*.json`, `requirements/data.txt` and actual installed lock, focused data tests, public source/recording/noise/recipe manifests, dataset/license/preparation documentation, local `results/step3/` and public `docs/evidence/step3/` snapshots |
| Agent verification | All requested official files passed checksums; all source recordings were decoded/audited; IDs/leakage/filters, clean retention and noise roles checked; train/validation features/labels and real component SNR verified; focused unit checks and pip check passed; full rerun produced identical manifest bytes; distribution plot visually inspected |
| Decisions after observed failures | Recovered HTTP 502 and partial/checksum-failed downloads with bounded retries/resume. Corrected literal TSV quote parsing. Train/validation text inspection led to v2 holding suspected mojibake without guessing repaired names; retained initial audit evidence. Preserved the DEMAND license discrepancy and source gender imbalance. |
| Student verification | Student review/understanding of code, quality exclusions, source attribution and academic claims is still required. No student listening review or semantic transcript confirmation has been reported for Step 3. Numeric audit and tokenizer roundtrip are not substitutes. |
| Scope and integrity | No fine-tuning, model forward pass, ASR WER/CER benchmark, checkpoint selection, test tuning/listening/feature smoke, fabricated feedback or five-user study. Test access was limited to integrity and fixed definitions. Raw data/weights/caches and private Step 2 logs stay outside Git. |

## Step 4 assistance — 2026-10-25

| Field | Record |
|---|---|
| Tool | OpenAI Codex; local Python/PyTorch/Git tools; official PyTorch and Transformers documentation/source; Matplotlib |
| Purpose | Build and execute the multilingual Whisper Tiny training pipeline and bounded diagnostics |
| Instruction summary | Continue Step 1–3 artifacts; verify hardware/data; implement loader/collator, full-model training, generated metrics, noise integration, complete checkpoints/resume and controlled configs; execute smoke/tiny-overfit/small-validation/reload/resume without full B/E1/E2 or final test |
| Outputs used | `src/training/`, `src/evaluation/generated.py` and `small.py`, training/CPU/diagnostic configs, requirements/actual lock, official weight provenance, training tests, Colab execution path, Step 4 report/evidence/plot and updated project documentation |
| Agent verification | Official weight checksum; all eligible train/validation audio hashes/shape; real batch/label decoding and padding; finite gradients and actual encoder/decoder updates; tiny-subset loss/generated-WER comparison; separate tiny validation clean/fixed-noise metrics; exact new-process token/loss reload; two-step complete-state resume matched uninterrupted continuation; effective16 CPU pilot; pip check and all20 Python regressions passed; diagnostic plot visually inspected |
| Decisions after evidence | Used CPU fp32 because no CUDA device was available; kept original4×4 and added consistent1×16 CPU profiles. Reused Step 3 augmentation/scoring. Fixed EOS/cap flags after inspecting the custom generation API; used raw sequences and one total decoder bound. Retained preliminary runs and warnings. Host regression replaced the sandbox Gradio mount that stalled. |
| Student verification | Student must review/understand the code, source attribution, configuration, diagnostic interpretation and academic claims. No new human listening or feedback was supplied in Step 4. |
| Scope and integrity | All training was explicitly diagnostic. Poor tiny validation is reported alongside memorization. No full B/E1/E2, P0 benchmark, final-test inference/tuning, hosted ASR, application replacement, fabricated user feedback or five-user study. Colab/CUDA/mixed precision is not claimed as executed. Raw data/weights/checkpoints/caches/private microphone events stay outside Git. |
| Factual evidence | Counts, run IDs, logs, metric pairs, parameter updates, complete-resume comparison and compute estimates have a single summary in `results/step4/verification_summary.json` with a review snapshot in `docs/evidence/step4/`; source numbers are not illustrative. |

## Ongoing report fields for later work

For each later AI-assisted contribution, add date, tool/model, purpose, concise prompt summary, output used, exact files/code/report portions affected, how the student verified correctness, changes/rejections after review, and whether any claim or result needs independent evidence. Never use generated text as user feedback or experimental results.
