# Decisions, assumptions, and risks

## Step 3 decisions and actual findings

- Kept the existing project and Python environment. Acquired only official FLEURS `id_id` archives/TSV and the three specified DEMAND 16 kHz environments, pinned in `configs/data_sources.json`. Download/checksum retries recovered observed failures. All requested artifacts are verified and reused on rerun.
- Preserved every original audio and transcript and the official splits. Unique recording IDs combine source split and filename; repeated source sentence IDs are not speaker IDs. No exact audio or repeated scoring transcript crosses splits in this audit. Speaker independence cannot be established from available metadata.
- Reused already-mono 16 kHz speech. Excluded over-30-second inputs without trimming full transcripts. After train/validation inspection, v2 also holds suspected mojibake pending human review rather than guessing labels. Counts and reasons have one factual source in `DATASET_CARD.md` and the statistics JSON; initial v1 evidence remains local.
- Used only NFC/whitespace changes for training targets, separate symmetric scoring normalization, and a pinned Indonesian/transcription processor. Whole-dataset padded features and mixed WAV collections were not generated.
- Assigned all channels from each noise environment to one split; selected channel 1 consistently. Implemented whole-utterance RMS-SNR, deterministic per-record/epoch seeds and shared anti-clipping gain. Clean baseline audio remains unchanged. Fixed validation/test definitions and a test manifest/config lock are stored.
- Test was used only for integrity audit and fixed definitions. No test feature smoke, listening selection, ASR inference, model choice or tuning occurred. Train/validation real mixes and features were checked.
- Completed full data verification and a rerun with identical JSONL hashes. No training, B/E1/E2 recognition experiment, WER, checkpoint selection, major streaming change or five-user study occurred.
- Remaining limitations: no manual listening/semantic alignment review; read-speech versus spontaneous-class domain gap; validation/test metadata are all MALE; suspicious encoding held for review; DEMAND metadata says CC BY 4.0 while description/PDF says CC BY-SA 3.0. Preserve both notices and attribution rather than inventing a resolved license.
- See `DATASET_CARD.md`, `DATA_SOURCES_AND_LICENSES.md`, `STEP3_DATA_PREPARATION.md` and `evidence/step3/` for measured evidence. Data is ready to build/verify the Step 4 training pipeline; PyTorch/training runtime still needs its own verification.

## Step 2 decisions and verified changes

- At the Step 2 checkpoint, the pretrained-only feasibility baseline used the upstream CTranslate2 conversion of multilingual Whisper Tiny with faster-whisper/CPU INT8. No project fine-tuning or full dataset workflow had begun at that checkpoint.
- Repaired `.venv` with the available Python 3.12.14 runtime and installed/pinned deployment dependencies; preserved Step 1's Python 3.14 environment. Full installed lock and imports are verified. Training remains separate.
- Gradio custom HTML with a Web Audio/AudioWorklet PCM bridge supplies explicit incremental chunks and an ordered Stop marker. One background decoder, no inference-job queue, 18-second context/three-second overlap, 24-second raw-audio bound and 300-second sessions.
- History is frozen by audio-window age. It is not settled/confident text or the final agreement policy. Recognition/timestamp errors remain.
- Bounded browser replay and real microphone use were verified on the detected i5-13420H machine. The student confirmed this as the demonstration computer and confirmed Start/live updates/Stop/Reset. No numeric assignment latency threshold is implied.
- A failed rapid restart was recorded and corrected by isolating old async AudioContext cleanup/status from new capture setup; consecutive sessions were rechecked. Keep failure evidence rather than deleting it.
- Microphone waveform files are not written. Timing logs may contain transcript text and stay local/ignored; export public replay logs or redacted numeric microphone summaries only.
- See `STEP2_STREAMING_FEASIBILITY.md` for actual observations, configuration and limits. No conceptual scope change was made.

## Project decisions

- Rename KelasTeks to **DengerKelas**; preserve the previously agreed problem, architecture, research direction, target users, and bounded scope.
- Share one technical foundation across both courses provisionally, with separate course-specific reports/submissions pending lecturer confirmation.
- Primary model: multilingual `openai/whisper-tiny`, fine-tuned with PyTorch and Hugging Face Transformers. Train the ASR model itself.
- Primary speech dataset: FLEURS Indonesian `id_id`, retaining official splits. Step 3 verified DEMAND environments: OOFFICE train, OHALLWAY validation, PCAFETER final test.
- Local Gradio application, live microphone, mono 16 kHz resampling, bounded overlap buffer, local inference, timestamp-aware merge, provisional/settled regions and timing logs.
- Use agreement-based stabilization as an evaluated display policy, not a novel algorithm or confidence/correctness guarantee.
- Direct study with at least five target users; separate independent practical recordings from training/tuning.
- No external ASR API. No model download, data acquisition, training, or app implementation in Step 1.

## Assumptions and open risks

| Item | Status / risk | Mitigation or next action |
|---|---|---|
| Official assignment forms | Not present in inspected workspace; source section mapping is from pasted summary only | Obtain both `.docx` forms and verify all tables/selected and struck-out options |
| Cross-course reuse | Administrative permission unknown | Ask both lecturers; keep reports distinct meanwhile |
| Group rules, deadlines, packaging | Not fully specified | Confirm individually with each course; do not mistake 31 Aug 2026 verification date for deadline |
| Python/runtime compatibility | Step 2 deployment verified with Python 3.12.14 and pinned installed stack | Verify training framework compatibility separately before training |
| Venv pip bootstrap | Step 1 failure resolved with environment-targeted pip and a Python 3.12 venv | Use the documented interpreter and installed lock |
| Hardware/GPU access | CPU/RAM detected; CTranslate2 sees zero CUDA devices; CPU INT8 verified | Other hardware/GPU modes need their own check |
| Streaming responsiveness | Bounded streaming verified in Step 2; transient lag/recognition errors remain | Broaden natural speech/noise evaluation and measure final trained-checkpoint performance |
| Model/data suitability | Indonesian lecture domain and accent/noise coverage uncertain | Inspect sources/metadata, test held-out and user audio, report bias and limits |
| Noise data/licensing | Official files/channel verified; metadata and description/PDF license notices conflict | Preserve both notices and attribution; consult source notices before distributing derivative audio |
| Dataset quality and representation | Mojibake/long rows held out; clipping flagged; validation/test metadata all MALE; listening pending | Keep raw data/reasons, perform train/validation listening review and independent target-user evaluation; do not claim speaker independence |
| Participant recordings | Consent, privacy, and storage details need an approved protocol | Confirm course/ethics expectations, consent, retention and anonymization before collecting |
| Fewer than five users | Would require lecturer approval under supplied summary | Recruit five or request approval before substituting |
| Stabilization trade-off | Added delay may outweigh lower visual churn | Matched ablation and user feedback; report measured trade-off |
| Export runtime | faster-whisper/CTranslate2 compatibility/quality not verified | Optional; compare against selected checkpoint before use |

## Lecturer confirmation checklist (not sent)

- [ ] Is one shared technical project/codebase permitted to satisfy COMP6826001 and COMP6822001?
- [ ] Are two distinct course reports/submissions required, and what should each contain?
- [ ] Is the project individual or group work; if group, what group size and contribution evidence apply?
- [ ] What are the exact deadlines and Exam Apps packaging requirements for each course?
- [ ] Confirm that the proposed five-user testing protocol and consent approach meet course expectations; what approval is needed if recruitment falls below five?
