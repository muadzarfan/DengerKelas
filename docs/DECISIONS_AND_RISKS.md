# Decisions, assumptions, and risks

## Step 2 decisions and verified changes

- Step 2 is a pretrained-only feasibility baseline, using the upstream CTranslate2 conversion of multilingual Whisper Tiny with faster-whisper/CPU INT8. No project fine-tuning or dataset workflow has begun.
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
- Primary speech dataset: FLEURS Indonesian `id_id`, retaining official splits. Proposed DEMAND environments: OOFFICE train, OHALLWAY validation, PCAFETER final test.
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
| Noise data/licensing | DEMAND file availability, channel selection, terms not yet checked | Verify official source and license before acquisition; retain attribution |
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
