# Project brief

## Identity and scope

Current implementation status is centralized in [PROJECT_STATUS.md](../PROJECT_STATUS.md). This brief describes the intended final system. Step 2 already implements the pretrained streaming prototype; Step 3 prepared data; Step 4 verified diagnostic training. Agreement-based stabilization, full experiments and user evaluation remain pending.

- **Name:** DengerKelas (renamed from KelasTeks; the agreed concept is otherwise preserved).
- **Formal title:** Noise Augmentation and Transcript Stability for Local Real-Time Indonesian Lecture Captioning.
- **Purpose:** Investigate whether noise augmentation improves a compact Indonesian ASR model on unseen background noise, and whether agreement-based stabilization improves reading experience enough to justify added display delay.
- **Problem hypothesis:** Students may miss spoken explanations while listening and taking notes; background noise and frequently changing captions may make following the explanation harder. Validate this with actual target users; it is not an established finding.
- **Target users:** Indonesian-speaking BINUS students who follow spoken course explanations and can participate in testing.
- **Bounded scenario:** One nearby speaker gives a short Indonesian explanation while a student follows live captions on a laptop. One microphone, one speaker at a time, short sessions.

## Research questions

1. Does noise augmentation improve a compact Indonesian ASR model under previously unseen background noise, and what happens to clean-speech performance?
2. Does agreement-based caption stabilization improve reading experience enough to justify its added display delay?

This is an engineering and empirical evaluation. Fine-tuning, augmentation, and agreement stabilization are not claimed as newly invented algorithms.

## Planned architecture

Local microphone → mono 16 kHz resampling → bounded overlapping audio buffer → local multilingual `openai/whisper-tiny` inference → timestamp-aware hypothesis merge → provisional and settled caption areas. Fine-tuning uses PyTorch and Hugging Face Transformers on Indonesian `google/fleurs` (`id_id`) training data. DEMAND environments are proposed for augmentation and noise evaluation. A local Gradio UI is planned. A faster-whisper/CTranslate2 export is optional and requires compatibility and quality checks.

Whisper does not inherently provide streaming; an explicit repeated-window/streaming strategy must be implemented and credited to [whisper_streaming](https://github.com/ufal/whisper_streaming). External hosted speech recognition APIs are prohibited. Training may use Colab if necessary; assessed inference must run locally on the demonstration computer.

## Expected I/O

- **Input:** live audio from one microphone, resampled correctly to mono 16 kHz; short single-speaker Indonesian speech.
- **Output:** interim/provisional words that update as speech arrives, then stable/settled text after successive updates agree; timestamps and event timing for evaluation. Stability is not a correctness or calibrated-confidence claim.

## Final MVP acceptance criteria (partially achieved; see current status)

- Continuous local microphone capture and inference while the speaker is talking, with updates before speech ends.
- Bounded overlapping buffer and timestamp-aware merging without systematic omissions/duplication in observed examples.
- Visible separation between provisional and settled text; behavior and settling policy documented.
- Runs on the intended demonstration laptop without sending audio to hosted inference.
- Logs first display, settled-text delay, processing/backlog, and resource measurements for evaluation.
- Supports reproducible evaluation against the selected trained checkpoint; a pretrained-only demo does not satisfy the DL work.
- User evaluation with at least five real target users, realistic tasks, observation, and recorded feedback, subject to consent and course requirements.

## Final-version differentiator

Stable live captions: newly recognized words occupy a provisional area and move into a settled area when successive recognition updates agree. Report the stability policy and its latency/reading-experience trade-off.

## Exclusions

Initial scope excludes diarization, translation, summarization, chatbots, mobile applications, accounts, databases, and production cloud hosting. No claim of broad classroom accessibility or educational outcome is made without evidence.

## Assumptions and validation needs

One shared technical foundation may serve both courses, with distinct reports/submissions, only if lecturers confirm. FLEURS, DEMAND suitability/licenses, local hardware feasibility, recording consent, and technical settings must be checked in their respective later stages. Confirm the actual demonstration computer and whether its hardware supports real-time local inference. No numeric latency acceptance threshold is specified by the supplied assignment summary; any engineering target must be labeled provisional.
