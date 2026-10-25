# Decisions, assumptions, and risks

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
| Python/runtime compatibility | Detected Python 3.14.6 may not be supported by chosen ML stack; packages absent | Establish supported Python version from official current package documentation during Step 2, then install pinned compatible dependencies in isolated env |
| Venv pip bootstrap | Standard `python -m venv` failed at `ensurepip`; `--without-pip` succeeded | Repair/install pip in the isolated environment using approved mechanism before dependency installation |
| Hardware/GPU access | CIM and `nvidia-smi` unavailable; GPU and physical memory unknown | Inspect intended demo machine directly and run local inference benchmark |
| Streaming responsiveness | Whisper is not natively streaming; repeated-window strategy can add latency/revisions/backlog | Step 2 proof of continuous local mic transcription; bound buffers and measure backlog |
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
