# Requirement-to-evidence matrix

**Source status:** The two official assignment forms were not found in the inspected workspace. Sections below refer to the supplied specification, not a direct form review. Verify exact wording, selected/struck options, tables, and formatting against the originals before treating section references as verified. Status is updated through Step 2; planning is not completion.

**Step 2 evidence:** Local pretrained Tiny streaming is implemented and tested with browser-paced speech and actual microphone sessions. The student confirmed the intended demonstration computer and Start/live updates/Stop/Reset. See `STEP2_STREAMING_FEASIBILITY.md`. Fine-tuning, formal metrics/noise experiments, final stabilization, reports and five-user testing remain outstanding.

## Deep Learning — COMP6826001

| Requirement | Source section/status | Planned implementation | Evidence needed | Stage | Status |
|---|---|---|---|---|---|
| Real-world problem, stakeholder, proposed solution, usage scenario | Supplied §3, Mandatory | Indonesian student lecture captioning; bounded one-speaker scenario | Validated user/problem evidence, brief, demo | Scope/user study | Planned |
| Relevant dataset: source, characteristics, examples/targets, distribution, quality, bias/limits | Supplied §3, Mandatory | FLEURS Indonesian `id_id`; verify actual metadata/counts and quality | Versioned manifest, dataset analysis, limitations | Data | Planned |
| Preparation and train/validation/test handling | Supplied §3, Mandatory | Preserve official FLEURS splits; document preparation and leakage controls | Split manifest, preprocessing record | Data | Planned |
| Construct a deep-learning model with modern framework | Supplied §3, Mandatory | Fine-tune multilingual Whisper tiny using PyTorch/Transformers | Configuration, code, checkpoint provenance, run logs | Training | Planned |
| Explain architecture, components, I/O, loss, optimizer, LR, batch, epochs, activations, rationale | Supplied §3, Mandatory | Explain Whisper seq2seq architecture and actual verified training configuration | Report technical section and config | Training/report | Planned |
| If pretrained: architecture, objective/data/domain, components, fine-tuning, suitability/limits | Supplied §3, Mandatory if applicable | Document checkpoint card/source and fine-tuning choices | Cited model documentation and report | Training/report | Planned |
| Baseline plus at least two additional conditions; variables and rationale | Supplied §3, Mandatory | B baseline, E1 LR, E2 augmentation; P0 reference | Run configs, logs, comparison table | Experiments | Planned |
| Task-appropriate evaluation | Supplied §3, Mandatory | WER primary, CER supplementary; clean/noisy and responsiveness | Scored evaluation outputs | Experiments | Planned |
| Training/validation loss and metric; analyze learning behavior/effects/errors/trade-offs | Supplied §3, Mandatory | Log loss and generated WER; curves and error analysis | Curves, logs, examples, analysis | Experiments/report | Planned |
| Practical suitability: robustness, speed, memory/model/hardware, scalability, limitations/deployment | Supplied §3, Mandatory | Measure local inference, compute, memory, model size and limits | Reproducible benchmark logs, deployment notes | Evaluation | Planned |
| Working application accepts input, runs constructed/trained model, displays output | Supplied §3, Mandatory | Local microphone caption application with selected fine-tuned checkpoint | Demonstration, screenshots, build/run instructions | Application | Partial: pretrained streaming works; trained checkpoint still required |
| Genuine practical evaluation | Supplied §3, Mandatory | Real target-user testing and technical tests | Protocol, observations, results | Evaluation | Planned |
| AI use disclosure, verification, reflection | Supplied §3 and §5, Mandatory | Maintain AI log; student verifies code/claims and reflects | Completed log and declarations in both reports | Throughout | In progress: Step 1/2 log entries; student review and final declaration pending |
| DL report structure: Introduction; Dataset/Data Preparation; DL Fundamentals/Model; System Design/Implementation; Experimental Methodology; Results; Analysis/Discussion; Real-World Evaluation; Conclusion; AI Usage Declaration | Supplied §3, Mandatory | Preserve headings in course-specific report | Report outline and final report | Reporting | Planned |
| AI Usage Log included in report | Supplied §3, Mandatory | Carry log into report | Log section/appendix | Reporting | Planned |
| Direct users recommended; alternative requires lecturer approval | Supplied §3, Recommended/conditional | Conduct direct testing as Speech Recognition requires | Five-user evidence | User study | Planned |

## Speech Recognition — COMP6822001

| Requirement | Source section/status | Planned implementation | Evidence needed | Stage | Status |
|---|---|---|---|---|---|
| Real problem, relevant target users | Supplied §4, Mandatory | BINUS Indonesian-speaking students; validate hypothesis | User protocol and findings | User study | Planned |
| Continuous microphone/audio input | Supplied §4, Mandatory | Local continuous capture, bounded buffer | Demo/screenshots/event logs | Streaming app | Verified within Step 2 tests; student microphone confirmation and local logs |
| Inference while speech is happening; intermediate updates before recording ends; continued updates | Supplied §4, Mandatory | Repeated overlapping-window Whisper inference with timestamp-aware merge | Live demo and timing logs | Streaming app | Verified within Step 2 tested durations; further quality evaluation pending |
| Model executes locally; explain/measure responsiveness | Supplied §4, Mandatory | On-device inference; no external ASR API | Architecture, network/data-flow statement, latency/RTF logs | App/evaluation | Verified CPU INT8 streaming; measured server events/DOM acknowledgements; visible word latency not measured |
| No external speech-recognition APIs | Supplied §4, Mandatory | Local model/library only | Architecture and runtime inspection | App | Verified local-file inference and process-level offline probe |
| Normal-condition baseline and changed condition | Supplied §4, Mandatory | Clean and controlled 15/5 dB SNR proposed conditions | Evaluation protocol/results | Experiments | Planned |
| WER and real-time responsiveness metric | Supplied §4, Mandatory | WER, first-display delay, settled delay, processing delay/RTF, backlog | Scored outputs and timing logs | Evaluation | Partial: Step 2 timing/load/backlog measured; formal WER and settled delay remain later |
| Transcription examples, error analysis, strengths/weaknesses/limits | Supplied §4, Mandatory | Representative examples and categorized errors | Report evidence | Analysis | Planned |
| At least five real target users; realistic tasks, observation, feedback instrument, report and connect to technical findings | Supplied §4, Mandatory | Five-user study with consented short tasks and questionnaire/interview | Protocol, anonymized records, summaries and analysis | User study | Planned |
| Screenshots, architecture diagram, experimental results, transcripts, WER, latency, user results/feedback/analysis | Supplied §4, Mandatory | Capture artifacts during actual implementation/evaluation | Report artifact set | App/evaluation/report | Partial: prototype diagram/screenshots/timing/example transcripts exist; formal experiments and five-user study pending |
| SR report: Introduction; Fundamentals; System Design; Implementation; Technical Experiment; User Testing; Discussion; Conclusion | Supplied §4, Mandatory | Separate course-specific report | Report outline/final report | Reporting | Planned |
| AI Usage Log and final AI Usage Declaration | Supplied §4/§5, Mandatory | Maintain log and declarations | Log/declaration in submission | Throughout | In progress: Step 1/2 log entries; student review and final declaration pending |

## Administrative unknowns

| Question | Current evidence/status | Action |
|---|---|---|
| Cross-course reuse allowed? | Not established by supplied summary | Confirm with both lecturers |
| Separate report/submission expectations? | One codebase plus two reports is a project decision, not confirmed | Confirm per course |
| Individual/group format and group size? | Unspecified | Confirm both courses |
| Exact deadline? | Submission follows exam timetable; department verification date 31 Aug 2026 is not a deadline | Check official timetable/lecturer |
| Packaging requirements? | Exam Apps submission stated; exact artifacts unclear | Confirm report/app/source packaging |
| Minimum dataset size, architecture, latency threshold, report length, public deployment? | Not established | Do not invent; confirm if needed |
| Alternative to five-user study? | Fewer than five requires lecturer approval; direct user study planned | Seek approval if recruitment makes five infeasible |
