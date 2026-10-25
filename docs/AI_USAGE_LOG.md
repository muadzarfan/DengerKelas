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

## Ongoing report fields

For each later AI-assisted contribution, add date, tool/model, purpose, concise prompt summary, output used, exact files/code/report portions affected, how the student verified correctness, changes/rejections after review, and whether any claim or result needs independent evidence. Never use generated text as user feedback or experimental results.
