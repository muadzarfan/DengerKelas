# DengerKelas

**Formal title:** Noise Augmentation and Transcript Stability for Local Real-Time Indonesian Lecture Captioning

DengerKelas is a planned local laptop application for live Indonesian captions from one nearby speaker. Its research evaluates supervised fine-tuning and noise robustness for multilingual Whisper tiny, plus a display that separates provisional words from settled text. “Settled” means agreement under the selected policy; it does not imply correctness or calibrated confidence.

## Status

**Step 1 foundation complete; implementation and evaluation have not started.** Assignment forms were not found in the inspected workspace. The supplied project specification is the current planning source; direct verification against the two official forms remains pending. No experiments, user sessions, dataset statistics, or results are claimed.

## Structure

- `app/`, `src/`, `training/`, `streaming/`, `evaluation/`: reserved for later implementation stages.
- `configs/`, `data/manifests/`: small tracked configuration and metadata only.
- `data/checkpoints/`: local model checkpoints, excluded from Git.
- `results/`: later logs and measurements; no results yet.
- `docs/`: scope, requirement mapping, experiment plan, environment, decisions, AI log.
- `reports/`: separate Deep Learning and Speech Recognition report workspaces and user study material.
- `requirements/`: core, training, and deployment dependency lists.

## Setup

Python 3.14.6 is detected. A project-local `.venv` was created without pip because the normal venv bootstrap failed while running `ensurepip`. The environment is isolated but currently has no project dependencies. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

Use a Python version supported by the selected PyTorch/Transformers releases when installing later; select and verify compatible versions before training. Intended dependency groups are listed in `requirements/`. Do not install into the global interpreter. Once pip is available in the venv, install only the needed group and record actual resolved versions.

## Verification

The isolated interpreter starts and reports Python 3.14.6. Core ASR/training/deployment imports are absent, and GPU availability is unverified. No model download or inference check is part of Step 1.

## Next step

**Step 2 checkpoint: prove continuous local microphone transcription on the intended demonstration computer.** Confirm the computer and microphone, establish a supported runtime and local inference path, and demonstrate partial transcript updates while speech is ongoing before proceeding to model training and the full caption interface.

