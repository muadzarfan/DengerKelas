# DengerKelas

**Noise Augmentation and Transcript Stability for Local Real-Time Indonesian Lecture Captioning**

Local Indonesian captions for one nearby speaker, one microphone, and one laptop. Step 2 uses the **unmodified pretrained multilingual Whisper Tiny feasibility baseline**: an upstream CTranslate2 conversion of `openai/whisper-tiny`, executed locally with CPU INT8. No project fine-tuning has occurred.

## Current status

**Step 2 feasibility: Pass within the tested conditions.** A 125-second browser-paced speech replay produced 123 text updates and eight buffer rollovers. Actual browser microphone sessions were logged, and the student confirmed live text, Stop/final flush, and Reset on the intended demonstration computer. See [findings and limits](docs/STEP2_STREAMING_FEASIBILITY.md).

Training, full dataset preparation, formal WER/CER/noise experiments, agreement-based stabilization and the five-user study remain later work. Neither course assignment is complete. Official assignment forms are still unavailable for direct verification; cross-course administrative questions remain open.

## Launch on this computer

In PowerShell:

```powershell
Set-Location -LiteralPath 'C:\Users\Muadz Arfan\Documents\Codex\2026-10-25\files-pasted-by-the-user-step\DengerKelas'
.\.venv\Scripts\python.exe -u -m app.main
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Click **Mulai mikrofon**, allow access, and speak Indonesian. **Stop dan selesaikan** sends the last packet and flushes recognition. **Reset sesi** cancels the session and clears text/audio before another session. Ctrl+C stops the server. One active session is supported; the server binds only to loopback.

The display has history and a changing active hypothesis. History is retained by the audio-window boundary, not agreement or confidence. The final provisional/settled feature remains deferred.

## Environment and reproducibility

The existing `.venv` now uses Python 3.12.14 and the installed deployment stack; the pip-less Step 1 environment is preserved in `.venv-step1-py314/`. See [environment details](docs/ENVIRONMENT.md).

For a new supported **Python 3.12** environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/step2-windows-py312.lock.txt
.\.venv\Scripts\python.exe -m src.evaluation.download_assets --model --fixture
.\.venv\Scripts\python.exe -m app.main
```

The recreation example's `python` must point to Python 3.12; this computer's global command currently points to 3.14.6. The current venv is already installed. Artifact download needs Internet. Inference requires only local files and uses `local_files_only=True` and `HF_HUB_OFFLINE=1`.

Edit `configs/streaming.json` to configure model/runtime/streaming. Environment overrides: `DENGERKELAS_MODEL_PATH`, `DENGERKELAS_DEVICE`, `DENGERKELAS_COMPUTE_TYPE`, or `DENGERKELAS_CONFIG`. A later compatible converted fine-tuned model can replace the configured folder.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.evaluation.offline_probe
.\.venv\Scripts\python.exe -m src.evaluation.replay
.\.venv\Scripts\python.exe -m src.evaluation.replay --seconds 125
```

If Node.js is available, run `node --test tests/client_lifecycle.test.cjs` for the two focused browser lifecycle regressions. Node is not needed to launch the application.

The development panel also offers short and sustained browser replays. These use one public FLEURS training clip, repeated with one-second pauses for the sustained fixture, and are feasibility checks rather than independent recognition benchmarks. Local event logs/summaries are under `results/step2/`. Microphone logs may contain private text and remain excluded from Git.

## Structure

- `app/`: Gradio UI, Web Audio capture and FastAPI WebSocket transport.
- `src/streaming/`: stateful resampling, local runtime, bounded audio, one worker and timestamp merge.
- `src/evaluation/`: provenance, paced replay, summaries, offline probe and inventory.
- `configs/`, `requirements/`, `tests/`: settings, verified dependencies and focused checks.
- `data/manifests/`: public provenance; recordings and model weights are excluded.
- `docs/`, `reports/`: planning, findings, AI log and separate course report outlines.

## Next checkpoint

Step 3: verify/acquire speech and noise sources, inspect metadata/licenses and official splits, and implement reproducible preprocessing. Keep this already-used training fixture out of independent final evaluation. Step 2 did not start training or the full dataset workflow.
