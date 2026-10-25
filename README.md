# DengerKelas

**Noise Augmentation and Transcript Stability for Local Real-Time Indonesian Lecture Captioning**

Local Indonesian captions for one nearby speaker, one microphone, and one laptop. The Step 2 app uses the **unmodified pretrained multilingual Whisper Tiny feasibility baseline**: an upstream CTranslate2 conversion of `openai/whisper-tiny`, executed locally with CPU INT8. Step 4 separately performed short full-model training diagnostics; these checkpoints are not installed in the app or presented as final models.

## Current status

**Step 2 feasibility: Pass within the tested conditions.** A 125-second browser-paced speech replay produced 123 text updates and eight buffer rollovers. Actual browser microphone sessions were logged, and the student confirmed live text, Stop/final flush, and Reset on the intended demonstration computer. See [findings and limits](docs/STEP2_STREAMING_FEASIBILITY.md).

**Step 3 data preparation: complete and verified.** Indonesian FLEURS official splits and the three assigned DEMAND environments are downloaded, audited and represented in reproducible manifests. See the [dataset card](docs/DATASET_CARD.md), [sources/licenses](docs/DATA_SOURCES_AND_LICENSES.md), [execution report](docs/STEP3_DATA_PREPARATION.md), and [preserved evidence](docs/evidence/step3/README.md). Manual listening review remains pending; semantic audio/label alignment is not established by numeric checks.

**Step 4 training pipeline: complete within CPU diagnostic scope.** Real batch/forward/backward/parameter updates, tiny-subset learning, generated validation metrics, checkpoint reload and full-state resume were executed. See the [training pipeline report](docs/STEP4_TRAINING_PIPELINE.md) and [actual evidence](docs/evidence/step4/README.md). The report contains measured compute estimates and the B/E1/E2 configurations. Colab/CUDA remain untested.

Full B/E1/E2 training, formal WER/CER/noise benchmarks, agreement-based stabilization and the five-user study remain later work. Neither course assignment is complete. Official assignment forms are still unavailable for direct verification; cross-course administrative questions remain open.

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

### Step 4 training diagnostics

The current `.venv` includes verified CPU PyTorch dependencies. Use `requirements/training-cpu.txt` for a fresh CPU installation; the actual Windows lock is `requirements/step4-windows-py312-cpu.lock.txt`. Existing data is reused. If training weights are missing, run `python -m src.training.assets` using the project interpreter.

```powershell
.\.venv\Scripts\python.exe -m src.training.verify_data
.\.venv\Scripts\python.exe -u -m src.training.run --config configs/training_smoke.json
.\.venv\Scripts\python.exe -u -m src.training.run --config configs/training_tiny_overfit.json
.\.venv\Scripts\python.exe -u -m src.training.reload
.\.venv\Scripts\python.exe -u -m src.evaluation.small
.\.venv\Scripts\python.exe -u -m src.training.run --config configs/training_cpu_pilot.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath results/step4/all_python_checks.txt
.\.venv\Scripts\python.exe -m src.training.report
```

Diagnostics use only train/validation. Full experiments require the explicit `--allow-experiment` flag and have not been run. The unexecuted Colab notebook in `notebooks/step4_colab.ipynb` calls the same modules. Checkpoints stay local under `checkpoints/`; source review evidence under `docs/evidence/step4/` excludes weights and optimizer tensors.

### Step 3 data pipeline

From this project root, in the existing Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/data.txt
.\.venv\Scripts\python.exe -m src.data.prepare --stage all
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_data_preparation.py -v
.\.venv\Scripts\python.exe -m src.data.documentation
```

The current environment already contains these packages. To recreate the full installed Step 3 environment, use `requirements/step3-windows-py312.lock.txt`. First acquisition needs Internet; verified files are reused. Source revisions, quality rules, SNR and seeds are pinned in `configs/data_sources.json` and `configs/data_preparation.json`. Audio, processors, model weights, caches and local results stay outside Git. This step does not run a model forward pass or fine-tuning.

### Step 2 streaming checks

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
- `src/data/`: resumable acquisition, safe extraction, audit, preprocessing, noise recipes, verification and factual documentation.
- `src/training/`: manifest adapter/collator, training, complete checkpoints/resume and diagnostics; generated metrics in `src/evaluation/generated.py`.
- `configs/`, `requirements/`, `tests/`: settings, verified dependencies and focused checks.
- `data/manifests/`: public provenance; recordings and model weights are excluded.
- `docs/`, `reports/`: planning, findings, AI log and separate course report outlines.

## Next checkpoint

Step 5: run the controlled B/E1/E2 experiments from the same pinned pretrained weights and consistent hardware/profile. Follow the Step 4 measured compute budget and validation-only selection policy. Keep the Step 2 training fixture out of independent final evaluation; final test remains locked. Tiny-overfit proves pipeline learning, not generalization.
