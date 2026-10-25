# Environment inventory and verification

## Step 3 current environment — verified 2026-10-25

The same project `.venv` and Python 3.12.14 were reused. Step 3 added Transformers 5.20.2, tokenizers 0.23.3 and Matplotlib 3.11.2. NumPy 2.5.4, SciPy 1.18.1 and SoundFile 0.14.0 supply waveform reading/resampling, features and numeric checks. `pip check` passed. Actual full installed versions are recorded in `requirements/step3-windows-py312.lock.txt`; the direct preparation requirements are in `requirements/data.txt`.

Whisper Tiny's pinned processor/tokenizer/config loads locally and produces NumPy features without PyTorch. PyTorch and the training runtime have not been installed or verified. The warning that models are unavailable without PyTorch is expected for this stage; no model forward pass, fine-tuning or recognition scoring is claimed. Source archives are fetched directly with Python's standard library; no datasets/Arrow decoder is needed. The existing Step 2 app was not changed.

Executed data verification and rerun evidence: [Step 3 report](STEP3_DATA_PREPARATION.md), [evidence snapshot](evidence/step3/README.md). Raw data and processor files remain project-local and excluded from Git. The distribution figure was rendered using Matplotlib's noninteractive Agg backend and visually inspected. Listening review remains pending.

## Step 2 environment at feasibility checkpoint — verified 2026-10-25

The project `.venv` now runs **Python 3.12.14**. The old pip-less Python 3.14 environment is preserved as `.venv-step1-py314/`, excluded from Git. The available bundled Python 3.12 runtime created the new isolated environment with `--without-pip`, and its pip was used with `--python .venv/Scripts/python.exe` to install pip and deployment dependencies into that environment. No global packages were changed.

| Item | Actual observation |
|---|---|
| OS | Windows 11, AMD64, `Windows-11-10.0.26200-SP0` |
| CPU | Intel Core i5-13420H; 8 physical cores, 12 logical CPUs (registry/psutil) |
| Physical RAM | 16,857,817,088 bytes, detected by psutil |
| CTranslate2 GPU visibility | 0 CUDA devices; CPU compute supports INT8 |
| Deployment | faster-whisper 1.2.1, CTranslate2 4.8.3, Gradio 6.30.0 |
| Audio/numeric | PyAV 19.0.2, NumPy 2.5.4, SciPy 1.18.1, SoundFile 0.14.0 |
| Supporting packages | psutil 7.2.2, FastAPI 0.162.0, Uvicorn 0.54.0, websockets 17.3 |
| pip/dependencies | pip 26.2.2; `pip check` passed; Step 2 installed lock in `requirements/step2-windows-py312.lock.txt` |
| Model readiness | Local Tiny artifact loaded; Indonesian decoding completed using CPU INT8 |
| Offline check | Model load/decode succeeded with Python socket connections denied; no OS network settings changed |
| Demonstration machine | Student confirmed this is the intended computer and that microphone Start/live text/Stop/Reset passed |

Eleven Python checks and two browser-controller Node regression checks passed. A 125-second browser replay and actual microphone sessions support local streaming feasibility; detailed measurements and limitations are in `STEP2_STREAMING_FEASIBILITY.md`. Colab was not used or inspected. Training dependencies remain uninstalled/unverified.

Exact current launch from the DengerKelas root: `.\.venv\Scripts\python.exe -u -m app.main`, then open `http://127.0.0.1:7860`. The app sets a project-local Gradio cache and offline inference configuration itself. To recreate elsewhere, use a supported Python 3.12 venv and the installed lock. This computer's global `python` still points to 3.14.6; it is not the application interpreter.

For exact recovery with this computer's available bundled interpreter:

```powershell
& 'C:\Users\Muadz Arfan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv --without-pip .venv
& 'C:\Users\Muadz Arfan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip --python .venv\Scripts\python.exe install -r requirements/step2-windows-py312.lock.txt
```

Run recreation commands only when creating a fresh environment. The current environment already exists. The automation sandbox initially blocked external temp access/normal Gradio mounting; checks succeeded with project-local scratch files and the normal host process. This does not imply the computer lacks microphone or inference capability.

## Historical Step 1 observations

Captured 2026-10-25 (client date, Asia/Jakarta) in the supplied Windows workspace.

| Check | Observed result | Verification status |
|---|---|---|
| Operating system | Windows 11, platform string `Windows-11-10.0.26200-SP0` | Detected via Python `platform.platform()` |
| Architecture | AMD64 | Detected via Python `platform.machine()` |
| System Python | Python 3.14.6 at `C:\Users\Muadz Arfan\AppData\Local\Programs\Python\Python314\python.exe` | Detected; `py -0p` found no registered installs |
| Project virtual environment | `DengerKelas/.venv`, Python 3.14.6 | Created with `python -m venv --without-pip`; interpreter starts |
| pip in project environment | Not available | Standard venv failed in `ensurepip`; isolated venv without pip succeeded |
| Core imports | `torch`, `transformers`, `datasets`, `gradio`, `soundfile`, `librosa` absent in system Python | Checked with `importlib.util.find_spec`; no dependency versions are installed/claimed |
| GPU | Not determined; `nvidia-smi` did not provide usable output | Unverified |
| CPU, RAM, GPU model | CIM access denied/unavailable to this client | Unverified; no hardware is inferred |
| Intended demonstration computer | Not identified | Must inspect in Step 2 |
| Colab hardware | Not inspected or assumed | Unverified |

## Historical Step 1 blockers (runtime resolved in Step 2)

The regular venv creation failed because `ensurepip` returned a nonzero status; a pip-less isolated environment was created as a reversible foundation. Before installing frameworks, determine a Python version supported by the selected current PyTorch/Transformers versions, provision pip within the project environment, choose compatible version constraints, install the required group, then record `pip freeze`, essential imports, and framework GPU availability from that environment. Do not treat planned dependencies as installed or the environment as training-ready.

No assignment forms were found in the inspected workspace tree; direct requirement verification against the originals is pending.
