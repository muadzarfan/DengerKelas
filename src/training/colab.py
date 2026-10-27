"""Generate a Colab notebook using the same project modules, not a second trainer."""
import json
from src.data.common import ROOT


def write_notebook():
    def markdown(text):
        return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
    def code(text):
        return {'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'outputs':[],'execution_count':None}
    cells=[markdown('''# DengerKelas Step 4 — Colab execution path

**Not executed in Colab.** Local CPU diagnostics are documented separately. Select a GPU runtime and verify it below. This notebook calls the repository's exact Python modules. No final test evaluation or full B/E1/E2 experiment is launched by this notebook.

Mount Drive only when you authorize it. Use a checkout or source bundle containing `src/training/`. Step 4 was published at commit `def24cf`; use the latest reviewed source when including later audit fixes. Preserve `checkpoints/` and `results/step4/` on Drive for runtime recovery. Keep raw audio/weights out of Git.
'''),code('''import os, sys, subprocess
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
# Put the current Step 4 source bundle under this folder before continuing.
PROJECT = Path('/content/drive/MyDrive/DengerKelas')
assert (PROJECT/'src/training/run.py').is_file(), 'Copy/unzip the current Step 4 source bundle into PROJECT'
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/training.txt'], check=True)
'''),code('''import torch
from src.training.config import environment
print(environment())
assert torch.cuda.is_available(), 'Enable a GPU runtime; do not assume it is available'
'''),markdown('''## Reuse verified data or acquire only missing official artifacts

Place the existing `data/` tree here to avoid another download. Otherwise the same Step 3 acquisition verifies and downloads the official Indonesian sources. The first run may need about 12 GB free temporary headroom; see the Step 3 report. Do not upload private microphone recordings.
'''),code('''subprocess.run([sys.executable, '-m', 'src.data.prepare', '--stage', 'all'], check=True)
subprocess.run([sys.executable, '-m', 'src.training.assets'], check=True)
subprocess.run([sys.executable, '-m', 'src.training.verify_data'], check=True)
'''),markdown('''## GPU diagnostic using the same runner

This is a new hardware/precision diagnostic, not the measured Windows CPU run. GPU fp32 first isolates runtime differences; mixed precision can be verified in a later diagnostic. Its results must be recorded before treating Colab as verified.
'''),code('''import json
cfg = json.loads((PROJECT/'configs/training_tiny_overfit.json').read_text())
cfg.update(run_id='diagnostic_colab_tiny_overfit', device='cuda', precision='fp32')
# Keep the file beside its inherited config so relative extends remains valid.
target = PROJECT/'configs/training_colab_diagnostic.local.json'
target.write_text(json.dumps(cfg, indent=2))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
subprocess.run([sys.executable, '-u', '-m', 'src.training.run', '--config', str(target)], check=True)
subprocess.run([sys.executable, '-u', '-m', 'src.training.reload'], check=True)
'''),markdown('''## Recover after interruption

Use the last complete checkpoint (not an `.incomplete` directory). The original scheduler horizon, data hashes, batch/accumulation, precision and other resume-sensitive settings must match. `training_state.pt` includes trusted local optimizer/scheduler/scaler/RNG/cursor state; do not load this file from untrusted sources.
'''),code('''# Example only: replace with an existing complete checkpoint after an interruption.
# checkpoint = PROJECT/'checkpoints/<run-id>/checkpoint-000012'
# subprocess.run([sys.executable, '-u', '-m', 'src.training.run',
#                 '--config', str(target), '--resume', str(checkpoint)], check=True)
''')]
    for i,cell in enumerate(cells):
        cell['id']=f'step4-{i:02d}'
    folder=ROOT/'notebooks'; folder.mkdir(exist_ok=True)
    path=folder/'step4_colab.ipynb'
    path.write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5},indent=2),encoding='utf-8')
    return path


if __name__=='__main__':
    print(write_notebook())
