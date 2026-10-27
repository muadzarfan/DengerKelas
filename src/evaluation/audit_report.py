"""Generate one current status page and an isolated, factual audit snapshot."""
import re
import shutil
import subprocess
from pathlib import Path
from src.data.common import ROOT,read_json,write_json,digest


def build():
    folder=ROOT/'results/audit_20261027'
    integrity=read_json(folder/'integrity.json')
    resume=read_json(folder/'resume_verification.json')
    health=read_json(folder/'app_health.json')
    model=read_json(folder/'model_integrity.json')
    output=(folder/'python_checks.txt').read_text(encoding='utf-8-sig')
    match=re.search(r'Ran (\d+) tests',output)
    if not match or not output.rstrip().endswith('OK'):
        raise ValueError('A complete passing regression result is required')
    tests=int(match.group(1))
    pip=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','pip','check'],capture_output=True,text=True)
    (folder/'dependency_check.txt').write_text(pip.stdout+pip.stderr,encoding='utf-8')
    if pip.returncode:raise ValueError('Dependency check failed')
    if integrity['status']!='passed' or resume['status']!='passed' or model['status']!='passed':
        raise ValueError('An audit verification is not passing')
    if resume['fingerprint_schema']!=2 or not resume['full_state_bitwise_equal']:
        raise ValueError('Complete updated resume state must match')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    remote=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT).decode().strip()
    summary={'date':'2026-10-27','status':'passed','stage_reached':'Step 4 CPU pipeline verification',
             'next_stage':'Step 5 full controlled B/E1/E2; not started',
             'python_tests':tests,'data':integrity,'resume':resume,'app_health':health,
             'pinned_model_integrity':model,'pip_check':'passed',
             'new_diagnostic_updates':4,'full_experiments_run':False,'final_test_asr_run':False,
             'local_HEAD':head,'fetched_remote_main':remote,
             'remote_ignore_rules_preserved':['docs/','PromptDengarKelas.md'],
             'historical_step3_step4_evidence_modified':False,
             'pending':['Manual train/validation listening review','Full B/E1/E2 and final evaluation',
                        'Checkpoint deployment and agreement-based stabilization',
                        'Five real target users and final course reports','Lecturer administrative confirmation',
                        'GPU/Colab verification if a GPU route is chosen'],
             'fixes':['Resume locks ordered example IDs and evaluation/decoding settings',
                      'Resume preserves the evaluation grid and rejects experiment horizon changes',
                      'Noise/processor/pretrained/checkpoint checksums verified before use',
                      'Full NumPy RNG state and CUDA-state presence included in fingerprint schema 2',
                      'Diagnostic runs do not select a best project model',
                      'Computed diagnostic budget is bounded, including max_steps=0',
                      'Stale status/Colab wording corrected and local server restarted',
                      'Remote ignore update synchronized without undoing it']}
    write_json(folder/'summary.json',summary)
    n=lambda number:f'{number:,}'.replace(',','.')
    speech=integrity['speech']
    status=f'''# DengerKelas — status proyek

**Diperbarui: 27 Oktober 2026. Posisi: Step 4 selesai untuk verifikasi pipeline CPU; Step 5 belum dijalankan.**

Halaman ini menjadi pintu masuk progres. Dokumen rencana menjelaskan target akhir; bukti 25 Oktober adalah snapshot historis. Audit 27 Oktober disimpan terpisah agar angka diagnosis lama tidak tertukar dengan pemeriksaan baru.

## Sudah sampai mana?

| Bagian | Status | Yang benar-benar dilakukan |
|---|---|---|
| Step 1 — fondasi | Selesai secara teknis | Scope, brief, matriks persyaratan, rencana eksperimen, environment dan AI log |
| Step 2 — streaming lokal | Feasibility lulus dalam kondisi uji | Pretrained multilingual Tiny CPU INT8; capture bertahap, bounded buffer, update selama bicara, Stop/final flush, Reset; replay125detik dan konfirmasi mikrofon pengguna |
| Step 3 — data | Persiapan otomatis selesai | Unduhan resmi, audit seluruh WAV, manifest traceable, filter durasi/encoding, preprocessing, processor, noise/resep tetap dan rerun identik; listening masih pending |
| Step 4 — training pipeline | Diagnosis CPU terverifikasi | Batch, forward/backward, parameter update, tiny-overfit, generated WER, validation kecil, save/reload dan resume |
| Step 5 — B/E1/E2 | **Belum dimulai** | Config sudah siap; belum ada hasil eksperimen penuh atau checkpoint final |
| Deployment model terpilih, stabilisasi, studi pengguna | Belum selesai | App masih pretrained; agreement-based stabilization dan minimal5pengguna belum dilaksanakan |

**App saat ini:** [localhost:7860](http://127.0.0.1:7860/). Server sempat tidak berjalan meskipun tab terbuka; sudah direstart. Health mengembalikan `model_ready={str(health['model_ready']).lower()}`, CPU INT8. Ini bukan konfirmasi mikrofon manusia baru. Model hasil tiny-overfit tidak dipasang ke aplikasi.

## Data dan bukti yang dicek ulang

| Split | Seluruh rekaman | Eligible | Dikecualikan |
|---|---:|---:|---:|
| Train | {n(speech['train']['total'])} | {n(speech['train']['eligible'])} | {speech['train']['excluded']} |
| Validation | {n(speech['validation']['total'])} | {n(speech['validation']['eligible'])} | {speech['validation']['excluded']} |
| Test | {n(speech['test']['total'])} | {n(speech['test']['eligible'])} | {speech['test']['excluded']} |

Audit membaca kembali **{n(integrity['speech_integrity']['waveforms_decoded'])} WAV**, memeriksa hash/shape/sample rate/finite dan hubungan target asli. Semua **21 artefak sumber** lolos checksum resmi. OOFFICE=train, OHALLWAY=validation, PCAFETER=test; channel1. Test hanya diperiksa integritasnya, tanpa generation/tuning. Tidak ada data rusak atau perubahan label/manifests/aturan Step3 yang ditemukan.

Sebanyak **124 hash laporan historis** dan **25 hash source pada commit Step4** cocok. Raw audio, weights, checkpoint, environment dan cache tetap diabaikan Git. Rawdata tidak dipindah atau diunduh ulang.

Bukti diagnosis lama tetap valid pada lingkupnya: empat train,24update, loss1,820432→0,000666 dan generatedWER69,23%→0% pada contoh yang sama. Dua validation terpisah menghasilkan WER100%; ini tidak disembunyikan atau dipakai untuk klaim generalisasi. Bukan hasil B/E1/E2.

## Masalah yang ditemukan dan diperbaiki

| Temuan | Perbaikan |
|---|---|
| Resume signature belum mengunci subset/order dan aturan scoring/generation | Kunci ordered train/validation IDs, kind, device/precision, epoch, metric dan evaluasi; perubahan ditolak |
| Resume dari step di luar boundary dapat menggeser jadwal evaluasi | Kembali ke grid asli; misalnya160→318 untuk interval159 |
| Noise dan artefak model belum diverifikasi ulang tepat sebelum dipakai | Periksa hash/rate/channel/environment; checkpoint diverifikasi sebelum model/processor di-load; dukung combined processor config Transformers5 |
| Fingerprint NumPy hanya mencakup array internal | Schema2 mencakup posisi dan cache RNG juga; kompatibilitas checkpoint lama diperiksa lewat run_config yang checksummed |
| Diagnosis bisa menyimpan label best-validation yang membingungkan | Pemilihan best project model hanya untuk run eksperimen; metadata checkpoint historis dipertahankan |
| Guard diagnostic hanya melihat max_steps eksplisit | Periksa juga budget hasil perhitungan saat max_steps=0; maksimum64update |
| Dokumen/Colab memuat status lama, app berhenti | Tambahkan status utama, beri penanda snapshot historis, perbarui petunjuk Colab dan restart app |

Baseline dua tes resume memang gagal sebelum patch. Setelah perbaikan, **{tests} tes Python lulus**, `pip check` lulus. Empat update tambahan audit terdiri dari2referensi+2resume, memakai empat train lama, bukan eksperimen penuh. Reload di proses baru menghasilkan token sama; model, optimizer, scheduler, progress, Python/Torch/NumPy RNG lengkap cocok bit demi bit pada schema2. CUDA tidak tersedia sehingga tidak diklaim teruji.

## Urutan berikutnya

1. Lakukan listening review kecil train/validation dan tetap bawa batas domain/gender/encoding/lisensi dari [dataset card](docs/DATASET_CARD.md).
2. Jalankan Step5 dengan satu profil konsisten: B clean1e-5, E1 clean5e-6, E2 noise1e-5; semuanya mulai dari pretrained yang sama, bukan dari tiny-overfit/B sebelumnya.
3. Evaluasi/select memakai validation. Final test tetap dikunci sampai evaluasi akhir yang sudah ditetapkan.
4. Setelah model terpilih: validasi konversi/deployment lokal, uji kebijakan stabilisasi dan responsivitas, lalu minimal5targetpengguna nyata dan dua laporan mata kuliah.

Estimasi CPU25Oktober (1×accum16): 2,42–2,56jam training inti/run; skenario termasuk evaluasi3,36–3,79jam/run. Itu pilot kecil historis, bukan jaminan durasi saat kondisi komputer berubah. GPU/Colab belum diuji; notebook hanya jalur eksekusi. Ketentuan administratif dosen dan dokumen tugas asli masih perlu dikonfirmasi.

## Lokasi dan Git

- Panduan utama: [README](README.md), [rencana eksperimen](docs/EXPERIMENT_PLAN.md), [laporan Step4 historis](docs/STEP4_TRAINING_PIPELINE.md).
- Audit lokal saat ini: `results/audit_20261027/summary.json`, `integrity.json`, `python_checks.txt`, `resume_verification.json`; salinan lokal di `docs/evidence/audit_20261027/`.
- GitHub memiliki commit tambahan `2020553` (`update gitignore`), sudah disinkronkan secara fast-forward. Aturan `docs/` dan `PromptDengarKelas.md` dipertahankan; dokumen baru di docs tidak otomatis masuk Git. Status ini ditempatkan di root.
- Baseline lokal/remote saat audit: `{head[:7]}` / `{remote[:7]}`. Patch audit, dokumentasi, status ini dan snapshot audit dipilih untuk publikasi atas permintaan pengguna; commit setelah baseline mencatat publikasinya. Tidak ada full experiment atau hasil diagnosis lama yang ditimpa.

## Jalankan audit ulang

```powershell
.\\.venv\\Scripts\\python.exe -m src.evaluation.project_audit
.\\.venv\\Scripts\\python.exe -u -m src.training.audit_resume
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath results/audit_20261027/python_checks.txt
```

Audit resume menambah empat update diagnostik terisolasi dan checkpoint lokal. Perintah tersebut tidak menjalankan B/E1/E2 atau ASR final test. File checksum snapshot memisahkan source audit baru dari source historis Step4. Warning Gradio pada negative-path tests tetap tercatat; pesan deliberate missing-model dari tes tidak berarti app aktif gagal.
'''
    (ROOT/'PROJECT_STATUS.md').write_text(status,encoding='utf-8')
    target=ROOT/'docs/evidence/audit_20261027'; target.mkdir(parents=True,exist_ok=True)
    for path in folder.rglob('*'):
        if path.is_file() and 'test_tmp' not in path.parts and path.suffix in ('.json','.jsonl','.txt'):
            dest=target/path.relative_to(folder); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
    names=[p for base in (ROOT/'src/training',ROOT/'src/evaluation',ROOT/'tests') for p in base.glob('*.py')]
    names.append(ROOT/'PROJECT_STATUS.md')
    write_json(target/'snapshot_checksums.json',{
        'date':'2026-10-27','report_sha256':{p.relative_to(target).as_posix():digest(p) for p in target.rglob('*') if p.is_file() and p.name not in ('snapshot_checksums.json','README.md')},
        'audit_source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in names}})
    (target/'README.md').write_text('''# Audit 27 October 2026

Local evidence from results/audit_20261027, separate from immutable Step 3/4 snapshots.
summary.json is the entry point. baseline_regressions.txt preserves the two failures
that reproduced the original resume-contract gaps; python_checks.txt is the passing
post-fix suite. The schema2 recovery comparison includes the complete NumPy RNG.
No raw audio, model weights, optimizer tensors, cache or private microphone logs.
The repository's docs/ ignore rule is preserved. The user explicitly selected this
audit snapshot for publication; other ignored/private documents remain excluded.
PROJECT_STATUS.md is at repo root.
''',encoding='utf-8')
    print({'status':summary['status'],'tests':tests,'status_file':'PROJECT_STATUS.md'})


if __name__=='__main__':
    build()
