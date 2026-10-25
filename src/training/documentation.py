"""Render the Step 4 report from executed diagnostic evidence."""
from pathlib import Path
from src.data.common import ROOT


def render(s,smoke,overfit,pilot,small):
    env=s['hardware']; learning=s['overfit']; validation=s['tiny_validation']; compute=s['compute_estimate']
    cases='\n'.join(f"| {k} | {v['loss']:.6f} | {100*v['wer']:.2f}% | {v['examples']} |" for k,v in small['conditions'].items())
    checks='\n'.join(f'- `{k}`: {v}' for k,v in s['completion_checks'].items())
    status='Selesai' if s['status']=='complete' else 'Parsial'
    doc=fr'''# Step 4 — training pipeline dan verifikasi

Status: **{status}**. Ini verifikasi pipeline dengan diagnosis singkat, bukan eksperimen penuh B/E1/E2, benchmark final, atau model siap deployment. Sumber angka utama: `results/step4/verification_summary.json`; salinan bukti untuk review ada di [evidence/step4](evidence/step4/README.md). Semua angka/grafik berasal dari eksekusi aktual.

## Lingkungan dan sumber yang digunakan

- {env['os']}; Python 3.12.14; CPU **{env['cpu']}**, {env['physical_cores']} physical/{env['logical_cores']} logical cores; RAM {env['ram_bytes']:,} byte.
- PyTorch **{env['versions']['torch']}**, Transformers **{env['versions']['transformers']}**; CPU fp32, 4 thread. `torch.cuda.is_available()` adalah **{env['cuda_available']}**, CUDA build `{env['torch_cuda_build']}`, daftar GPU CUDA `{env['gpus']}`. Intel UHD terdeteksi dari Windows, tidak dipakai sebagai CUDA. CUDA/bf16/fp16 dan Colab belum dijalankan.
- Environment `.venv` Step 2–3 dipakai kembali; `pip check` lulus. Lock aktual: `requirements/step4-windows-py312-cpu.lock.txt`. Requirements portable training dan CPU dipisahkan. Tidak ada perubahan package global.
- FLEURS `id_id` dan DEMAND memakai lokasi/revisi/aturan Step 3. Semua **2.538 train + 345 validation eligible** diperiksa kembali terhadap file, hash, sample rate/channel dan sample count. Tidak ada unduhan ulang dataset. Test audio/features tidak dibuka. Keterbatasan dataset/listening/lisensi tetap berlaku sebagaimana [dataset card](DATASET_CARD.md).
- Bobot PyTorch `openai/whisper-tiny` multilingual revisi `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` diunduh dari sumber resmi: **151.061.672 byte**, SHA-256 `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`. Provenance: `data/manifests/step4_model.json`. Bobot CTranslate2 Step 2 terpisah dan tetap digunakan aplikasi.

## Arsitektur, transfer learning dan kontrak model

Audio mono 16 kHz → log-Mel 80 channel, padding 30 detik → dua convolution/GELU, stride total 2 → encoder Transformer bidirectional 4 layer → decoder Transformer causal 4 layer dengan cross-attention → proyeksi vocabulary multilingual. Hidden size 384, 6 attention heads per blok. Parameter aktual **{s['model']['total']:,}**, trainable **{s['model']['trainable']:,}**, frozen **0**. Semua parameter termasuk embedding posisi encoder dilatih secara eksplisit sesuai keputusan full-model; tidak memakai adapter/LoRA.

Transfer learning selalu dimulai dari bobot pretrained yang sama untuk run baru. B tidak menjadi initialization E1/E2. Loss adalah teacher-forced token cross-entropy implementasi model; label padding diabaikan dengan -100. Optimizer AdamW; matriks diberi weight decay, bias/vektor normalisasi tidak. Gradient accumulation memakai total loss token valid dibagi jumlah token valid seluruh grup, termasuk grup terakhir yang parsial.

`language=id`, `task=transcribe`, no-timestamps. Prefix processor `[50258,50275,50359,50363]`. Collator membuang tepat satu decoder-start; model menggeser label dengan decoder-start sekali. EOS tetap dihitung loss; pad dengan ID yang sama tidak keliru dimask sebagai EOS karena masking memakai attention mask tokenizer.

Batch nyata dua contoh: fitur **(2,80,3000) float32**, attention mask **(2,3000) int64**, label **(2,20) int64**, enam padding token -100. Semua finite, label tidak kosong, hasil decode sama dengan target. Audio tidak dipotong sambil mempertahankan label penuh. Fitur dibuat on-the-fly; cache LRU dibatasi delapan contoh, bukan seluruh korpus.

Generation tidak menerima label atau target prompt. Satu generation config memuat language/task, legacy forced_decoder_ids=None, greedy/beam 1, tanpa sampling/timestamps. Satu batas total decoder length digunakan: prefix 4 ditambah budget token, paling banyak 448. Diagnosis memakai budget 96, pilot 160, eksperimen resmi 444. Raw decoder sequences mengandung prefix/EOS untuk pemeriksaan cap/reload; teks tetap di-decode dengan skip special tokens. Train `use_cache=False`, generation `use_cache=True`. Gradient checkpointing tersedia tetapi nonaktif pada pilot CPU; bila aktif, train cache tetap nonaktif.

## Konfigurasi yang disiapkan

| Run | LR | Audio train | Status |
|---|---:|---|---|
| P0 | tidak dilatih | pretrained asli | referensi tambahan direncanakan; bukan benchmark Step 4 |
| B | 1e-5 | clean | config siap, eksperimen penuh belum dijalankan |
| E1 | 5e-6 | clean | hanya LR berubah dari B |
| E2 | 1e-5 | clean + noise | hanya augmentation berubah dari B |

`configs/training_B.json` + inheritance E1/E2: batch fisik 4, accumulation 4, effective 16, maksimum 4 epoch, seed 20261025. Precision auto memilih hanya hardware CUDA yang terdeteksi; CPU menggunakan fp32. Profil CPU terpisah `training_cpu_B/E1/E2.json` memakai batch 1 × accumulation 16, effective tetap 16. Gunakan profil yang sama untuk semua kondisi; jangan membandingkan run berbeda perangkat/profil tanpa disclosure.

Keputusan tambahan yang sama untuk ketiganya: AdamW eps1e-8, weight decay0.01, linear decay dengan warmup5%, gradient clipping1.0, evaluasi setiap159 optimizer steps (satu epoch saat ini), pemilihan berdasarkan **clean validation generated corpus WER**, tie mempertahankan checkpoint lebih awal. Train metric adalah subset tetap32 yang dipilih dengan hash ID/seed, bukan seluruh train. Validation seluruh345 pada clean/OHALLWAY15/5dB. Tidak ada early stopping eksperimen; maksimum4epoch. Simpan best dan latest, retention2, hanya checkpoint lengkap dalam direktori run miliknya yang dapat dihapus. Checkpoint diagnostik tidak ikut dipilih sebagai model final.

E2 memanggil fungsi recipe/materialize Step 3, hanya OOFFICE; seed berdasarkan ID/condition/epoch. Integrasi membuktikan label/panjang/SR sama, fitur reproducible, SNR benar dan sumber/fitur clean tidak berubah. Validation fixed-noise memakai resep Step 3 persis. PCAFETER/test tidak diteruskan ke loader training/evaluasi Step 4.

## Diagnosis yang benar-benar dijalankan

1. **Smoke:** `{smoke['run_id']}`, empat contoh train tersedia, dua dipakai dalam satu optimizer update melalui dua micro-batch fisik1; dua contoh validation terpisah. Loss finite, gradient finite tersedia pada **{s['smoke']['gradient_tensors_checked']} tensor parameter**, **{s['smoke']['changed_parameter_tensors']} tensor berubah**; encoder/decoder sama-sama berubah. LR1e-5, tanpa augmentation.
2. **Tiny-overfit:** `{overfit['run_id']}`, empat contoh train pendek dengan ID konten berbeda, dipilih berdasarkan durasi/ID sebelum melihat model output. Batch1, accumulation1, LR1e-4 khusus diagnosis, AdamW constant scheduler, tanpa augmentation. Budget48 updates; berhenti pada **{learning['optimizer_steps']} updates** karena kriteria diagnosis train terpenuhi. Satu pass atas4contoh bukan epoch korpus penuh.
3. **Pilot compute:** `{pilot['run_id']}`,16contoh train dipilih dengan hash ID/seed; dua optimizer updates,32micro-batch/presentasi contoh, CPU batch1×accum16. Tidak menjalankan B.
4. **Continuation/resume:** dua langkah referensi tanpa interupsi setelah checkpoint tiny-overfit dan dua langkah resume dalam proses baru, hanya untuk membandingkan state.

Tiny-overfit pada **empat contoh yang sama**, bukan generalisasi:

| Ukuran | Sebelum | Sesudah |
|---|---:|---:|
| Teacher-forced loss subset | {learning['initial_loss']:.9f} | {learning['final_loss']:.9f} |
| Generated corpus WER subset | {100*learning['initial_generated_wer']:.2f}% | {100*learning['final_generated_wer']:.2f}% |

Validation kecil setelah overfit: **{validation['examples']} contoh**, loss **{validation['loss']:.6f}**, generated WER **{100*validation['generated_wer']:.2f}%**. Ini membuktikan fungsi evaluasi berjalan dan memperlihatkan memorisasi yang tidak memberikan hasil baik pada dua contoh terpisah; tidak mewakili seluruh validation.

Evaluator mandiri pada checkpoint diagnosis yang sama:

| Kondisi validation | Loss | Generated WER | Contoh |
|---|---:|---:|---:|
{cases}

Ini wiring check kecil, **bukan clean/noisy benchmark proyek**. Reference, raw prediction, normalized reference/prediction, ID, checkpoint, jumlah error S/D/I dan cap flag tersimpan. WER adalah jumlah edit seluruh korpus dibagi jumlah kata referensi, dengan normalisasi Step 3 identik di kedua sisi; bukan rata-rata WER per contoh dan bukan argmax logits teacher forcing.

Seluruh pekerjaan aktual, termasuk dua run pengembangan awal yang tetap disimpan: **{s['actual_diagnostic_work']['all_recorded_optimizer_updates']} optimizer updates**, **{s['actual_diagnostic_work']['all_recorded_micro_batches']} micro-batch**, **{s['actual_diagnostic_work']['all_recorded_training_example_presentations']} presentasi contoh**. Evidence utama mencakup {s['actual_diagnostic_work']['primary_evidence_updates']} updates; sisanya run awal. Jumlah ini tidak boleh diberi label epoch/eksperimen korpus penuh.

![Grafik diagnosis aktual](evidence/step4/diagnostics.png)

## Save, reload dan resume

Checkpoint utama: `{Path(learning['checkpoint']).relative_to(ROOT).as_posix()}`. Direktori berisi model.safetensors, processor/tokenizer, model/generation config, run config, environment, identitas data, checksum artefak dan training_state.pt. Semua checkpoint besar lokal/ignored.

Proses baru memuat model dan processor dari direktori lokal tersebut: generated token IDs, referensi, loss dan WER **identik**. Resume mengembalikan optimizer AdamW, scheduler, scaler, global/micro step, epoch, cursor micro-batch serta Python/NumPy/Torch RNG. Dua update melanjutkan **step24→26**, bukan reset. Fingerprint model, optimizer, scheduler, progress dan RNG yang diperiksa sama dengan kelanjutan tanpa interupsi. Sampler order diturunkan dari seed+epoch; cursor disimpan di batas optimizer group. Resume menolak identitas data, batch/precision/LR dan horizon scheduler yang tidak cocok. Memuat weights saja tidak disebut resume.

`training_state.pt` memuat objek Python/NumPy dari run lokal yang dipercaya; jangan memuat state dari sumber tak dipercaya. Untuk aplikasi nanti, gunakan format Transformers safetensors + processor + kontrak generation tersebut, lalu konversi/validasi CTranslate2 dalam tahap deployment. Checkpoint diagnosis tidak dipasang ke aplikasi Step 2.

## Estimasi Step 5 dari pilot aktual

- Waktu per optimizer update effective16: **{min(compute['core_training_hours_per_run'])*3600/636:.3f}–{max(compute['core_training_hours_per_run'])*3600/636:.3f} detik**; maksimum RSS yang diamati saat update pilot **{s['pilot']['max_observed_rss_bytes']:,} byte**. Ini sampled RSS proses, bukan pengukuran peak seluruh sistem.
- 2.538 contoh, 4 epoch, batch 1 × accumulation 16 menghasilkan 159 updates/epoch, **636 updates/run**. Extrapolasi training inti: **{compute['core_training_hours_per_run'][0]:.2f}–{compute['core_training_hours_per_run'][1]:.2f} jam/run**, tiga run **{compute['core_training_hours_three_runs'][0]:.2f}–{compute['core_training_hours_three_runs'][1]:.2f} jam**.
- Jika kecepatan generation kecil yang diamati bertahan,32train metric +345×3validation =1.067contoh/evaluasi, empat evaluasi menambah biaya. Skenario kasar gabungan sekitar **{compute['illustrative_total_hours_per_run'][0]:.2f}–{compute['illustrative_total_hours_per_run'][1]:.2f} jam/run** sebelum variasi tambahan. Ini bukan janji waktu: panjang token, noise, thermal throttling, proses lain, cache/I/O dan penyimpanan checkpoint dapat memperlambatnya. Step pilot pertama beririsan dengan pemeriksaan data ringan; step kedua lebih mendekati kondisi stabil. Tidak ada estimasi GPU yang diklaim terukur.
- Colab GPU dapat menjadi jalur Step5, tetapi notebook `notebooks/step4_colab.ipynb` **belum dijalankan**. Notebook memakai modul proyek yang sama dan penyimpanan Drive untuk recovery, memerlukan pilihan GPU/login oleh pengguna jika dipakai. Tidak diperlukan tindakan akun untuk menyelesaikan verifikasi CPU Step4 ini.

## Menjalankan ulang

Dari root DengerKelas, environment saat ini sudah terpasang:

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

Instalasi CPU baru: `python -m pip install -r requirements/training-cpu.txt`, memakai Python3.12; bobot yang hilang: `python -m src.training.assets`. Jangan mengganti interpreter global. Nama run unik memakai timestampUTC. Full run memerlukan `--allow-experiment`; **belum dijalankan**. Contoh Step5 CPU: `python -m src.training.run --config configs/training_cpu_B.json --allow-experiment`; gunakan E1/E2 dengan config masing-masing dan initialization pretrained baru. Resume interupsi: tambahkan `--resume checkpoints/<run>/checkpoint-<step>` dengan konfigurasi dan horizon asli, target step lebih besar daripada progres tersimpan.

## Masalah, batas, dan kesiapan

Run smoke awal salah menandai EOS/cap karena output tensor Whisper menghapus EOS. Diperbaiki dengan raw decoder sequences dan run ulang. Konflik default panjang generation dihilangkan dengan satu batas total. Warning deprecation suppression-token berasal dari pemanggilan internal Transformers versi terpasang; language/task proyek tetap eksplisit dan output/reload terverifikasi. Dua run awal tetap tersedia dalam evidence. Regression transport Gradio sempat stall dalam sandbox; dihentikan dan diulang pada host normal. Semua **{s['python_regression_checks']['count']} Python tests lulus**, termasuk Step2/Step3; warning unclosed event-loop dari Gradio pada failure-path tests tetap tercatat.

Tidak ada training B/E1/E2 penuh, final-test evaluation/tuning, hosted ASR, penggantian prototype streaming, lima pengguna, atau kesimpulan kualitas. Manual listening dan keterbatasan representasi/lisensi Step3 tetap terbuka. Data/model/pipeline lokal **siap masuk Step5 untuk eksperimen terkontrol**, dengan profil hardware konsisten, waktu/compute di atas dan verifikasi CUDA tersendiri bila pindah runtime.

Kriteria yang diperiksa:

{checks}

Referensi implementasi resmi: [Whisper Transformers](https://huggingface.co/docs/transformers/model_doc/whisper), [PyTorch installation](https://pytorch.org/get-started/locally/), serta source Python versi terpasang yang diperiksa untuk forward/generation, label shifting dan cross-entropy. Tidak ada metrik ilustratif.
'''
    (ROOT/'docs/STEP4_TRAINING_PIPELINE.md').write_text(doc,encoding='utf-8')
