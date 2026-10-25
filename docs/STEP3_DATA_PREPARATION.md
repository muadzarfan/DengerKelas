# Step 3 — persiapan data DengerKelas

Status: **Selesai**. Satu sumber angka utama adalah `results/step3/speech_statistics.json`, dirender dalam [dataset card](DATASET_CARD.md); daftar revisi/checksum ada di [sumber dan lisensi](DATA_SOURCES_AND_LICENSES.md). Salinan bukti ringkas untuk review tersedia di [evidence/step3](evidence/step3/README.md), diekspor dari laporan aktual saat dokumentasi dirender. Tidak ada training, eksperimen B/E1/E2, ASR scoring, pemilihan checkpoint, atau pengujian lima pengguna pada tahap ini.

## Cara menjalankan ulang

Dari root DengerKelas, gunakan interpreter proyek Python 3.12.14:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/data.txt
.\.venv\Scripts\python.exe -m src.data.prepare --stage all
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_data_preparation.py -v
.\.venv\Scripts\python.exe -m src.data.documentation
```

`--stage` juga menerima acquire, extract, audit, recipes, verify. Instalasi pertama/berkas yang hilang perlu Internet; rerun semua berkas terverifikasi menggunakan data lokal. `configs/data_sources.json` mengunci sumber, `configs/data_preparation.json` mengunci aturan/seed. Lock dependensi aktual: `requirements/step3-windows-py312.lock.txt`.

## Pelaksanaan dan penyimpanan

1. Pemeriksaan disk awal: 250,882,379,776 byte kosong. Estimasi konservatif termasuk unduhan, ekstraksi dan cache: 11,678,289,776 byte.
2. Unduhan aktual: 2,651,136,988 byte. File .part, Range resume, tiga percobaan terbatas dan checksum mencegah berkas parsial dianggap selesai. Unduhan terverifikasi dilewati saat rerun.
3. Ekstraksi hanya anggota WAV yang terdaftar di TSV, dengan path aman, receipt/hash, dan kecocokan nama/jumlah. Tiga channel noise konsisten diekstraksi; 15 channel lain per arsip tidak diduplikasi ke disk.
4. Seluruh 3616 speech dibaca untuk audit; 3561 masuk subset kompatibel. Original panjang/encoding mencurigakan tetap ada dengan alasan pengecualian; tidak ada waveform yang dipotong sambil mempertahankan label penuh.

Ruang data aktual:

- `data/raw`: 5,584,480,354 byte (5.584 GB desimal).
- `data/processed`: 0 byte (0.000 GB desimal).
- `data/manifests`: 11,422,638 byte (0.011 GB desimal).
- `data/checkpoints/whisper-tiny-processor`: 4,372,752 byte (0.004 GB desimal).

Total kelompok di atas: 5,600,275,744 byte; tidak termasuk runtime lama/venv/cache dependensi. Data raw mencakup arsip yang dipertahankan untuk checksum/reproduksi dan WAV sumber yang benar-benar dipakai, bukan dua salinan audio hasil preprocessing. Tidak ada fitur berpadded seluruh dataset atau kombinasi noisy WAV yang disimpan. Semua berada di root workspace DengerKelas, terutama `data/raw/`, `data/manifests/` dan `data/checkpoints/whisper-tiny-processor/`.

## Audit dan keputusan kualitas

Audit awal v1 hanya menandai potensi encoding. Inspeksi train/validation menemukan pola mojibake pada nama seperti `SÃ£o`; v2 mengubahnya menjadi pengecualian pending review dengan pola yang tetap, tanpa menebak label benar. Laporan awal disimpan di `results/step3/initial_v1/`. Tidak ada keputusan yang berasal dari hasil pengenalan test; semua pengecekan test terbatas integritas dan kebijakan kualitas yang sama.

Rekaman yang memiliki potensi clipping dipertahankan dengan flag. Transkrip berulang dengan audio berbeda dipertahankan. Speaker ID tidak tersedia, sehingga tidak ada speaker-independence claim. Distribusi gender dan durasi/token ada di dataset card serta `results/step3/distributions.png`. **Listening review manusia belum dilakukan** dan audit ini tidak membuktikan seluruh label cocok secara semantik.

## Augmentation dan test lock

E2: probabilitas noise 0.5, SNR uniform 5.0–20.0 dB, hanya OOFFICE. Resep epoch 0 aktual: 1270 clean dan 1268 noisy; epoch berikutnya memakai seed turunan dari ID/condition/epoch/global seed. Baseline B/E1 tetap memakai audio clean yang sama.

Definisi SNR adalah rasio mean-square seluruh utterance, termasuk jeda. Noise dipotong sesuai durasi/offset, atau diulang siklis bila perlu. Alpha = sqrt(Pspeech / (Pnoise × 10^(SNR/10))). Jika peak campuran >0.99, kedua komponen diberi gain bersama sehingga SNR tidak rusak. Speech energi nol tetap clean dengan alasan; noise energi nol ditolak. Panjang, sample rate dan target tetap.

Validation memakai OHALLWAY, test memakai PCAFETER; masing-masing memiliki resep tetap 15/5 dB dengan seed, offset sample, channel, hash dan parameter. Jumlah resep validation/test adalah 690/1356. Tidak ada campuran test yang dipakai untuk pengembangan/feature smoke. `test_lock.json` mengunci hash manifest, resep dan aturan. `prepare_example` menolak test kecuali evaluasi final meminta akses secara eksplisit.

## Verifikasi aktual

- 23,349 pemeriksaan integritas/manifest/resep/fitur lulus; gagal 0.
- Lima focused unit checks lulus: Unicode/scoring, kutip literal TSV/resampling stereo, reproducible seed dan role split, real RMS SNR/shared clipping gain, dan energi nol/wrap.
- Enam contoh train/validation menghasilkan fitur float32 (80,3000), label roundtrip yang sama, prefix Indonesia/transcription benar, padding -100 dan valid EOS tidak hilang; empat campuran nyata train/validation mempertahankan label dan SNR 5/15 dB.
- Rerun seluruh pipeline selesai dalam 51.172 detik. Semua 7 manifest JSONL memiliki hash identik: `True`; output diganti atomik, bukan ditambah/append.
- `pip check` lulus. PyTorch belum dipasang; processor/feature smoke memakai NumPy dan tidak menjalankan model forward.

## Masalah yang benar-benar ditemukan

- Endpoint viewer FLEURS sebelumnya gagal pada Step 2; Step 3 memakai arsip resmi yang tersedia pada revisi yang sama.
- Satu request DEMAND.pdf memperoleh HTTP 502 dan berhasil setelah retry. OOFFICE/PCAFETER sempat gagal pemeriksaan integritas; retry/resume menghasilkan file yang akhirnya cocok checksum resmi. Tidak ada file gagal yang dilabeli terverifikasi.
- Parsing CSV bertanda kutip biasa mengubah bentuk beberapa baris; QUOTE_NONE diverifikasi menjaga ketujuh kolom dan tanda kutip sumber.
- Perbedaan lisensi DEMAND serta bias gender tetap dinyatakan sebagai temuan, bukan disembunyikan.

## Kesiapan Step 4

Data dan preprocessing siap untuk **membangun serta memverifikasi training pipeline**, dengan subset dan keterbatasan di atas. Step 4 masih perlu framework training/PyTorch yang sesuai, batching/dataloader/loss smoke dan rencana runtime; belum ada fine-tuning atau hasil eksperimen. Tidak ada tindakan akun/unduhan manual pengguna yang diperlukan untuk menyelesaikan persiapan otomatis ini. Penggunaan lintas mata kuliah dan lima pengguna nyata tetap merupakan pekerjaan terpisah.
