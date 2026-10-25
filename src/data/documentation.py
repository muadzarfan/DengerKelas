"""Render factual dataset documents from the verified machine-readable reports."""
import json
from .common import ROOT,read_json,read_jsonl,event


def build_documents():
    cfg=read_json(ROOT/'configs/data_preparation.json')
    sources=read_json(ROOT/'configs/data_sources.json')
    stats=read_json(ROOT/'results/step3/speech_statistics.json')
    leakage=read_json(ROOT/'results/step3/leakage_audit.json')
    verify=read_json(ROOT/'results/step3/verification.json')
    rerun=read_json(ROOT/'results/step3/rerun_verification.json')
    recipes=read_json(ROOT/'results/step3/recipe_statistics.json')
    downloads=read_json(ROOT/'data/manifests/step3_downloads.json')
    noise=read_jsonl(ROOT/'data/manifests/demand_noise.jsonl')
    status='Selesai' if verify['status']=='passed' and rerun['same_manifest_bytes'] else 'Parsial'
    table='| Split resmi | Awal | Jam awal | Kompatibel | Jam kompatibel | Dikecualikan unik |\n|---|---:|---:|---:|---:|---:|\n'
    for split,s in stats.items():
        table+=f"| {split} | {s['before_count']:,} | {s['before_duration_seconds']/3600:.6f} | {s['after_count']:,} | {s['after_duration_seconds']/3600:.6f} | {s['before_count']-s['after_count']} |\n"
    reasons='\n'.join(f"- {split}: {json.dumps(s['exclusion_reasons'],ensure_ascii=False)}" for split,s in stats.items())
    flags='\n'.join(f"- {split}: {json.dumps(s['flag_counts'],ensure_ascii=False)}" for split,s in stats.items())
    gender='\n'.join(f"- {split}: {json.dumps(s['source_gender'])}; {s['unique_source_sentence_ids']} ID kalimat untuk {s['before_count']} nama file unik." for split,s in stats.items())
    noise_table='| Peran | Lingkungan | Channel | Sample rate | Sample aktual | Detik aktual |\n|---|---|---:|---:|---:|---:|\n'
    for n in noise:
        noise_table+=f"| {n['split']} | {n['environment']} | {n['channel']} | {n['sample_rate']} | {n['num_samples']} | {n['duration_seconds']:.3f} |\n"
    card=f'''# Dataset card — DengerKelas

Status Step 3: **{status}**, konfigurasi `{cfg['pipeline_version']}`. Dokumen faktual ini dirender oleh `python -m src.data.documentation`. Sumber angka utama: `results/step3/speech_statistics.json`; dokumen lain merujuk ke sini agar statistik tidak berbeda.

## Identitas dan cakupan

Speech: [google/fleurs](https://huggingface.co/datasets/google/fleurs), konfigurasi `id_id`, revisi `{sources['fleurs']['revision']}`. Source `train/dev/test` dipetakan ke split resmi `train/validation/test`. Dataset adalah read speech bahasa Indonesia dengan referensi teks, bukan rekaman kelas spontan. Label task adalah transkripsi urutan token.

{table}
Angka dihitung dari seluruh WAV yang diunduh dan dibaca, bukan perkiraan lintas bahasa. Durasi presisi detik sebelum/sesudah tersedia di JSON sumber. Semua baris dan audio asli dipertahankan. `eligible` menandai subset yang lolos aturan kualitas/encoding dan batas masukan tunggal <=30 detik; contoh panjang dapat diperiksa dalam evaluasi streaming terpisah nanti tanpa pemotongan label sembarangan.

## Filtering dan kualitas

Alasan pengecualian per split:

{reasons}

Hitungan alasan dapat tumpang tindih; satu rekaman train memiliki dua alasan. Pola mojibake dipilih setelah inspeksi train/validation (misalnya `SÃ£o`): pasangan karakter C2/C3 ala Latin-1 diikuti U+0080–U+00BF dikecualikan sampai diverifikasi manusia. Tidak ada perbaikan nama/label yang ditebak. Seluruh raw text dan target tetap tersimpan. Aturan sama diterapkan ke audit integritas test; tidak ada inferensi atau tuning test.

Flag lain dan flag encoding yang tercatat:

{flags}

Clipping adalah indikasi numerik, tidak otomatis membuktikan audio rusak, sehingga tidak menjadi alasan penghapusan. Semua contoh telah diperiksa keterbacaan, SR/channel, NaN/Inf, kosong/energi nol, peak/RMS, durasi, panjang teks dan token. Listening review manusia belum dilakukan; audit otomatis dan token roundtrip tidak membuktikan kesesuaian semantik setiap audio dengan label.

## Identitas dan leakage

Recording ID: `fleurs:id_id:<source_split>:<filename_stem>`. Kolom sumber `id` berulang dan tidak dipakai sebagai recording ID atau speaker ID. Hash file dan decoded PCM disimpan; seluruh split resmi tetap dapat ditelusuri.

- Grup audio identik decoded PCM: {len(leakage['exact_audio_duplicate_groups'])}; antarsplit: {len(leakage['exact_audio_cross_split_groups'])}.
- Grup transkrip scoring berulang: {len(leakage['repeated_scoring_transcript_groups'])}; antarsplit: {len(leakage['repeated_scoring_transcript_cross_split_groups'])}.
- ID sumber yang muncul lebih dari sekali: {leakage['source_sentence_ids_reused']}.

Teks berulang dengan waveform berbeda dipertahankan; ini dapat berupa take/pembicara berbeda tetapi tidak membuktikan identitas pembicara. Metadata speaker ID tidak tersedia. Tidak ada klaim speaker independence.

Distribusi gender sebagaimana metadata sumber (bukan identitas yang ditebak):

{gender}

Validation/test berlabel MALE seluruhnya: ini keterbatasan representasi penting. Jangan mengacak ulang split resmi untuk menyembunyikannya; evaluasi independen target pengguna tetap diperlukan.

Distribusi seluruh baris resmi sebelum filtering (batas 30 detik dan batas decoder ditandai):

![Distribusi durasi dan token referensi FLEURS Indonesia](evidence/step3/distributions.png)

## Preprocessing dan fitur

Seluruh speech sumber sudah mono 16 kHz dan dipakai kembali; tidak ada salinan clean yang tidak perlu. `read_audio` menyediakan konversi stereo dan resample_poly sungguhan jika diperlukan untuk sumber lain. Tidak ada denoising/gain normalization agresif. Target training hanya Unicode NFC dan collapse whitespace; raw text serta normalized source text dipertahankan.

Aturan scoring terpisah: Unicode NFKC, casefold, tanda baca Unicode menjadi spasi, collapse whitespace; angka, kata Indonesia/nama/istilah Inggris tetap dipertahankan. Terapkan aturan yang sama ke referensi dan prediksi. Belum ada WER/CER atau inferensi model.

Processor/tokenizer multilingual Whisper Tiny revisi `{sources['processor']['revision']}` menghasilkan `float32 (80,3000)` per contoh, prefix `<|startoftranscript|> <|id|> <|transcribe|> <|notimestamps|>`, EOS asli dipertahankan, padding label -100, dan decoder-start dibuang satu kali pada batch label. Batas decoder aktual 448 token; fitur dihitung on-the-fly, tidak disimpan seluruhnya sebagai cache berpadded.

## Noise dan penggunaan yang sesuai

{noise_table}
Hanya channel 1 diekstraksi; semua 16 channel setiap arsip tercatat berada dalam peran lingkungan yang sama. Noise tidak memiliki target transkripsi utama: target tetap speech FLEURS. Campuran adalah kondisi terkontrol, bukan bukti rekaman kelas nyata.

Data ini mendukung baseline B/E1 clean serta E2 noisy dan clean/noisy evaluation terkontrol. Read speech, kalimat berulang, bias gender, domain/aksen, dan background noise dari lingkungan berbeda membatasi generalisasi ke kelas BINUS. Dataset publik tidak menggantikan minimal lima pengguna nyata. Lihat [sumber/lisensi](DATA_SOURCES_AND_LICENSES.md) dan [prosedur/hasil](STEP3_DATA_PREPARATION.md).
'''
    archive_table='| Artefak resmi | Byte terunduh | Checksum resmi |\n|---|---:|---|\n'
    for d in downloads['files']:
        if d['name'].endswith(('.tar.gz','.zip')):
            c=d.get('official_checksum',{})
            archive_table+=f"| `{d['name']}` | {d['actual_bytes']} | {c.get('algorithm')}: `{c.get('value')}` |\n"
    licensing=f'''# Sumber data dan lisensi

## FLEURS

Sumber resmi: [Google FLEURS dataset card/repository](https://huggingface.co/datasets/google/fleurs). Konfigurasi `id_id`, revisi `{sources['fleurs']['revision']}`; lisensi dataset card `cc-by-4.0`. Atribusi proyek: Google/FLEURS, dataset card dan publikasi *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech* yang ditautkan sumber, dengan revisi dan URL tersebut. Lisensi [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) dan atribusi dipertahankan dalam manifest. Salinan README resmi ada di `data/raw/metadata/FLEURS_README.md`.

Repositori saat ini menyediakan TSV dan arsip WAV per bahasa. Pipeline mengunduh hanya Indonesia melalui URL resolve yang dipin revisinya, bukan loader script lama, keseluruhan korpus multibahasa, atau Parquet bahasa lain. Kolom TSV aktual: ID konten, filename, raw_transcription, normalized transcription, grapheme/uroman text, num_samples, gender. Parsing memakai QUOTE_NONE agar kutip literal tidak hilang. Statistik card khusus `id_id` dicocokkan dengan metadata dan audio aktual.

## DEMAND

Sumber resmi: [Zenodo record 1227121](https://zenodo.org/records/1227121), DOI [10.5281/zenodo.1227121](https://doi.org/10.5281/zenodo.1227121), versi {sources['demand']['version']}. Atribusi: Joachim Thiemann, Nobutaka Ito, Emmanuel Vincent, *DEMAND: a collection of multi-channel recordings of acoustic noise in diverse environments* (2013).

**Perbedaan sumber lisensi dipertahankan:** API metadata Zenodo menyebut `{sources['demand']['metadata_license']['id']}`; deskripsi record serta dokumen DEMAND.pdf menyebut **CC BY-SA 3.0 Unported** ([teks lisensi](https://creativecommons.org/licenses/by-sa/3.0/)). Metadata dan dokumen asli disimpan lokal; pipeline tidak mengklaim perbedaan ini sudah diselesaikan. Pertahankan atribusi dan pemberitahuan ShareAlike dokumen ketika merencanakan distribusi audio turunan; jangan mengganti lisensi dengan klaim tunggal yang mengabaikan konflik. Tidak ada audio mentah/campuran yang dipublikasikan ke Git.

Hanya OOFFICE_16k.zip, OHALLWAY_16k.zip dan PCAFETER_16k.zip serta DEMAND.pdf yang diunduh. Arsip masing-masing memiliki 16 WAV mono; hanya `ch01.wav` dipakai. Tidak ditemukan file lisensi terpisah dalam ketiga arsip yang dipilih, sebagaimana `results/step3/extraction.json`; PDF dan deskripsi merupakan bukti pemberitahuan lisensi yang ditemukan.

## Processor

Sumber [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny), revisi `{sources['processor']['revision']}`, dataset/model card berlabel Apache-2.0. Hanya processor/tokenizer/config yang diunduh untuk Step 3; bobot training baru tidak diunduh. Runtime inferensi Step 2 dan manifestnya tetap terpisah.

## Integritas unduhan

{archive_table}
Semua {len(downloads['files'])} file yang diminta telah lolos ukuran/checksum resmi. Berkas LFS memakai SHA-256, metadata kecil memakai Git blob SHA-1 sumber, dan DEMAND memakai MD5 resmi serta SHA-256 lokal. Daftar URL, ukuran, revisi dan checksum lokal lengkap: `data/manifests/step3_downloads.json`. Semua sumber publik, tanpa autentikasi atau API berbayar. Kegagalan unduhan dan retry tercatat di `results/step3/acquisition.jsonl`.
'''
    storage_table='\n'.join(f"- `{k}`: {v:,} byte ({v/1e9:.3f} GB desimal)." for k,v in verify['storage_bytes'].items())
    procedure=f'''# Step 3 — persiapan data DengerKelas

Status: **{status}**. Satu sumber angka utama adalah `results/step3/speech_statistics.json`, dirender dalam [dataset card](DATASET_CARD.md); daftar revisi/checksum ada di [sumber dan lisensi](DATA_SOURCES_AND_LICENSES.md). Salinan bukti ringkas untuk review tersedia di [evidence/step3](evidence/step3/README.md), diekspor dari laporan aktual saat dokumentasi dirender. Tidak ada training, eksperimen B/E1/E2, ASR scoring, pemilihan checkpoint, atau pengujian lima pengguna pada tahap ini.

## Cara menjalankan ulang

Dari root DengerKelas, gunakan interpreter proyek Python 3.12.14:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements/data.txt
.\\.venv\\Scripts\\python.exe -m src.data.prepare --stage all
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_data_preparation.py -v
.\\.venv\\Scripts\\python.exe -m src.data.documentation
```

`--stage` juga menerima acquire, extract, audit, recipes, verify. Instalasi pertama/berkas yang hilang perlu Internet; rerun semua berkas terverifikasi menggunakan data lokal. `configs/data_sources.json` mengunci sumber, `configs/data_preparation.json` mengunci aturan/seed. Lock dependensi aktual: `requirements/step3-windows-py312.lock.txt`.

## Pelaksanaan dan penyimpanan

1. Pemeriksaan disk awal: {sources['storage_preflight']['free_bytes']:,} byte kosong. Estimasi konservatif termasuk unduhan, ekstraksi dan cache: {sources['storage_preflight']['conservative_required_bytes']:,} byte.
2. Unduhan aktual: {sum(d['actual_bytes'] for d in downloads['files']):,} byte. File .part, Range resume, tiga percobaan terbatas dan checksum mencegah berkas parsial dianggap selesai. Unduhan terverifikasi dilewati saat rerun.
3. Ekstraksi hanya anggota WAV yang terdaftar di TSV, dengan path aman, receipt/hash, dan kecocokan nama/jumlah. Tiga channel noise konsisten diekstraksi; 15 channel lain per arsip tidak diduplikasi ke disk.
4. Seluruh {sum(s['before_count'] for s in stats.values())} speech dibaca untuk audit; {sum(s['after_count'] for s in stats.values())} masuk subset kompatibel. Original panjang/encoding mencurigakan tetap ada dengan alasan pengecualian; tidak ada waveform yang dipotong sambil mempertahankan label penuh.

Ruang data aktual:

{storage_table}

Total kelompok di atas: {sum(verify['storage_bytes'].values()):,} byte; tidak termasuk runtime lama/venv/cache dependensi. Data raw mencakup arsip yang dipertahankan untuk checksum/reproduksi dan WAV sumber yang benar-benar dipakai, bukan dua salinan audio hasil preprocessing. Tidak ada fitur berpadded seluruh dataset atau kombinasi noisy WAV yang disimpan. Semua berada di root workspace DengerKelas, terutama `data/raw/`, `data/manifests/` dan `data/checkpoints/whisper-tiny-processor/`.

## Audit dan keputusan kualitas

Audit awal v1 hanya menandai potensi encoding. Inspeksi train/validation menemukan pola mojibake pada nama seperti `SÃ£o`; v2 mengubahnya menjadi pengecualian pending review dengan pola yang tetap, tanpa menebak label benar. Laporan awal disimpan di `results/step3/initial_v1/`. Tidak ada keputusan yang berasal dari hasil pengenalan test; semua pengecekan test terbatas integritas dan kebijakan kualitas yang sama.

Rekaman yang memiliki potensi clipping dipertahankan dengan flag. Transkrip berulang dengan audio berbeda dipertahankan. Speaker ID tidak tersedia, sehingga tidak ada speaker-independence claim. Distribusi gender dan durasi/token ada di dataset card serta `results/step3/distributions.png`. **Listening review manusia belum dilakukan** dan audit ini tidak membuktikan seluruh label cocok secara semantik.

## Augmentation dan test lock

E2: probabilitas noise {cfg['augmentation']['noise_probability']}, SNR uniform {cfg['augmentation']['train_snr_db_min']}–{cfg['augmentation']['train_snr_db_max']} dB, hanya OOFFICE. Resep epoch 0 aktual: {recipes['train']['clean']} clean dan {recipes['train']['noisy']} noisy; epoch berikutnya memakai seed turunan dari ID/condition/epoch/global seed. Baseline B/E1 tetap memakai audio clean yang sama.

Definisi SNR adalah rasio mean-square seluruh utterance, termasuk jeda. Noise dipotong sesuai durasi/offset, atau diulang siklis bila perlu. Alpha = sqrt(Pspeech / (Pnoise × 10^(SNR/10))). Jika peak campuran >0.99, kedua komponen diberi gain bersama sehingga SNR tidak rusak. Speech energi nol tetap clean dengan alasan; noise energi nol ditolak. Panjang, sample rate dan target tetap.

Validation memakai OHALLWAY, test memakai PCAFETER; masing-masing memiliki resep tetap 15/5 dB dengan seed, offset sample, channel, hash dan parameter. Jumlah resep validation/test adalah {recipes['validation']['recipes']}/{recipes['test']['recipes']}. Tidak ada campuran test yang dipakai untuk pengembangan/feature smoke. `test_lock.json` mengunci hash manifest, resep dan aturan. `prepare_example` menolak test kecuali evaluasi final meminta akses secara eksplisit.

## Verifikasi aktual

- {verify['check_count']:,} pemeriksaan integritas/manifest/resep/fitur lulus; gagal {verify['failed']}.
- Lima focused unit checks lulus: Unicode/scoring, kutip literal TSV/resampling stereo, reproducible seed dan role split, real RMS SNR/shared clipping gain, dan energi nol/wrap.
- Enam contoh train/validation menghasilkan fitur float32 (80,3000), label roundtrip yang sama, prefix Indonesia/transcription benar, padding -100 dan valid EOS tidak hilang; empat campuran nyata train/validation mempertahankan label dan SNR 5/15 dB.
- Rerun seluruh pipeline selesai dalam {rerun['elapsed_seconds']:.3f} detik. Semua {len(rerun['current_hashes'])} manifest JSONL memiliki hash identik: `{rerun['same_manifest_bytes']}`; output diganti atomik, bukan ditambah/append.
- `pip check` lulus. PyTorch belum dipasang; processor/feature smoke memakai NumPy dan tidak menjalankan model forward.

## Masalah yang benar-benar ditemukan

- Endpoint viewer FLEURS sebelumnya gagal pada Step 2; Step 3 memakai arsip resmi yang tersedia pada revisi yang sama.
- Satu request DEMAND.pdf memperoleh HTTP 502 dan berhasil setelah retry. OOFFICE/PCAFETER sempat gagal pemeriksaan integritas; retry/resume menghasilkan file yang akhirnya cocok checksum resmi. Tidak ada file gagal yang dilabeli terverifikasi.
- Parsing CSV bertanda kutip biasa mengubah bentuk beberapa baris; QUOTE_NONE diverifikasi menjaga ketujuh kolom dan tanda kutip sumber.
- Perbedaan lisensi DEMAND serta bias gender tetap dinyatakan sebagai temuan, bukan disembunyikan.

## Kesiapan Step 4

Data dan preprocessing siap untuk **membangun serta memverifikasi training pipeline**, dengan subset dan keterbatasan di atas. Step 4 masih perlu framework training/PyTorch yang sesuai, batching/dataloader/loss smoke dan rencana runtime; belum ada fine-tuning atau hasil eksperimen. Tidak ada tindakan akun/unduhan manual pengguna yang diperlukan untuk menyelesaikan persiapan otomatis ini. Penggunaan lintas mata kuliah dan lima pengguna nyata tetap merupakan pekerjaan terpisah.
'''
    for name,text in [('DATASET_CARD.md',card),('DATA_SOURCES_AND_LICENSES.md',licensing),('STEP3_DATA_PREPARATION.md',procedure)]:
        (ROOT/'docs'/name).write_text(text,encoding='utf-8')
    from .evidence import export_evidence
    export_evidence()
    event('dataset_documents_rendered',status=status)


if __name__=='__main__':
    build_documents()
