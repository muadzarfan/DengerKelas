# Sumber data dan lisensi

## FLEURS

Sumber resmi: [Google FLEURS dataset card/repository](https://huggingface.co/datasets/google/fleurs). Konfigurasi `id_id`, revisi `70bb2e84b976b7e960aa89f1c648e09c59f894dd`; lisensi dataset card `cc-by-4.0`. Atribusi proyek: Google/FLEURS, dataset card dan publikasi *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech* yang ditautkan sumber, dengan revisi dan URL tersebut. Lisensi [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) dan atribusi dipertahankan dalam manifest. Salinan README resmi ada di `data/raw/metadata/FLEURS_README.md`.

Repositori saat ini menyediakan TSV dan arsip WAV per bahasa. Pipeline mengunduh hanya Indonesia melalui URL resolve yang dipin revisinya, bukan loader script lama, keseluruhan korpus multibahasa, atau Parquet bahasa lain. Kolom TSV aktual: ID konten, filename, raw_transcription, normalized transcription, grapheme/uroman text, num_samples, gender. Parsing memakai QUOTE_NONE agar kutip literal tidak hilang. Statistik card khusus `id_id` dicocokkan dengan metadata dan audio aktual.

## DEMAND

Sumber resmi: [Zenodo record 1227121](https://zenodo.org/records/1227121), DOI [10.5281/zenodo.1227121](https://doi.org/10.5281/zenodo.1227121), versi 1.0. Atribusi: Joachim Thiemann, Nobutaka Ito, Emmanuel Vincent, *DEMAND: a collection of multi-channel recordings of acoustic noise in diverse environments* (2013).

**Perbedaan sumber lisensi dipertahankan:** API metadata Zenodo menyebut `cc-by-4.0`; deskripsi record serta dokumen DEMAND.pdf menyebut **CC BY-SA 3.0 Unported** ([teks lisensi](https://creativecommons.org/licenses/by-sa/3.0/)). Metadata dan dokumen asli disimpan lokal; pipeline tidak mengklaim perbedaan ini sudah diselesaikan. Pertahankan atribusi dan pemberitahuan ShareAlike dokumen ketika merencanakan distribusi audio turunan; jangan mengganti lisensi dengan klaim tunggal yang mengabaikan konflik. Tidak ada audio mentah/campuran yang dipublikasikan ke Git.

Hanya OOFFICE_16k.zip, OHALLWAY_16k.zip dan PCAFETER_16k.zip serta DEMAND.pdf yang diunduh. Arsip masing-masing memiliki 16 WAV mono; hanya `ch01.wav` dipakai. Tidak ditemukan file lisensi terpisah dalam ketiga arsip yang dipilih, sebagaimana `results/step3/extraction.json`; PDF dan deskripsi merupakan bukti pemberitahuan lisensi yang ditemukan.

## Processor

Sumber [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny), revisi `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, dataset/model card berlabel Apache-2.0. Hanya processor/tokenizer/config yang diunduh untuk Step 3; bobot training baru tidak diunduh. Runtime inferensi Step 2 dan manifestnya tetap terpisah.

## Integritas unduhan

| Artefak resmi | Byte terunduh | Checksum resmi |
|---|---:|---|
| `OHALLWAY_16k.zip` | 77941854 | md5: `fe918bbb0e63e73d09ba7f4843ef33f1` |
| `OOFFICE_16k.zip` | 88995191 | md5: `7b61cc2d182d5a654cb9c3101ddd4041` |
| `PCAFETER_16k.zip` | 107431494 | md5: `99927d148128254141a9417d051510bb` |
| `data/id_id/audio/dev.tar.gz` | 220586217 | sha256: `74e5f1c8676a321dc4c34944b171846d40923dabe765a767d8dbc2f56cac3142` |
| `data/id_id/audio/test.tar.gz` | 451704157 | sha256: `bf7e7f608f8834012d7f79a583b6b0dd5bb82e74e2f5fb6336fc99d3af7ffcf8` |
| `data/id_id/audio/train.tar.gz` | 1697523639 | sha256: `44983febf5a3d9670352f9765d05b9238056bf6ab64ee1479af198670597c11d` |

Semua 21 file yang diminta telah lolos ukuran/checksum resmi. Berkas LFS memakai SHA-256, metadata kecil memakai Git blob SHA-1 sumber, dan DEMAND memakai MD5 resmi serta SHA-256 lokal. Daftar URL, ukuran, revisi dan checksum lokal lengkap: `data/manifests/step3_downloads.json`. Semua sumber publik, tanpa autentikasi atau API berbayar. Kegagalan unduhan dan retry tercatat di `results/step3/acquisition.jsonl`.
