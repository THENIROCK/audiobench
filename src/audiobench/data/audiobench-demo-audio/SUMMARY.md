# AudioBench Demo Audio — Summary & Notes

## Overview

Sample audio files collected for 9 use-case categories demonstrating real-world audio AI applications. Each folder contains one representative audio clip sourced from a published dataset.

---

## Category Samples

### 1. Healthcare + Medical
**Folder:** `healthcare/`
**File:** `heart_sound_M_ESM_RUSB.wav`
**Dataset:** HLS-CMDS — Heart and Lung Sounds Dataset recorded from a Clinical Manikin using Digital Stethoscope
**Source:** https://archive.ics.uci.edu/dataset/1202
**License:** CC BY 4.0
**Label status:** ✅ CSV label — `HS.csv` maps the filename to structured fields
**Labels:** Gender: M | Heart Sound Type: **Early Systolic Murmur** | Location: **RUSB** (Right Upper Sternal Border)
**Notes:** 15-second recording at 22,050 Hz. The full dataset contains 535 WAV files covering heart, lung, and mixed cardiopulmonary sounds — normal and abnormal variants.

---

### 2. Criminal Investigations & Public Safety
**Folder:** `criminal_investigations/`
**File:** `sesa_gunshot_000.wav`
**Dataset:** SESA — Sound Events for Surveillance Applications
**Source:** https://zenodo.org/record/3519845
**License:** CC BY 4.0
**Target Dataset:** MIVIA Audio Events (gunshots, glass breaking, screams) — https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/
**Label status:** ⚠️ Filename only — no separate annotation CSV. `SESA.txt` defines the class mapping; the label is encoded in the filename convention.
**Labels:** Class **1 = Gunshot** (classes: 0 = casual/non-threat, 1 = gunshot, 2 = explosion, 3 = siren/alarm)
**Notes:** SESA is a direct equivalent to MIVIA — purpose-built for surveillance threat detection with the same event classes. All files are WAV, mono, 16 kHz, 16-bit. The full dataset has 585 files split across train/test. MIVIA access still requires a form request at mivia.unisa.it for the original labeled dataset with SNR variants.

---

### 3. Security & Surveillance
**Folder:** `security_surveillance/`
**File:** `esc50_glass_breaking.wav`
**Dataset:** ESC-50 — Environmental Sound Classification
**Source:** https://github.com/karolpiczak/ESC-50
**License:** CC BY-NC 3.0
**Target Dataset:** MIVIA Audio Events — https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/
**Label status:** ✅ CSV label — `meta/esc50.csv` has a row for every file
**Labels:** fold: 1 | target: **39** | category: **glass_breaking** | ESC-10: False
**Notes:** Same MIVIA access limitation as above. Both criminal investigations and security surveillance map to MIVIA, which detects the same threat-class events in surveillance contexts.

---

### 4. Industrial Maintenance
**Folder:** `industrial_maintenance/`
**File:** `mimii_fan_abnormal_00000000.wav`
**Dataset:** MIMII — Malfunctioning Industrial Machine Investigation and Inspection
**Source:** https://zenodo.org/records/3384388
**License:** CC BY-SA 4.0
**Label status:** ✅ Folder structure — the path itself encodes all label fields
**Labels:** Machine type: **fan** | Machine ID: **id_00** | Condition: **abnormal** | SNR: 6 dB
**Notes:** 10-second recording at 16 kHz, 16-bit PCM. The full dataset is ~100 GB covering fans, pumps, sliders, and valves at multiple noise levels. This file was streamed directly from the Zenodo ZIP archive without downloading the full 10 GB file.

---

### 5. Environmental + Wildlife
**Folder:** `environmental_wildlife/`
**File:** `xc1_bird_recording.mp3`
**Dataset:** Xeno-canto — Collaborative database of bird and bat sounds (serves as BirdCLEF source data)
**Source:** https://www.xeno-canto.org/1
**Target Dataset:** BirdCLEF 2026 — https://www.kaggle.com/competitions/birdclef-2026
**Label status:** ⚠️ Metadata via API only — species label is not packaged with the audio file; it lives on the xeno-canto web page and API. BirdCLEF supplies its own `train_metadata.csv` with labels for competition use.
**Labels:** Species: **Collared Warbling Finch** (*Poospiza hispaniolensis*)
**Notes:** BirdCLEF 2026 requires a Kaggle account and competition registration. Xeno-canto is the underlying source library for BirdCLEF competitions. Recording XC1 is a direct download (MP3, 44.1 kHz mono). Xeno-canto API v3 requires a registered key for bulk queries.

---

### 6. Customer Service & Quality Control
**Folder:** `customer_service/`
**File:** `axondata_contact_center_call.mp3`
**Dataset:** English Contact Center Audio Dataset
**Source:** https://huggingface.co/datasets/AxonData/english-contact-center-audio-dataset
**Label status:** ⚠️ Binary, undocumented — the label field exists but its meaning is not defined in the public dataset card.
**Labels:** `label: 0` — binary outcome (0 or 1); likely resolved/unresolved or satisfied/unsatisfied, but not confirmed without the paid full-dataset documentation.
**Notes:** Sample retrieved via the HuggingFace datasets-server preview API. The full dataset (1,000+ hours) is a commercial product available through AxonLabs.

---

### 7. Media, Journalism & Misinformation
**Folder:** `media_misinformation/`
**File:** `scenefake_deepfake_audio.wav`
**Dataset:** SceneFake — Scene Fake Audio Detection
**Source:** https://zenodo.org/record/7663324
**Target Dataset:** Audio Deepfake Detection collection — https://github.com/media-sec-lab/Audio-Deepfake-Detection
**License:** CC BY-NC-ND 4.0
**Label status:** ✅ Folder structure — `dev/fake/` vs `dev/real/` encodes the binary authenticity label
**Labels:** Split: **dev** | Class: **fake** (acoustically manipulated)
**Notes:** This clip (`dev/fake/B_10000_20_C.wav`) is a speech sample where the acoustic scene has been manipulated using speech enhancement. The referenced GitHub repo is a meta-index of deepfake detection datasets; SceneFake is one of several listed.

---

### 8. Defense & Aerospace
**Folder:** `defense_aerospace/`
**File:** `atco2_atc_communication.wav`
**Dataset:** ATCO2 Corpus — 1h Free Subset
**Source:** https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h
**Target Dataset:** LiveATC.net recordings — https://www.liveatc.net/recordings.php
**Label status:** ✅ Text transcript + timestamps — the free 1h subset includes ASR transcription and segment timing. The full ATCO2 corpus additionally provides NER tags (callsign, command, value) and speaker role labels (pilot vs. controller), which are not in the free subset.
**Labels:** Text: *"oscar kilo foxtrot alfa oscar taxi to holding point runway two seven via alfa charlie"* | Start: 0.0 s | End: 5.36 s | Duration: 5.36 s
**Notes:** LiveATC's recordings page is fully JavaScript-rendered and restricts third-party use of its streams. The ATCO2 1h corpus is the free public subset of a 5,281-hour European ATC communications dataset. Recorded at LKTB Brno Tuřany Airport. Sampled at 16 kHz.

---

### 9. Audio & Music Analysis (Valence / Emotions)
**Folder:** `audio_music_emotions/`
**File:** `ravdess_happy_speech_actor01.wav`
**Dataset:** RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song
**Source:** https://zenodo.org/record/1188976
**License:** CC BY-NA-SC 4.0
**Label status:** ✅ Filename-encoded — all label dimensions are embedded in the filename as a structured code
**Labels:** Modality: audio-only | Channel: speech | Emotion: **happy** | Intensity: normal | Statement: 1 ("Kids are talking by the door") | Repetition: 1 | Actor: 01
**Notes:** Filename `03-01-03-01-01-01-01.wav` fully decodes to the above. The full dataset covers 8 emotions (calm, happy, sad, angry, fearful, disgust, surprised, neutral) across 24 professional actors. Commonly used for valence/arousal-based emotion recognition benchmarks. Sampled at 48 kHz.

---

## Label Status Summary

| # | Category | File | Label Status | Label Source |
|---|---|---|---|---|
| 1 | Healthcare | `heart_sound_M_ESM_RUSB.wav` | ✅ Full | `HS.csv` — Gender, Heart Sound Type, Location |
| 2 | Criminal Investigations | `sesa_gunshot_000.wav` | ⚠️ Filename only | Class 1 = Gunshot (no annotation CSV) |
| 3 | Security Surveillance | `esc50_glass_breaking.wav` | ✅ Full | `meta/esc50.csv` — fold, target ID, category name |
| 4 | Industrial Maintenance | `mimii_fan_abnormal_00000000.wav` | ✅ Full | Folder path — machine type, ID, normal/abnormal |
| 5 | Environmental/Wildlife | `xc1_bird_recording.mp3` | ⚠️ API only | Xeno-canto page/API — not bundled with file |
| 6 | Customer Service | `axondata_contact_center_call.mp3` | ⚠️ Binary, undocumented | `label: 0` — meaning undefined without paid docs |
| 7 | Media/Misinformation | `scenefake_deepfake_audio.wav` | ✅ Full | Folder path — `dev/fake/` |
| 8 | Defense/Aerospace | `atco2_atc_communication.wav` | ✅ Partial | Text transcript + timestamps (NER/speaker role in paid version only) |
| 9 | Audio/Music Emotions | `ravdess_happy_speech_actor01.wav` | ✅ Full | Filename-encoded — emotion, intensity, actor, statement |

---

## Access Notes

| Dataset | Access Method | Barrier |
|---|---|---|
| HLS-CMDS | Direct ZIP download | None — open access |
| SESA | Zenodo direct download | None — open access |
| MIVIA Audio Events | Form request | Requires registration at mivia.unisa.it |
| MIMII | Zenodo direct download | None — 100 GB total, file-level streaming possible |
| BirdCLEF 2026 | Kaggle competition page | Requires Kaggle login + competition signup |
| AxonData CC Audio | HuggingFace preview | Full dataset is commercial |
| WaveFake | Zenodo | 28.9 GB ZIP with data descriptors — no easy partial extraction |
| SceneFake | Zenodo | None — 5.8 GB ZIP, partial streaming works |
| LiveATC | Web stream | Third-party use restricted; JS-rendered page |
| ATCO2 1h | HuggingFace | None — free public subset |
| RAVDESS | Zenodo | None — 208.5 MB ZIP, partial streaming works |

---

## File Format Summary

| File | Format | Sample Rate | Duration (approx) |
|---|---|---|---|
| `heart_sound_M_ESM_RUSB.wav` | WAV PCM | 22,050 Hz | 15 s |
| `sesa_gunshot_000.wav` | WAV PCM | 16,000 Hz | ~3 s |
| `esc50_glass_breaking.wav` | WAV PCM | 22,050 Hz | 5 s |
| `mimii_fan_abnormal_00000000.wav` | WAV PCM | 16,000 Hz | 10 s |
| `xc1_bird_recording.mp3` | MP3 64kbps | 44,100 Hz | ~20 s |
| `axondata_contact_center_call.mp3` | MP3 | — | ~11 min |
| `scenefake_deepfake_audio.wav` | WAV PCM | — | ~2 s |
| `atco2_atc_communication.wav` | WAV PCM | 16,000 Hz | 5.4 s |
| `ravdess_happy_speech_actor01.wav` | WAV PCM | 48,000 Hz | ~4 s |
