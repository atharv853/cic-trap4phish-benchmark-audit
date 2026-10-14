# Dataset Preparation & Acquisition Guide: CIC-Trap4Phish

This directory is designated for the feature datasets evaluated in the research paper:

> **“How Much of Malicious-Document Detection Accuracy Is Real? An Audit of CIC-Trap4Phish”**
> *Atharv Kulkarni, Bhargav Joshi, Hrushikesh Jadhav, Manisha Dabade, and Dr. Pawan Kumar*

---

## 1. Dataset Overview

The benchmark evaluated in this research is the **CIC-Trap4Phish** tabular dataset released by the Canadian Institute for Cybersecurity (CIC) / University of New Brunswick (UNB) (Nejati et al., 2026). It contains static-feature extractions across four document formats:

- **Microsoft Word** (`.doc`, `.docx`): 20,000 samples, 43 static features + label.
- **Microsoft Excel** (`.xls`, `.xlsx`): 20,000 samples, 48 static features + label + provenance path.
- **Adobe PDF** (`.pdf`): 19,296 samples, 40 static features + label + provenance path.
- **HTML** (`.html`): 19,997 samples, 40 static features + label + provenance path.

Per the dataset paper:
- **Benign Excel files** were generated synthetically.
- **Benign Word and HTML files** were crawled from public repositories (Google, Wikipedia, PhishTank).
- **Malicious Word and Excel files** were collected from MalwareBazaar.
- **Malicious PDF files** originate from PDFMal2022.
- **Malicious HTML files** originate from PhishTank.

---

## 2. Redistribution Policy & Download Instructions

Because the raw CIC-Trap4Phish dataset is licensed and distributed by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB), redistributing the complete uncompressed 20 MB tabular dump directly in this public GitHub repository is restricted under third-party benchmark distribution terms.

To obtain the official dataset:
1. Visit the Canadian Institute for Cybersecurity (CIC) dataset portal:
   [https://www.unb.ca/cic/datasets/](https://www.unb.ca/cic/datasets/)
2. Request / download the **CIC-Trap4Phish (2025/2026)** feature release archives.
3. Place the four CSV files into this `data/` directory (or the project root):
   - `data/Word_All_features.csv`
   - `data/Excel_All_Features.csv`
   - `data/PDF_All_features.csv`
   - `data/HTML_All_Features.csv`

---

## 3. Verified SHA-256 Checksums and File Properties

To ensure exact reproducibility, verify your downloaded files against the cryptographic checksums used in our paper:

| File Name | Rows | Columns | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `Word_All_features.csv` | 20,000 | 44 | `4137e7324fe53b2a057526bd344bea34de04f2bc735a29545ce1f04370b192af` |
| `Excel_All_Features.csv` | 20,000 | 50 | `c7bc6d70050dd9d622ef158a3c6db868bc881b4a124726c7e90bcf5d64dbb867` |
| `PDF_All_features.csv` | 19,296 | 42 | `6bc698b9d4d96e3befab59e8cf445aeb01ace92461a48a7a141f617f8f3a1467` |
| `HTML_All_Features.csv` | 19,997 | 42 | `075a2d6e2851cba69fdee3428706aac242bbe49814246b9bc4456d85b9b8e121` |

Verify checksums locally using:
```bash
sha256sum data/*.csv
```

---

## 4. Sample Test Data (`data/sample/`)

For smoke testing, CI/CD pipelines, and immediate code verification without downloading the multi-megabyte raw release, this repository includes a stratified 50-sample slice in `data/sample/`:
- `data/sample/Word_Sample.csv` (50 samples, 25 benign / 25 malicious)
- `data/sample/Excel_Sample.csv` (50 samples, 25 benign / 25 malicious)
- `data/sample/PDF_Sample.csv` (50 samples, 25 benign / 25 malicious)
- `data/sample/HTML_Sample.csv` (50 samples, 25 benign / 25 malicious)

The test suite (`tests/test_smoke.py`) and feature adapters (`adapters/universal_adapter.py`) run directly against these samples in under two seconds.
