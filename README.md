# How Much of Malicious-Document Detection Accuracy Is Real? An Audit of CIC-Trap4Phish

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Verification Status](https://img.shields.io/badge/Verification-Passed%20100%25-brightgreen.svg)](#16-reproducing-the-experiments)
[![IEEE Conference](https://img.shields.io/badge/Paper-IEEE-blue.svg)](#20-citation)

This repository contains the complete empirical code, benchmark audit suite, split indices, tuned hyperparameters, statistical tests, and result artifacts for the research paper:

> **“How Much of Malicious-Document Detection Accuracy Is Real? An Audit of CIC-Trap4Phish”**
> *Atharv Kulkarni, Bhargav Joshi, Hrushikesh Jadhav, Manisha Dabade, and Dr. Pawan Kumar*
> Walchand College of Engineering, Sangli & Indian Institute of Information Technology (IIIT) Dharwad, India.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Research Motivation](#2-research-motivation)
3. [Research Questions and Contributions](#3-research-questions-and-contributions)
4. [Dataset and Formats](#4-dataset-and-formats)
5. [Evaluation Protocol: L0, L1, and L2](#5-evaluation-protocol-l0-l1-and-l2)
6. [Duplicate and Twin-Leakage Audit](#6-duplicate-and-twin-leakage-audit)
7. [Provenance and Shortcut Analysis](#7-provenance-and-shortcut-analysis)
8. [Honest Leaderboard](#8-honest-leaderboard)
9. [Universal-Feature and Cross-Format Experiments](#9-universal-feature-and-cross-format-experiments)
10. [Sensitivity Analysis](#10-sensitivity-analysis)
11. [SHAP and HTML Error Analysis](#11-shap-and-html-error-analysis)
12. [Statistical Evaluation](#12-statistical-evaluation)
13. [Repository Structure](#13-repository-structure)
14. [Installation](#14-installation)
15. [Dataset Preparation](#15-dataset-preparation)
16. [Reproducing the Experiments](#16-reproducing-the-experiments)
17. [Understanding the Results](#17-understanding-the-results)
18. [Limitations and Threats to Validity](#18-limitations-and-threats-to-validity)
19. [Responsible Use and Data Privacy](#19-responsible-use-and-data-privacy)
20. [Citation](#20-citation)
21. [Acknowledgments](#21-acknowledgments)

---

## 1. Overview

Public static-feature benchmarks for malicious office, PDF, and HTML attachments frequently report near-saturated detection accuracy ($\ge 98\%$). However, machine-learning security evaluations are notoriously vulnerable to subtle dataset collection artifacts, duplicate records, train/test leakage, and shortcut learning.

This repository provides an empirical audit of the **CIC-Trap4Phish (2025/2026)** benchmark release across all four document formats (Microsoft Word, Microsoft Excel, Adobe PDF, and HTML). Rather than claiming a new state-of-the-art detector, this work rigorously decomposes what fraction of reported detection accuracy reflects real malicous document detection versus experimental artifacts.

**Key Quantitative Takeaways:**
- **High Duplication & Twin Leakage:** Between $22.8\%$ and $33.0\%$ of benchmark rows lie inside exact-duplicate feature groups, causing $20.9\%$ to $32.9\%$ of test rows in standard naive splits to possess an identical "feature twin" in the training set.
- **Twin vs. Non-Twin Gap:** On HTML, LightGBM achieves an F1 of $0.9860$ $[0.9804, 0.9909]$ on twin test rows, but plummets to $0.8566$ $[0.8402, 0.8731]$ on non-twin test rows.
- **Word Shortcut Collapse:** In Word, $30$ of $43$ features are missing in exactly $50\%$ of rows (perfectly partitioned by class), and $19$ features yield single-split decision stump accuracies $\ge 0.95$. Removing these $41$ shortcut columns drops Word F1 from $1.0000$ to an honest $0.9366$ $[0.9292, 0.9448]$.
- **Excel Provenance Separability:** Removing all $22$ macro and API features from Excel leaves the model with an F1 of $1.0000$ $[1.0000, 1.0000]$, reflecting synthetic benign generation rather than genuine malicious content detection.
- **Cross-Format Generalization:** A pooled 6-feature format-agnostic model trails specialists by only $0.0000$ on Excel and $0.0093$ on PDF, but lags by $0.0731$ on HTML, where the transfer gap narrows linearly as the feature budget expands.

---

## 2. Research Motivation

Modern email filtering and endpoint protection rely heavily on fast static tabular classifiers to flag malicious document attachments before expensive dynamic sandboxing. When a static tabular benchmark reports $99.8\%$ or $100\%$ detection rates, practitioners may conclude that malicious document detection is an essentially solved problem.

However, recent methodological critiques in machine learning security (e.g., Arp et al., 2022; Geirhos et al., 2020; Allamanis, 2019) emphasize that tabular benchmarks frequently encode collection artifacts:
1. Benign and malicious files gathered from mismatched sources or different environments (provenance confounding).
2. Data collection scrapers inserting systematic missing-value patterns.
3. Near-identical documents (variants, identical templates) split across train and test partitions without deduplication.

This audit tests whether the saturated metrics reported in the literature on CIC-Trap4Phish reflect genuine document security capabilities or benchmark-specific leakage.

---

## 3. Research Questions and Contributions

This research investigates three central questions:
1. **RQ1:** How much reported accuracy survives rigorous deduplication, twin removal, and shortcut stripping?
2. **RQ2:** What is the honest per-format baseline performance when evaluating against genuine held-out samples?
3. **RQ3:** Can a compact set of semantically shared, format-agnostic static features match per-format specialist models across Word, Excel, PDF, and HTML?

### Core Contributions
- **C1 (Measured Audit):** Systematic quantification of exact duplicate groups, train/test twin leakage, single-feature decision stump shortcuts, and file-path provenance leakage across all four formats.
- **C2 (Three-Level Protocol & Honest Leaderboard):** Definition of evaluation levels **L0** (naive), **L1** (deduplicated), and **L2** (shortcut/missingness removed), establishing an honest leaderboard with $95\%$ bootstrap confidence intervals across five model families (LightGBM, XGBoost, Random Forest, Decision Tree, Logistic Regression).
- **C3 (Universal Feature Study):** Development of 10 format-agnostic feature slots, evaluating pooled cross-format models, feature-budget sensitivity ($k \in \{4, 6, 8, 10\}$), and comparing shortcut-bearing versus shortcut-free Word adapters.
- **C4 (Zero-Filled Union & Attribution Limits):** SHAP importance attribution on zero-filled format unions, leave-one-format-out (LOFO) transfer failure analysis, and local explanation instability (LIME vs. SHAP vs. Permutation).
- **C5 (Reproducibility & Benchmark Checklist):** A fully reproducible release including seeds, exact train/test split indices, tuned hyperparameter configurations, and an empirical checklist for future benchmark creators.

---

## 4. Dataset and Formats

The audit evaluates the static-feature tabular release of **CIC-Trap4Phish (2025/2026)** (Nejati et al., 2026):

| Document Format | Extension | Raw Rows | Feature Columns | Benign Source | Malicious Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Word** | `.doc`, `.docx` | 20,000 | 43 (+ label) | Public Crawl (Google/Wikipedia) | MalwareBazaar |
| **Excel** | `.xls`, `.xlsx` | 20,000 | 48 (+ path, label) | Synthetically Generated | MalwareBazaar |
| **PDF** | `.pdf` | 19,296 | 40 (+ path, label) | PDFMal2022 | PDFMal2022 |
| **HTML** | `.html` | 19,997 | 40 (+ path, label) | Public Crawl (PhishTank/Alexa) | PhishTank |

> **Important Cross-Paper Note:** Prior publications on this dataset (e.g., Sen & Chowdhury, 2026; Ayodeji et al., 2026; Nejati et al., 2026) utilized different splits (e.g., 80/20 unstratified, 70/30), different feature subsets (top-10/13 selected features), or pooled 8-class classification without deduplication. Consequently, raw metric values across papers are **not directly comparable** unless experimental setups, deduplication rules, and target labels match identically.

---

## 5. Evaluation Protocol: L0, L1, and L2

To disentangle the sources of performance, we define three hierarchical evaluation tiers:

```text
Raw Data (N=79,293)
   │
   ├─► Level 0 (L0): Naive Split
   │     Stratified 80/20 train/test split on raw, uncleaned feature vectors (Seed 42).
   │
   ├─► Level 1 (L1): Exact Deduplication
   │     Remove exact duplicate feature vectors (keep-first).
   │     Remove 6 conflicting-label vectors (identical features appearing as both benign and malicious).
   │     Stratified 80/20 train/test split (guaranteeing 0 exact twins between train and test).
   │
   └─► Level 2 (L2): Honest Partition (Shortcut & Missingness Free)
         Builds upon L1 deduplicated split.
         Step (a): Remove any column containing missing values (NaN) in raw data.
         Step (b): Iteratively remove features with 5-fold CV decision-stump accuracy >= 0.95.
```

- **L0:** Reproduces naive benchmark splitting commonly reported in literature.
- **L1:** Eliminates memorization of exact training samples during testing.
- **L2:** Eliminates single-feature shortcuts and collection artifacts, isolating residual signal.

---

## 6. Duplicate and Twin-Leakage Audit

Our audit reveals severe exact duplication and train/test leakage in naive splits (`results/audit_duplicates.csv`, `results/audit_twin_leakage.csv`):

| Format | Raw Rows | Rows in Duplicate Groups | Duplicate Ben./Mal. | Rows Removed (Keep-First) | Clean L1 Rows | L0 Test Twin Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Word** | 20,000 | 4,558 (22.8%) | 4,558 / 0 | 3,395 | 16,605 | **20.9%** (834 / 4,000) |
| **Excel** | 20,000 | 6,591 (33.0%) | 0 / 6,591 | 6,028 | 13,972 | **32.9%** (1,314 / 4,000) |
| **PDF** | 19,296 | 5,703 (29.6%) | 28 / 5,675 | 3,943 | 15,353 | **27.8%** (1,074 / 3,860) |
| **HTML** | 19,997 | 5,683 (28.4%) | 434 / 5,249 | 4,587 (+6 conflicts) | 15,404 | **27.1%** (1,083 / 4,000) |

### Key Duplicate Findings
1. **Extreme Label Skew:** Duplicates are almost entirely one-sided: all 4,558 Word duplicate rows are benign; all 6,591 Excel duplicate rows are malicious; $99.5\%$ of PDF duplicates and $92.4\%$ of HTML duplicates are malicious.
2. **Conflicting Feature Vectors:** In HTML, six feature vectors are associated with both benign and malicious labels simultaneously, demonstrating feature ambiguity in static HTML parsing.
3. **Twin Leakage:** In naive splits (L0), roughly one in four test samples is an identical twin of a training sample.

---

## 7. Provenance and Shortcut Analysis

### Provenance Confounding
In datasets where benign and malicious files originate from separate repositories, directory names and file metadata encode strong labels (`results/audit_provenance.csv`):
- **Directory Name Depth-1 Tree:** Predicts the label with **$100.0\%$ accuracy** ($1.0000$) in Excel, PDF, and HTML (Word release omitted file path metadata).
- **File Extension Alone:** Yields $0.6310$ accuracy in Excel (`.xls` vs `.xlsx`), $0.5180$ in PDF, and $0.5000$ in HTML.

### Single-Feature Shortcuts
A single-feature decision stump screens whether individual features trivially separate classes (`results/audit_shortcuts.csv`):
- **Word Missingness Artifact:** Exactly $30$ of the $43$ feature columns are missing in exactly $10,000$ rows ($22$ columns missing only in benign files; $8$ missing only in malicious files). A simple indicator `isnan(column)` achieves **$100.0\%$ accuracy** on Word malware detection.
- **Word Content Shortcuts:** $19$ features reach stump accuracy $\ge 0.95$. Features `ole_object_count`, `ole_object_type_count`, and `vba_keywords_count` yield **$1.0000$ accuracy**; `ole_object_count` is $0$ for every benign file and between $11$ and $100$ for every malicious file.
- **Excel Shortcuts:** $9$ features reach stump accuracy $\ge 0.95$, led by `entropy_of_text` ($0.9870$), `macro_chr_count` ($0.9850$), and `macro_token_count` ($0.9800$).
- **PDF & HTML:** No feature reaches the $0.95$ threshold (maxima: PDF $0.9290$, HTML $0.7880$).

Consequently, L2 removes **$41$ of $43$ Word columns** (leaving only `struct_typeface` and `struct_script`) and **$9$ of $48$ Excel columns**.

---

## 8. Honest Leaderboard

Specialist LightGBM models tuned via Optuna (TPE sampler, 8 trials, 3-fold inner CV on training sets) evaluated across L0, L1, and L2 (`results/honest_leaderboard.csv`):

| Level | Format | Test F1 | 95% Bootstrap CI | Accuracy | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L0** | Word | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1.0000 | 1.0000 |
| **L0** | Excel | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1.0000 | 1.0000 |
| **L0** | PDF | 0.9988 | [0.9978, 0.9997] | 0.9987 | 0.9985 | 0.9990 |
| **L0** | HTML | 0.9187 | [0.9096, 0.9272] | 0.9175 | 0.9057 | 0.9320 |
| **L1** | Word | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1.0000 | 1.0000 |
| **L1** | Excel | 0.9993 | [0.9979, 1.0000] | 0.9993 | 0.9985 | 1.0000 |
| **L1** | PDF | 0.9974 | [0.9955, 0.9989] | 0.9974 | 0.9974 | 0.9974 |
| **L1** | HTML | 0.9276 | [0.9178, 0.9375] | 0.9276 | 0.9243 | 0.9309 |
| **L2** | Word | **0.9366** | **[0.9292, 0.9448]** | **0.9401** | **0.9859** | **0.8920** |
| **L2** | Excel | 0.9989 | [0.9977, 1.0000] | 0.9989 | 1.0000 | 0.9978 |
| **L2** | PDF | 0.9974 | [0.9955, 0.9989] | 0.9974 | 0.9974 | 0.9974 |
| **L2** | HTML | 0.9276 | [0.9178, 0.9375] | 0.9276 | 0.9243 | 0.9309 |

### Empirical Insights
- **Word Saturated Scores are Fragile:** Stripping the $41$ shortcut/missing columns causes Word F1 to drop by over $6.3$ percentage points ($1.0000 \to 0.9366$).
- **Deduplication Alone Does Not Degrade F1 Across All Formats:** Deduplication changes class balance and sample composition; on HTML, L1 F1 ($0.9276$) is slightly higher than L0 ($0.9187$). Thus, hypothesis H1 (deduplication degrades F1 in every format) is **not supported**. Leakage is revealed by twin versus non-twin stratification.

---

## 9. Universal-Feature and Cross-Format Experiments

We map document-specific features into $10$ semantic format-agnostic slots (`adapters/universal_adapter.py`):
1. `content_entropy`
2. `active_content_flag`
3. `external_reference_count`
4. `log_file_size`
5. `structural_complexity_ratio`
6. `obfuscation_indicator`
7. `script_text_volume`
8. `encoded_content_counts`
9. `embedded_object_count`
10. `stream_script_size`

### Transfer Gap $\Delta F1$ (Specialist F1 $-$ Universal F1)
Evaluated across per-format universal models ($k \in \{4, 6, 8, 10\}$) and a single pooled cross-format LightGBM ($k=6$) (`results/sensitivity_k.csv`, `results/transfer_gap_significance.csv`):

| Format | $k=4$ | $k=6$ | $k=8$ | $k=10$ | Pooled Model ($k=6$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Word (Shortcut-bearing)** | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -- |
| **Word (No Shortcuts)** | 0.1578 [0.1432, 0.1746] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | **0.6319** [0.6085, 0.6540] |
| **Excel** | -0.0007 [-0.0019, 0.0000] | -0.0007 [-0.0019, 0.0000] | -0.0007 [-0.0019, 0.0000] | -0.0007 [-0.0019, 0.0000] | **0.0000** [-0.0015, 0.0015] |
| **PDF** | 0.0116 [0.0079, 0.0159] | 0.0038 [0.0013, 0.0065] | 0.0019 [-0.0006, 0.0045] | 0.0025 [0.0003, 0.0051] | **0.0093** [0.0059, 0.0131] |
| **HTML** | 0.0696 [0.0601, 0.0854] | 0.0553 [0.0469, 0.0707] | 0.0490 [0.0410, 0.0631] | 0.0384 [0.0307, 0.0533] | **0.0731** [0.0605, 0.0853] |

- **Excel & PDF:** Format-agnostic abstractions match or nearly match specialists.
- **HTML Penalty:** HTML suffers a statistically significant transfer penalty ($\Delta F1 = 0.0731$, $p = 0.00018$, Cohen's $d = 12.2$).
- **Word Shortcut Masking:** Shortcut-bearing Word adapters appear universally generalizable ($\Delta F1 = 0.0000$), but when evaluated on clean features, the pooled model collapses to $F1 = 0.3047$ ($\Delta F1 = 0.6319$), proving that apparent universality was driven by shortcut leakage.

---

## 10. Sensitivity Analysis

As the feature budget $k$ increases from $4$ to $10$ (`results/sensitivity_k.csv`):
- **HTML Gap Shrinkage:** The HTML transfer gap contracts steadily from $0.0696$ ($k=4$) down to $0.0384$ ($k=10$), following an empirical regression slope of **$-0.004995$ per feature** (Hypothesis H5 **supported**). The gap does not plateau within $k \le 10$, indicating that higher feature budgets continuously recover format-specific nuances.
- **Clean Word Recovery:** "Word (No Shortcuts)" shows a large gap at $k=4$ ($0.1578$) because only `struct_script` is available; once $k \ge 6$ introduces `struct_typeface`, the gap drops to $0.0000$.

---

## 11. SHAP and HTML Error Analysis

### Zero-Filled Union Representation (Arm 4)
Concatenating all columns into a zero-filled union (as done by prior work) leads to severe misattribution (`results/arm4_shap_share.csv`):
- In L1 models, **$33\%$ to $51\%$ of total feature importance** falls onto columns corresponding to other document formats (which are zero-filled for that sample).
- Own-format mean $|$SHAP$|$ shares: Word $67.0\%$, Excel $62.1\%$, PDF $48.6\%$, HTML $51.8\%$.
- Because PDF own-format share is $< 50\%$, Hypothesis H6 is **not supported**.

### HTML Diagnostic Analysis & Explainability Instability
Auditing all $223$ HTML errors at L1 ($117$ false positives, $106$ false negatives) (`results/html_error_analysis.md`):
- **LIME Instability:** LIME ranks `mailto_link_count` as the #1 feature for $216$ of $223$ error samples ($96.8\%$), reflecting local tabular perturbation sensitivity rather than true causal drivers.
- **Methodological Divergence:** TreeSHAP, LIME, and Permutation Importance share **zero features** in their respective top-5 rankings across all three methods.
- **Error Clustering:** K-Means identifies $3$ distinct error clusters with a silhouette score of $0.8840$.

---

## 12. Statistical Evaluation

All empirical hypotheses were pre-specified and evaluated using rigorous statistical tests (`results/hypotheses.csv`):
- **Paired Comparisons:** Nadeau-Bengio corrected resampled $t$-test ($5 \times 5$ repeated CV, $J=25$, 24 d.o.f.), McNemar's exact test, and Holm step-down family-wise error rate corrections.
- **Confidence Intervals:** Vectorized non-parametric bootstrap ($2,000$ resamples, $95\%$ percentile intervals).

### Formal Hypothesis Decisions

| ID | Formal Pre-specified Hypothesis | Outcome | Empirical Evidence |
| :--- | :--- | :--- | :--- |
| **H1** | Deduplication lowers test metrics in every format | **NOT SUPPORTED** | Word $1.000 \to 1.000$; Excel $1.000 \to 0.999$; PDF $0.999 \to 0.997$; HTML $0.919 \to 0.928$ |
| **H2** | Word and Excel each contain a feature with stump accuracy $\ge 0.95$ | **SUPPORTED** | Word has $19$ columns $\ge 0.95$; Excel has $9$ columns $\ge 0.95$ |
| **H3** | L2 F1 is strictly lower than L1 for Word and Excel | **SUPPORTED** | Word $1.0000 \to 0.9366$ (disjoint CIs); Excel $0.9993 \to 0.9989$ (overlapping CIs) |
| **H4** | Pooled universal F1 is within $0.01$ of specialist for Word, Excel, PDF but not HTML | **INCONCLUSIVE** | PDF CI upper bound is $0.0131 > 0.01$; Excel $\Delta F1=0.000$; HTML $\Delta F1=0.073$ |
| **H5** | The HTML universal transfer gap shrinks as feature budget $k$ grows | **SUPPORTED** | Linear regression slope of $\Delta F1$ on $k$ is $-0.004995$ per feature |
| **H6** | Own-format columns carry $\ge 50\%$ importance in zero-filled union | **NOT SUPPORTED** | Own-format shares at L1: Word $0.670$, Excel $0.621$, PDF $0.486$, HTML $0.518$ |

---

## 13. Repository Structure

```text
cic-trap4phish-benchmark-audit/
├── README.md                          # Comprehensive 21-section research documentation
├── LICENSE                            # MIT Open-Source License
├── CITATION.cff                       # Citation File Format metadata
├── requirements.txt                   # Exact pinned dependencies
├── config.yaml                        # Configuration paths, seeds, and hyperparameters
├── .gitignore                         # Research-grade git ignore configuration
├── VERSION                            # Release version (1.6.0)
├── CLAIMS.md                          # Research claims and implementation ledger
├── run_pipeline.py                    # Master execution CLI (check, verify, smoke, full)
├── check_13_files.py                  # Standalone verification for 13 target artifacts
├── verify.py                          # Comprehensive empirical audit verification suite
├── adapters/                          # Universal feature extraction modules
│   ├── __init__.py
│   └── universal_adapter.py           # 10-slot format-agnostic semantic adapter
├── scripts/                           # Modular execution stages
│   ├── __init__.py
│   ├── run_all.py                     # Master sequential orchestrator
│   ├── utils.py                       # Shared path resolvers and sanitization utilities
│   ├── audit/                         # Stage 0: Data audit, duplicates, shortcuts, provenance
│   │   ├── __init__.py
│   │   └── stage0_audit.py
│   ├── preprocessing/                 # Stage 1: L0, L1, L2 partitioning & shortcut erosion
│   │   ├── __init__.py
│   │   └── stage1_levels.py
│   ├── training/                      # Stage 2: Optuna tuning, specialists, ensembles
│   │   ├── __init__.py
│   │   └── stage2_models.py
│   ├── evaluation/                    # Stages 3 & 4: Universal models & statistical testing
│   │   ├── __init__.py
│   │   ├── stage3_universal.py
│   │   └── stage4_stats.py
│   ├── explainability/                # Stage 5: TreeSHAP, HTML LIME, K-Means clustering
│   │   ├── __init__.py
│   │   └── stage5_explainability.py
│   ├── visualization/                 # Stage 6: Figures (1-16) and LaTeX tables
│   │   ├── __init__.py
│   │   └── stage6_figures.py
│   └── gates/                         # Stage-by-stage verification gates
│       ├── __init__.py
│       └── gate0.py ... gate6.py
├── results/                           # Pre-computed research outputs
│   ├── honest_leaderboard.csv         # Table 2: Multi-level honest leaderboard
│   ├── audit_provenance.csv           # Path / extension predictive accuracy
│   ├── audit_shortcuts.csv            # Single-feature decision stump ranking
│   ├── l2_removed_features.csv        # Features removed at L2 with justification
│   ├── twin_vs_nontwin.csv            # Twin vs. non-twin evaluation metrics
│   ├── sensitivity_k.csv              # Sensitivity study across feature budgets k
│   ├── transfer_gap_significance.csv  # Nadeau-Bengio, McNemar & bootstrap tests
│   ├── hypotheses.csv                 # Decisions for pre-specified hypotheses H1-H6
│   ├── arm4_shap_share.csv            # Own-format vs other-format SHAP shares
│   ├── lofo_matrix.csv                # Leave-one-format-out transfer matrix
│   ├── ensemble_comparison.csv        # Stacking and soft-voting ensemble metrics
│   ├── mapping_log.json               # Formal semantic mappings for 10 universal slots
│   ├── html_error_analysis.md         # Diagnostic report on HTML test errors
│   ├── ablation_table.csv             # Full ablation breakdown across models
│   ├── audit_duplicates.csv           # Exact duplicate group counts and label skew
│   ├── audit_missingness.csv          # Per-feature missingness audit
│   ├── audit_shapes.csv               # Dataset dimensions and class balance
│   ├── audit_twin_leakage.csv         # Train/test twin counts across splits
│   ├── control_experiments.csv        # Control experiments for Word and Excel
│   ├── html_errors_lime.csv           # Per-sample signed LIME weights for HTML errors
│   ├── manifest.json                  # Pipeline execution timestamps and status
│   ├── metrics_all.csv                # Comprehensive model metrics across all runs
│   ├── paper_numbers.json             # Key claim-to-source-file mapping ledger
│   ├── shortcut_erosion_curve.csv     # F1 progression as shortcuts are stripped
│   ├── summary.md                     # High-level empirical summary
│   ├── word_missingness_analysis.csv  # Decomposition of Word missingness flag
│   ├── params/                        # Tuned Optuna hyperparameter JSONs (74 files)
│   └── splits/                        # Exact train/test .npz index splits (12 files)
├── figures/                           # High-resolution research figures (PNG + PDF)
│   ├── fig1_duplicate_twin_leakage.{png,pdf}
│   ├── fig2_stump_auc_top_features.{png,pdf}
│   ├── fig3_word_missingness_separation.{png,pdf}
│   ├── fig4_shortcut_erosion_curves.{png,pdf}
│   ├── fig5_l0_l1_l2_grouped_bars.{png,pdf}
│   ├── fig6_confusion_matrices.{png,pdf}
│   ├── fig7_roc_pr_curves.{png,pdf}
│   ├── fig8_threshold_vs_f1_curves.{png,pdf}
│   ├── fig9_shap_summary.{png,pdf}
│   ├── fig10_arm4_shap_share.{png,pdf}
│   ├── fig11_transfer_gap_forest.{png,pdf}
│   ├── fig12_sensitivity_k.{png,pdf}
│   ├── fig13_leave_one_format_out_heatmap.{png,pdf}
│   ├── fig14_latency_bars.{png,pdf}
│   ├── fig15_html_error_clusters.{png,pdf}
│   └── fig16_ensemble_comparison.{png,pdf}
├── tables/                            # Publication-ready LaTeX tables
│   ├── hypotheses.tex
│   ├── lofo_matrix.tex
│   ├── table2_leaderboard.tex
│   ├── table3_ablation.tex
│   └── table4_transfer.tex
├── logs/                              # Execution logs for stages 0 through 6
├── data/
│   ├── README.md                      # Dataset acquisition and checksum verification guide
│   └── sample/                        # Stratified 50-row samples for smoke testing
├── paper/                             # IEEE manuscript LaTeX sources and PDF preprint
└── tests/                             # Unit tests and automated smoke validation
    ├── test_artifacts.py
    ├── test_adapters.py
    └── test_smoke.py
```

---

## 14. Installation

### Requirements
- **OS:** Linux, macOS, or Windows
- **Python:** Version `3.10` or higher (verified on Python `3.13.7`)

### Environment Setup
Clone the repository and install dependencies in a clean virtual environment:

```bash
git clone git@github.com:atharv853/cic-trap4phish-benchmark-audit.git
cd cic-trap4phish-benchmark-audit

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 15. Dataset Preparation

To evaluate or re-train models on the full benchmark:
1. Download the official **CIC-Trap4Phish (2025/2026)** feature files from the Canadian Institute for Cybersecurity (UNB): [https://www.unb.ca/cic/datasets/](https://www.unb.ca/cic/datasets/).
2. Place the four feature CSV files into `data/` (or the repository root):
   - `data/Word_All_features.csv`
   - `data/Excel_All_Features.csv`
   - `data/PDF_All_features.csv`
   - `data/HTML_All_Features.csv`
3. Verify SHA-256 checksums:
   ```bash
   sha256sum data/*.csv
   ```
   *(Expected checksums are documented in [data/README.md](data/README.md)).*

> **Testing Without the Full Dataset:** You do not need to download the full 20 MB dataset to inspect or test the codebase. The repository includes pre-computed outputs and lightweight sample datasets (`data/sample/`) that support immediate verification and smoke testing.

---

## 16. Reproducing the Experiments

### Quick Validation (1 second)
Check that all 13 core result artifacts exist, are non-empty, and parse cleanly:
```bash
python run_pipeline.py --check
```
*Alternatively:* `python check_13_files.py`

### Automated Empirical Verification (3 seconds)
Run the full verification suite to validate all 20 CSV schemas, manifest timestamps, hypothesis consistency, bootstrap confidence intervals, and figure files:
```bash
python run_pipeline.py --verify
```
*Alternatively:* `python verify.py`

### Run Smoke Test Suite (5 seconds)
Execute the complete unit and smoke test suite against sample datasets:
```bash
python run_pipeline.py --smoke
```
*Alternatively:* `python -m unittest discover -s tests`

### Re-generating Figures and LaTeX Tables (15 seconds)
Re-generate all 16 figures (PNG and PDF) and LaTeX tables from pre-computed results without running model training:
```bash
python run_pipeline.py --stage 6
```

### Full Pipeline Reproduction (~30 minutes)
To re-run the entire pipeline from scratch (data audit $\to$ level partitioning $\to$ Optuna hyperparameter optimization $\to$ specialist models $\to$ universal adapters $\to$ statistical hypothesis tests $\to$ SHAP/LIME explainability $\to$ figures):
```bash
# Requires placing full dataset files into data/
python run_pipeline.py --full
```

---

## 17. Understanding the Results

### The 13 Key Research Artifacts

1. **`results/honest_leaderboard.csv`:** Complete performance breakdown across L0, L1, and L2 for all four formats with $95\%$ bootstrap confidence intervals.
2. **`results/audit_provenance.csv`:** Predictive accuracy of directory names and file extensions on ground-truth labels.
3. **`results/audit_shortcuts.csv`:** Decision stump accuracies for every feature, identifying shortcut candidates ($\ge 0.95$).
4. **`results/l2_removed_features.csv`:** Exact list of columns stripped at Level 2, with detailed removal justifications.
5. **`results/twin_vs_nontwin.csv`:** Stratified evaluation of naive split test sets on twin versus non-twin instances.
6. **`results/sensitivity_k.csv`:** Universal feature transfer performance across feature budgets $k \in \{4, 6, 8, 10\}$.
7. **`results/transfer_gap_significance.csv`:** Statistical test results ($t$-stats, degrees of freedom, raw & Holm-adjusted $p$-values, McNemar $p$, Cohen's $d$).
8. **`results/hypotheses.csv`:** Structured evaluation decisions and empirical evidence for hypotheses H1 through H6.
9. **`results/arm4_shap_share.csv`:** TreeSHAP importance shares allocated to own-format vs. foreign zero-filled columns.
10. **`results/lofo_matrix.csv`:** Leave-one-format-out cross-format transfer accuracy and F1 performance.
11. **`results/ensemble_comparison.csv`:** Comparative performance of tuned single models, soft voting, and stacking ensembles.
12. **`results/mapping_log.json`:** Formal semantic mapping rules defining the 10 universal feature slots for each format.
13. **`results/html_error_analysis.md`:** Detailed error report analyzing false positives, false negatives, LIME instability, and K-Means clusters on HTML.

---

## 18. Limitations and Threats to Validity

1. **Static Tabular Scope:** This study audits the static tabular feature representations provided in the CIC-Trap4Phish benchmark release; it does not perform dynamic analysis, raw binary disassembling, or byte-level deep learning.
2. **Synthetic Benign Files:** Benign Excel files in the release were synthetically generated, rendering content-based separation artificial for Excel.
3. **Missing Word Path Metadata:** Provenance confounding could not be evaluated on Word because the release omitted file paths.
4. **Exact vs. Near Deduplication:** Only exact duplicate feature vectors were stripped. Near-duplicate filtering (e.g., Euclidean clustering on standardized vectors) was explored but omitted from reporting because missing-value imputation introduces artificial distance distortions.
5. **Hand-Designed Universal Adapters:** Universal feature slots were hand-designed based on semantic document properties; other domain-specific mappings may achieve different trade-offs.
6. **Local Explainability Tool Limits:** LIME proved unstable on error instances, and explanations disagreed across methods; explainability outputs are reported as diagnostic observations rather than ground-truth causal mechanisms.

---

## 19. Responsible Use and Data Privacy

- **Benchmarking Ethics:** This repository is intended strictly for academic evaluation, reproducibility, and defensive machine-learning research.
- **No Malicious Payloads:** This repository contains only tabular numerical features, summary statistics, and analysis code. **No raw malicious files, executable macros, or active payloads are distributed.**
- **Redistribution Terms:** The raw feature datasets remain the property of the Canadian Institute for Cybersecurity (CIC) / University of New Brunswick and must be acquired via their official portal.

---

## 20. Citation

If you use this benchmark audit, code, or experimental results in your research, please cite our paper:

```bibtex
@inproceedings{kulkarni2026cictrap4phish,
  title     = {How Much of Malicious-Document Detection Accuracy Is Real? An Audit of {CIC-Trap4Phish}},
  author    = {Kulkarni, Atharv and Joshi, Bhargav and Jadhav, Hrushikesh and Dabade, Manisha and Kumar, Pawan},
  booktitle = {Proceedings of the IEEE Conference on Secure Development and Machine Learning},
  year      = {2026},
  url       = {https://github.com/atharv853/cic-trap4phish-benchmark-audit}
}
```

---

## 21. Acknowledgments

This research was conducted in collaboration between the Department of Computer Science and Engineering, **Walchand College of Engineering, Sangli**, and the **Indian Institute of Information Technology (IIIT) Dharwad**, India.

We thank the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB) for creating and publishing the CIC-Trap4Phish benchmark dataset.
