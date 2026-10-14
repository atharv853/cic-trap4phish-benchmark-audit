# Executive Research Summary: CIC-Trap4Phish 2025 Audit & Transfer Study

## Abstract
This study evaluates the static feature representation of the CIC-Trap4Phish 2025 dataset across Word, Excel, PDF, and HTML attachments. We investigate data provenance artifacts, shortcut features, and the viability of format-agnostic universal feature models versus per-format specialists.

## Key Research Answers (Q1 - Q3)

### Q1. Provenance Artifacts and Shortcuts
- **Word**: 30 features are missing in 50% of samples. Presence/absence of these missing columns perfectly separates benign and malicious labels with 100.0% accuracy.
- **Excel**: Removing all macro and API features yields F1 = 1.0000. Rows in duplicate groups: 6591 (6028 removed by deduplication).
- **Twin Leakage**: In standard L0 stratified splits, test twin leakage rates are Word (20.8%), Excel (32.9%), PDF (27.8%), HTML (27.1%).

### Q2. Honest Performance (L1 & L2 Levels)
- After exact deduplication (L1) and removal of single-feature shortcuts (L2), Word retains detectability with L2 F1 = 0.9366 (clean feature set F1 = 0.9366).
- Excel retains detectability with L2 F1 = 0.9989.
- HTML exhibits a test F1 of 0.9276 on honest deduplicated splits.

### Q3. Universal Feature Transfer Gap
- **Word Transfer Gap**: Delta F1 = 0.6319 (95% CI [0.6085, 0.6540])
- **Excel Transfer Gap**: Delta F1 = 0.0000 (95% CI [-0.0015, 0.0015])
- **PDF Transfer Gap**: Delta F1 = 0.0093 (95% CI [0.0059, 0.0131])
- **HTML Transfer Gap**: Delta F1 = 0.0731 (95% CI [0.0605, 0.0853])
- **HTML Sensitivity Across k**:
  - k=4: Delta F1 = 0.0696 (95% CI [0.0601, 0.0854])
  - k=6: Delta F1 = 0.0553 (95% CI [0.0469, 0.0707])
  - k=8: Delta F1 = 0.0490 (95% CI [0.0410, 0.0631])
  - k=10: Delta F1 = 0.0384 (95% CI [0.0307, 0.0533])

## Manifest
Generated automatically with seed 42.
