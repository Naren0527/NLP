# Empirical Metrics Summary (Phase 5 Full-Scale GPU Run)

**Model:** `google/gemma-3-4b-it` (NF4 4-bit) on NVIDIA RTX 4050 Laptop GPU  
**Dataset:** 150 questions ($P_{cal}=75, P_{val}=30, P_{test}=45$) across 11 conditions (4,050 model inferences)  
**Selected Recalibration Model:** `Model_2_Language_Shift_Platt`

## 1. Model Selection Validation Brier Score ($P_{val}$)
- **Verbalized Track:**
  - Model 1 (Pooled Platt): `0.2182`
  - Model 2 (Language-Shift Platt): `0.1215` (**Selected**, 44.3% improvement)
- **Statistical Track:**
  - Model 1 (Pooled Platt): `0.1361`
  - Model 2 (Language-Shift Platt): `0.1224` (**Selected**, 10.1% improvement)

## 2. Condition-Wise Held-Out Test Evaluation ($P_{test}$, $N=45$ per condition)

| Condition | Test N | Accuracy | Mean Verb Conf | Uncal ECE (Verb) | Recal ECE (Verb) | Uncal BS (Verb) | Recal BS (Verb) | Uncal ECE (Stat) | Recal ECE (Stat) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `en` | 45 | 0.956 | 0.984 | 0.0440 | 0.1062 | 0.0440 | 0.0548 | 0.0634 | 0.0933 |
| `hi` | 42 | 0.238 | 0.918 | 0.6798 | 0.1043 | 0.6623 | 0.1885 | 0.5771 | 0.1107 |
| `bn` | 45 | 0.356 | 0.914 | 0.6049 | 0.2126 | 0.6463 | 0.2714 | 0.5123 | 0.2664 |
| `te` | 12 | 0.250 | 0.953 | 0.7025 | 0.2894 | 0.6692 | 0.2987 | 0.5999 | 0.2613 |
| `ta` | 45 | 0.378 | 0.916 | 0.5516 | 0.0676 | 0.5791 | 0.2501 | 0.4120 | 0.0985 |
| `hi_trans_en` | 45 | 0.956 | 0.944 | 0.0113 | 0.0653 | 0.0009 | 0.0326 | 0.0603 | 0.0692 |
| `bn_trans_en` | 45 | 0.956 | 0.944 | 0.0113 | 0.0653 | 0.0009 | 0.0326 | 0.0603 | 0.0692 |
| `te_trans_en` | 45 | 0.956 | 0.944 | 0.0113 | 0.0653 | 0.0009 | 0.0326 | 0.0603 | 0.0692 |
| `ta_trans_en` | 45 | 0.956 | 0.944 | 0.0113 | 0.0653 | 0.0009 | 0.0326 | 0.0603 | 0.0692 |
| `en_trans_hi` | 43 | 0.116 | 0.919 | 0.8023 | 0.1472 | 0.7828 | 0.1288 | 0.7259 | 0.1112 |
| `en_trans_ta` | 44 | 0.364 | 0.925 | 0.5614 | 0.0615 | 0.5870 | 0.2370 | 0.4399 | 0.1541 |

## 3. Confirmatory Hypothesis Testing ($B=10,000$ Bootstrap Replicates)
Testing $\Delta \text{ECE}_l = \text{ECE}_l - \text{ECE}_{en} > 0$ on held-out test data:

| Language | $\Delta \text{ECE}$ vs English | Level A 99.0% Bootstrap CI | Level B 99.75% Bootstrap CI | Confirmatory Significant |
| :--- | :---: | :---: | :---: | :---: |
| `hi` | **+0.6358** | [+0.4762, +0.7920] | [+0.4509, +0.8202] | **True** |
| `bn` | **+0.5609** | [+0.3838, +0.7300] | [+0.3604, +0.7531] | **True** |
| `te` | **+0.6585** | [+0.3890, +0.9296] | [+0.3015, +0.9453] | **True** |
| `ta` | **+0.4947** | [+0.3385, +0.6689] | [+0.3126, +0.7025] | **True** |
