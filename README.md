# Cross-Lingual Confidence Calibration in Multilingual LLMs

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Model: Gemma-3-4B-IT](https://img.shields.io/badge/model-gemma--3--4b--it-orange.svg)](https://huggingface.co/google/gemma-3-4b-it)
[![Hardware: NVIDIA RTX 4050](https://img.shields.io/badge/GPU-RTX%204050%20(6GB)-green.svg)](https://www.nvidia.com)
[![Quantization: NF4](https://img.shields.io/badge/quantization-bitsandbytes%204bit%20NF4-purple.svg)](https://github.com/bitsandbytes-foundation/bitsandbytes)
[![Tests: 132 Passed](https://img.shields.io/badge/tests-132%20passed-brightgreen.svg)](tests/)

An empirical research framework and full-scale experimental benchmark investigating **cross-lingual confidence calibration, overconfidence degradation, and post-hoc recalibration** in modern open multilingual Large Language Models.

---

## 🔬 Research Overview

While multilingual Large Language Models (LLMs) demonstrate remarkable factual competence across high-resource languages such as English, their **internal epistemic confidence calibration** degrades severely when queried in lower-resource and morphologically diverse languages. This repository contains the complete empirical codebase, datasets, and experiment ledgers investigating:

1. **Epistemic Discrepancy:** Does verbalized and statistical confidence calibration degrade significantly in Indic languages (Hindi, Bengali, Telugu, Tamil) compared to English?
2. **Murphy Decomposition of Miscalibration:** Is the cross-lingual error gap driven by lack of resolution ($\text{RES}$) or severe overconfidence/reliability degradation ($\text{REL}$)?
3. **Post-Hoc Recalibration:** Can parameter-efficient Language-Shift Platt Scaling (Model 2) restore probabilistic calibration on held-out test data without task-specific fine-tuning?

---

## 📊 Key Empirical Findings (Phase 5 Full-Scale GPU Run)

All empirical evaluations were executed on local dedicated GPU hardware with zero simulation or synthetic generation:
- **Testbed Model:** `google/gemma-3-4b-it` (Revision: `093f9f388b31de276ce2de164bdc2081324b9767`, BitsAndBytes 4-bit NF4).
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,144 MB VRAM, CUDA 13.3).
- **Total Workload:** Exactly **4,050 actual model inferences** (1,650 answers, 1,650 verbalized confidences, 750 cross-lingual symmetrical comparisons).

### 1. Held-Out Test Evaluation ($P_{\text{test}}$, 45 Questions)
| Condition | Description | Uncalibrated ECE (Verb) | Recalibrated ECE (Verb) | Uncalibrated Brier Score | Recalibrated Brier Score | ECE Relative Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `en` | English Baseline | **0.0440** | 0.1062 | **0.0440** | 0.0548 | Baseline |
| `hi` | Hindi Direct | 0.6798 | **0.1043** | 0.6623 | **0.1885** | **-84.7%** |
| `bn` | Bengali Direct | 0.6049 | **0.2126** | 0.6463 | **0.2714** | **-64.9%** |
| `te` | Telugu Direct | 0.7025 | **0.2894** | 0.6692 | **0.2987** | **-58.8%** |
| `ta` | Tamil Direct | 0.5516 | **0.0676** | 0.5791 | **0.2501** | **-87.7%** |
| `hi_trans_en` | Hindi $\to$ English Transferred | **0.0113** | 0.0653 | **0.0009** | 0.0326 | High Alignment |
| `bn_trans_en` | Bengali $\to$ English Transferred | **0.0113** | 0.0653 | **0.0009** | 0.0326 | High Alignment |
| `te_trans_en` | Telugu $\to$ English Transferred | **0.0113** | 0.0653 | **0.0009** | 0.0326 | High Alignment |
| `ta_trans_en` | Tamil $\to$ English Transferred | **0.0113** | 0.0653 | **0.0009** | 0.0326 | High Alignment |
| `en_trans_hi` | Eng $\to$ Hindi Trans-Aided | 0.6860 | **0.1105** | 0.6826 | **0.1654** | **-83.9%** |
| `en_trans_ta` | Eng $\to$ Tamil Trans-Aided | 0.5570 | **0.1557** | 0.5699 | **0.2319** | **-72.0%** |

### 2. Confirmatory Multiplicity-Controlled Hypothesis Tests ($B = 10,000$ Bootstrap)
Testing $\Delta \text{ECE}_l = \text{ECE}_l - \text{ECE}_{\text{en}} > 0$ on held-out test data:
- **Level A (Within-Track):** $K = 5$, $\alpha_{\text{comp}} = 0.01$, 99.0% Confidence Intervals
- **Level B (Global Across-Track):** $K = 20$, $\alpha_{\text{comp}} = 0.0025$, 99.75% Confidence Intervals

| Language Comparison | Observed $\Delta \text{ECE}$ | Level A 99.0% CI | Level B 99.75% CI | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: |
| **`hi` vs `en`** | **+0.6358** | $[+0.4762, +0.7920]$ | $[+0.4509, +0.8202]$ | **$p < 0.0025$ (Confirmatory Significant)** |
| **`bn` vs `en`** | **+0.5609** | $[+0.3838, +0.7300]$ | $[+0.3604, +0.7531]$ | **$p < 0.0025$ (Confirmatory Significant)** |
| **`te` vs `en`** | **+0.6585** | $[+0.3890, +0.9296]$ | $[+0.3015, +0.9453]$ | **$p < 0.0025$ (Confirmatory Significant)** |
| **`ta` vs `en`** | **+0.4947** | $[+0.3385, +0.6689]$ | $[+0.3126, +0.7025]$ | **$p < 0.0025$ (Confirmatory Significant)** |

All lower confidence bounds are strictly positive, confirming significant cross-lingual calibration degradation across all Indic languages.

---

## 📁 Repository Structure

```
cross_lingual_calibration/
├── configs/
│   ├── model.yaml              # Gemma 3 4B NF4 quantization and generation parameters
│   └── prompts.yaml            # Localized condition prompt templates & regex contracts
├── data/
│   └── full_150_questions.jsonl# 150 curated benchmark questions (75 Cal, 30 Val, 45 Test)
├── results/
│   ├── empirical/              # Full-scale Phase 5 empirical GPU run artifacts
│   │   ├── model_call_logs.jsonl              # Granular log of all 4,050 GPU calls
│   │   ├── phase5_empirical_records.jsonl     # Complete 1,650 empirical answer records
│   │   ├── phase5_empirical_comparisons.jsonl # 750 symmetrical judge comparison records
│   │   ├── phase5_final_empirical_report.json # Primary machine-readable report
│   │   ├── phase5_qc_audit_manifest.json      # 10% Hamilton-Hare QC sample manifest
│   │   └── progress_status.json               # Live progress state ledger
│   ├── full_scale/             # Preserved simulated reference baseline
│   └── pilot/                  # 30-question pilot diagnostics
├── scripts/
│   ├── run_phase_5_empirical_full_scale.py   # Full-scale empirical execution driver
│   ├── validate_recovery_resume.py           # Minimal recovery and resume validator
│   ├── smoke_test_judge_inference.py         # Judge GPU execution preflight test
│   ├── smoke_test_checkpoint_resume.py       # Checkpoint resumption smoke test
│   └── generate_full_dataset.py              # Benchmark dataset builder
├── src/
│   ├── calibration/
│   │   ├── empirical_driver.py # End-to-end inference driver across all 11 conditions
│   │   ├── eligibility.py      # Conjunction eligibility & Table A/B attrition calculators
│   │   ├── instrumentation.py  # ModelCallLogger and invocation accounting validator
│   │   ├── metrics.py          # ECE, MCE, RMSCE, Brier Score & Murphy Decomposition
│   │   ├── recalibration.py    # Standardizer & Recalibration Models 1, 2, 3, 4
│   │   ├── schemas.py          # Atomic claims, ground-truth verification schemas
│   │   └── statistical_tests.py# Hamilton-Hare QC sampling & Cluster-Bootstrap CIs
│   ├── answer_comparator.py    # Gemma-based cross-lingual factual judge
│   ├── confidence.py           # Dual statistical confidence & verbalized regex parsing
│   ├── model_engine.py         # Singleton GPU inference engine with logprob extraction
│   └── storage.py              # Thread-safe JSONL atomic storage
├── tests/
│   ├── calibration/
│   │   ├── test_calibration_framework.py # 128-test calibration & statistical suite
│   │   └── test_empirical_driver.py      # Empirical pipeline integration tests
│   ├── test_answer_comparator.py         # Symmetrical judge unit tests
│   ├── test_confidence.py                # Regex parser & confidence unit tests
│   └── test_model_engine.py              # ModelEngine instantiation & tokenizer tests
├── pytest.ini
├── requirements.txt
└── .env.example
```

---

## 🛠️ Installation & Reproduction

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU with $\ge$ 6 GB VRAM (tested on NVIDIA GeForce RTX 4050 Laptop GPU)
- Hugging Face account with access to `google/gemma-3-4b-it`

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Naren0527/NLP.git
cd NLP

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Edit `.env` and set your read-only Hugging Face access token:
```env
HF_TOKEN=hf_your_token_here
```

### 3. Run Regression Test Suite
Run the 132-test regression test suite verifying calibration metrics, Murphy decomposition invariants, regex contracts, and bootstrap algorithms:
```bash
pytest -q
```
*Expected result:* `132 passed`.

### 4. Run Full-Scale Empirical Pipeline
To reproduce the full-scale Phase 5 empirical run (150 questions $\times$ 11 conditions = 4,050 model inferences):
```bash
python -u scripts/run_phase_5_empirical_full_scale.py \
    --manifest data/full_150_questions.jsonl \
    --output_dir results/empirical \
    --seed 20260330
```
Execution supports atomic checkpointing and will automatically resume from the last completed record if interrupted.

### 5. Launch Interactive Multilingual UI
Launch the comprehensive 7-page interactive research application:
```bash
streamlit run app.py
```
**Interactive UI Capabilities:**
- **Page 1 (Research Overview):** Core research questions, pipeline diagrams, executive summaries.
- **Page 2 (Live Multilingual Demo):** Interactive prompt playground across English, Hindi, Bengali, Telugu, and Tamil with intermediate candidate inspection (English pivot, fluent translation, arbitration notes) and Model 2 Platt scaling.
- **Page 3 (Baseline vs. Mitigation):** Real-time comparative tables, accuracy gains (+46.1 pp), and relative error reduction.
- **Page 4 (Calibration Analysis):** Equal-frequency quantile reliability diagrams, ECE curves, and Language-Shift Platt parameter shifts ($\delta_l$).
- **Page 5 (Cross-Lingual Consistency):** 750 pairwise consistency matrix, LLM judge agreement (88.4%) vs grounded factual evidence consistency (28.7% - 37.3%).
- **Page 6 (Strategy Experiments & Selective Answering):** Validation selection on $P_{\text{val}}$ ($N=30$) and Strategy E Accuracy-Coverage Pareto curve with interactive $\tau$ slider.
- **Page 7 (Research Report & Downloads):** Publication-ready paper viewer and one-click data/metrics download center.

### 6. Run Mitigation Benchmark
To benchmark the mitigation strategies on held-out test data and regenerate all 14 publication figures:
```bash
python -u scripts/run_mitigation_benchmark.py
```

---

## 📚 Complete Research Documentation

Full formal documentation is organized in `docs/`:
- **[`docs/final_research_report.md`](docs/final_research_report.md):** Complete publication-grade research paper with abstract, introduction, methodology, empirical results, and discussion.
- **[`docs/methodology.md`](docs/methodology.md):** Methodological specification, dataset partitioning ($P_{\text{cal}}=75, P_{\text{val}}=30, P_{\text{test}}=45$), prompt templates, and calibration formulations.
- **[`docs/results.md`](docs/results.md):** Comprehensive empirical findings, baseline collapse tables, recalibration metrics, and mitigation gains.
- **[`docs/limitations.md`](docs/limitations.md):** Explicit scientific limitations regarding model scale, NF4 quantization, instruction-tuning causality caveats, and translation fidelity.

---

## 📜 Dataset & Licensing
- The benchmark questions in `data/full_150_questions.jsonl` are derived from the [Belebele](https://huggingface.co/datasets/facebook/belebele) benchmark (CC-BY-SA 4.0) and manually curated factual questions across Geography, History, and Science.
- Evaluation follows Frozen Methodological Specification v4.5.

---

## 📑 Citation
```bibtex
@article{crosslingual_calibration_2026,
  title={Cross-Lingual Confidence Calibration in Multilingual Large Language Models},
  author={Naren},
  year={2026},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/Naren0527/NLP}}
}
```
