# Research Methodology

**Project:** Cross-Lingual Calibration Collapse Under Instruction-Tuning: Verbalized vs. Statistical Confidence, Answer Consistency and Mitigation  
**Model:** Google Gemma 3 4B Instruct (`google/gemma-3-4b-it`, NF4 4-bit Quantization)  
**Execution Environment:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.x, Windows 11  
**Methodological Specification:** Frozen Specification v4.5  

---

## 1. Experimental Design & Question Partitions

The benchmark dataset comprises $N = 150$ curated factual knowledge questions spanning Geography, History, Science, and Culture, drawn from multilingual resources (Belebele, IndicQA, and curated high-evidence sources). Each question contains:
1. `text_en`: Original English formulation.
2. `ground_truth_claims`: Atomic factual claims defining truth conditions.
3. `reference_answer`: Canonical ground-truth answer.
4. `evidence_passage`: Ground-truth context enabling strict verification.

The 150 questions are partitioned strictly according to Frozen Specification v4.5:
- **$P_{\text{cal}}$ (Calibration Partition, $N=75$):** Used strictly for fitting post-hoc calibration models (Platt scaling, Language-Shift Platt).
- **$P_{\text{val}}$ (Validation Partition, $N=30$):** Used strictly for hyperparameter tuning and mitigation strategy selection.
- **$P_{\text{test}}$ (Held-out Test Partition, $N=45$):** Held strictly unseen during calibration and strategy selection, used solely for final unbiased empirical evaluation.

---

## 2. Experimental Conditions

The benchmark exercises 11 distinct experimental conditions across 5 languages:
- **English ($L_0$):** `en_direct` (direct English prompting).
- **Indic Native ($L_1 \dots L_4$):**
  - `hi_native` (Hindi in Devanagari script)
  - `bn_native` (Bengali in Bengali script)
  - `te_native` (Telugu in Telugu script)
  - `ta_native` (Tamil in Tamil script)
- **Cross-Lingual Transferred ($L_1 \dots L_4 \to L_0$):**
  - `hi_trans_en` (Hindi translated from English)
  - `bn_trans_en` (Bengali translated from English)
  - `te_trans_en` (Telugu translated from English)
  - `ta_trans_en` (Tamil translated from English)
- **Bidirectional Inversion Controls:**
  - `en_trans_hi` (English translated from Hindi)
  - `en_trans_ta` (English translated from Tamil)

Total baseline inference workload: $150 \text{ questions} \times 11 \text{ conditions} = 1,650$ generated answers, $1,650$ verbalized confidence elicitations, and $750$ symmetric pairwise consistency judge evaluations ($150 \times 5 \text{ pairs}$), totaling $4,050$ real GPU model calls.

---

## 3. Confidence Metrics & Mathematical Formulations

### 3.1 Verbalized Confidence ($C_{\text{verb}}$)
Elicited directly via an isolated prompt instructing the model to evaluate the factual truth of its candidate answer:
$$\text{Prompt: } \text{"Provide your confidence as a single percentage between 0\% and 100\%."}$$
The output is parsed under strict numerical extraction and normalized to $[0.0, 1.0]$.

### 3.2 Statistical Confidence ($C_{\text{stat}}$)
Calculated from the model's auto-regressive generation token probabilities. For generated content tokens $t_1, t_2, \dots, t_K$ (excluding special BOS/EOS tokens):
$$C_{\text{stat}} = \exp\left( \frac{1}{K} \sum_{k=1}^K \ln P(t_k \mid t_{<k}, x) \right)$$
representing the sequence geometric mean probability.

### 3.3 Expected Calibration Error (ECE)
To avoid empty-bin artifacts under skewed verbalized confidence distributions, we utilize **Equal-Frequency Quantile Binning** with $M = 5$ bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where each bin $B_m$ contains exactly $N / M$ samples.

### 3.4 Brier Score
Strict proper scoring rule measuring mean squared probability error:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (c_i - y_i)^2$$
where $y_i \in \{0, 1\}$ is strict correctness and $c_i \in [0, 1]$ is the assessed confidence.

---

## 4. Post-Hoc Recalibration Models

To address cross-lingual miscalibration, three logistic calibration models are fitted on $P_{\text{cal}}$:

### Model 1: Global Platt Scaling
Single global affine transformation on logit-transformed verbalized confidence $\tilde{z} = \text{logit}(c_v)$:
$$p_1 = \sigma(w \cdot \tilde{z} + b_0)$$
where $\sigma(u) = \frac{1}{1 + e^{-u}}$.

### Model 2: Language-Shift Platt Scaling (Selected Production Model)
Shared discrimination slope $w$ with language-specific intercept shifts $\delta_l$ capturing language competence differentials:
$$p_2(l) = \sigma(w \cdot \tilde{z} + b_0 + \delta_l), \quad \text{with } \delta_{\text{en}} = 0$$
Fitted via L-BFGS-B with $L_2$ regularization $\frac{\lambda}{2} (w^2 + \sum_l \delta_l^2)$.

### Model 3: Language-Specific Platt Scaling (Exploratory Benchmark)
Fully independent slope and intercept per language condition:
$$p_3(l) = \sigma(w_l \cdot \tilde{z} + b_l)$$

---

## 5. Mitigation Strategies (A through F)

1. **Strategy A: Native Prompt Optimization**
   - Implements native script meta-prompts enforcing conciseness, entity preservation, and factual verification.
2. **Strategy B: Cross-Lingual Pivot Answering**
   - Generates reasoning and candidate answers in high-competence English ($95.6\%$ baseline accuracy), then translates the verified output into the target Indic language.
3. **Strategy C: Bilingual Verification & Arbitration**
   - Generates dual candidates (direct native and English pivot), executes model arbitration to resolve factual discrepancies.
4. **Strategy D: Evidence-Grounded Refinement**
   - Compares initial draft against retrieved evidence premises and self-corrects ungrounded entity names or dates.
5. **Strategy E: Confidence-Aware Selective Answering**
   - Applies calibrated confidence gating: if $p_2(l) < \tau$, the model abstains from answering, providing explicit abstention rationale. Sweeps $\tau \in [0.10, 0.90]$ to characterize the accuracy-coverage Pareto front.
6. **Strategy F: Hybrid Pivot + Calibration Pipeline**
   - Integrates Strategy B (answer generation) with Model 2 Recalibration and Strategy E gating.
