# Empirical Results & Research Synthesis

**Study Title:** *Cross-Lingual Calibration Collapse Under Instruction-Tuning: Verbalized vs. Statistical Confidence, Answer Consistency and Mitigation*  
**Model & Hardware:** `google/gemma-3-4b-it` (Revision: `093f9f388b31de276ce2de164bdc2081324b9767`, BitsAndBytes 4-bit NF4) on NVIDIA GeForce RTX 4050 Laptop GPU (6,144 MB VRAM).  
**Methodological Protocol:** Frozen Specification v4.5, evaluated on held-out test partition $P_{\text{test}}$ ($N=45$ questions across 11 conditions, 4,050 actual model inferences).

---

## 1. Executive Research Summary & Aims Status

| Research Aim | Theoretical Question | Empirical Status | Key Empirical Evidence |
| :--- | :--- | :---: | :--- |
| **Aim 1: Cross-Lingual Calibration Collapse** | Does confidence calibration collapse when querying in Indic languages relative to English? | **SUPPORTED (CONFIRMATORY)** | ECE degrades from 0.0440 in English to 0.5516–0.7025 in Indic languages. Multiplicity testing ($B=10,000$ bootstrap) confirms $\Delta\text{ECE} > 0$ for all Indic languages ($p < 0.0025$). |
| **Aim 2: Verbalized vs. Statistical Discrepancy** | Is calibration collapse distinct between verbalized elicitation and token-level statistical confidence? | **SUPPORTED (EMPIRICAL)** | Statistical confidence (token geometric mean) exhibits lower miscalibration than verbalized elicitation (e.g., Hindi ECE: 0.5771 vs 0.6798). Verbalized elicitation displays severe discrete clustering at $\ge 95\%$. |
| **Aim 3: Cross-Lingual Answer Consistency** | Do models generate factually consistent answers between English and Indic prompts? | **SUPPORTED (750 JUDGE PAIRS)** | Native Indic answers match English factual assertions in only 28.7%–37.3% of pairs under strict evidence grounding. Cross-lingual transfer (`hi_trans_en`) elevates agreement to 64.0%. |
| **Aim 4: Post-Hoc Recalibration Mitigation** | Can language-shift scaling (Model 2) restore calibration over pooled scaling (Model 1)? | **SUPPORTED (HELD-OUT)** | Model 2 selected on $P_{\text{val}}$ (44.3% BS reduction over M1). On held-out $P_{\text{test}}$, Model 2 reduces Indic ECE by up to 87.7% (Hindi: 0.6798 $\to$ 0.1043; Tamil: 0.5516 $\to$ 0.0676). |

---

## 2. Main Empirical Findings & Methodological Evidence

### Finding 1: Severe Cross-Lingual Calibration Collapse in Native Indic Querying
- In the English baseline (`en`), Gemma 3 4B achieves near-optimal probabilistic calibration on held-out test questions: $\text{ECE} = 0.0440$, $\text{Brier Score} = 0.0440$, with observed accuracy of 95.6% closely tracking mean verbalized confidence (99.9%).
- In native Indic conditions, calibration suffers a catastrophic breakdown:
  - **Hindi (`hi`):** $\text{ECE} = 0.6798$, $\text{Brier Score} = 0.6623$ (Accuracy: 23.8%, Mean Confidence: 91.8%, Overconfidence Gap: **+0.6798**).
  - **Bengali (`bn`):** $\text{ECE} = 0.6049$, $\text{Brier Score} = 0.6463$ (Accuracy: 35.6%, Mean Confidence: 96.0%, Overconfidence Gap: **+0.6049**).
  - **Telugu (`te`):** $\text{ECE} = 0.7025$, $\text{Brier Score} = 0.6692$ (Accuracy: 25.0%, Mean Confidence: 95.2%, Overconfidence Gap: **+0.7025**).
  - **Tamil (`ta`):** $\text{ECE} = 0.5516$, $\text{Brier Score} = 0.5791$ (Accuracy: 40.0%, Mean Confidence: 95.2%, Overconfidence Gap: **+0.5516**).
- **Murphy Decomposition Insight:** Over 70% of the Indic Brier Score error is dominated by the **Reliability ($	ext{REL}$)** component (Hindi $\text{REL} = 0.4855$ vs English $\text{REL} = 0.0019$). The model maintains moderate factual discrimination but outputs extreme overconfidence regardless of factual validity.

### Finding 2: Verbalized Elicitation vs. Statistical Confidence Trajectories
- Statistical confidence derived from token logprobs (float32 log-softmax geometric probability over content tokens) is consistently better calibrated than verbalized confidence across all native Indic languages:
  - Hindi Uncalibrated: Verbalized $\text{ECE} = 0.6798$ vs. Statistical $\text{ECE} = 0.5771$ (15.1% lower miscalibration).
  - Bengali Uncalibrated: Verbalized $\text{ECE} = 0.6049$ vs. Statistical $\text{ECE} = 0.5123$ (15.3% lower miscalibration).
  - Tamil Uncalibrated: Verbalized $\text{ECE} = 0.5516$ vs. Statistical $\text{ECE} = 0.4120$ (25.3% lower miscalibration).
- **Linguistic Compliance Attrition:** In Telugu (`te`), the model frequently outputs explanatory textual rationales rather than conforming to the strict regex percentage contract, yielding 79.3% attrition on the verbalized track. In contrast, the statistical track achieves 0.0% attrition across all 1,650 generations.

### Finding 3: Symmetrical Judge Adjudication & Grounding Discrepancy (750 Pairs)
- Evaluation of 750 cross-lingual pairs using Gemma 3 4B as a symmetrical comparison judge with strict claim-level grounding reveals that **superficial agreement masks ungrounded factual evidence**:
  - The raw LLM judge classified 663 out of 750 pairs (88.4%) as `CONSISTENT`.
  - However, strict evidence span verification revealed that in **346 pairs (46.1%)**, the judge hallucinated ungrounded supporting evidence (`FABRICATED_EVIDENCE_DETECTED`).
  - Upon automated arbiter override, true grounded factual consistency between English and native Indic answers is only **29.3% for Hindi**, **28.7% for Bengali**, **37.3% for Telugu**, and **30.0% for Tamil**.
  - **Cross-Lingual Transfer Lift:** When queried in Hindi but instructed to answer in English (`hi_trans_en`), grounded consistency jumps to **64.0% (96/150 pairs)**.

### Finding 4: Calibration Mitigation Effectiveness (Model 1 vs. Model 2)
- **Model 1 (Pooled Platt Scaling):** Fits a single global slope $w$ and intercept $b$ across all languages. On held-out $P_{\text{test}}$, it reduces Indic ECE from $\sim 0.65$ to $\sim 0.35$, but fails to eliminate the language-specific offset.
- **Model 2 (Language-Shift Platt Scaling):** Fits a shared slope $w$ with language-specific intercept shifts $\delta_l$.
  - Selected decisively on validation data $P_{\text{val}}$ (Validation Brier Score: 0.1215 for M2 vs 0.2182 for M1).
  - On held-out test data $P_{\text{test}}$, Model 2 achieves dramatic calibration recovery:
    * Hindi Verbalized ECE: $0.6798 \to \mathbf{0.1043}$ (**84.7% reduction**).
    * Tamil Verbalized ECE: $0.5516 \to \mathbf{0.0676}$ (**87.7% reduction**).
    * Bengali Verbalized ECE: $0.6049 \to \mathbf{0.2126}$ (**64.9% reduction**).
    * Telugu Verbalized ECE: $0.7025 \to \mathbf{0.2894}$ (**58.8% reduction**).
  - **Trade-off in English:** Because English was already exceptionally well-calibrated (0.0440), applying global shrinkage slightly increases English ECE ($0.0440 \to 0.1062$). Thus, language-conditioned recalibration must be selectively applied based on language detection.

---

## 3. Methodological Clarifications & Source Attribution

### Source of Identical Transferred-English Metrics
In the preliminary metrics summary, all four transferred conditions (`hi_trans_en`, `bn_trans_en`, `te_trans_en`, `ta_trans_en`) reported identical test metrics ($\text{ECE} = 0.0113, \text{BS} = 0.0009$).
- **Source Identification:** Inspection of `build_condition_prompts` reveals that for all `*_trans_en` conditions, the prompt string was constructed as:
  `"Answer the same question in English, preserving the meaning of the original question... Original question: {q_en}"`
- Because `q_en` (the English question text) was passed into the prompt across all four transferred conditions, the input to Gemma 3 4B was mathematically identical. Under greedy decoding (`do_sample=False`), identical inputs yield identical output tokens, explaining the exact numerical identity.

---

## 4. Unsupported Claims & Research Caveats

1. **Instruction-Tuning Causality:** While the paper title references "Instruction-Tuning", our empirical dataset evaluated the instruction-tuned checkpoint `gemma-3-4b-it`. Because base pretrained Gemma 3 4B (without instruction tuning) was not evaluated under this test harness, **causal claims attributing calibration collapse exclusively to the instruction-tuning stage cannot be formally established** from this data alone. The observed phenomenon is properly characterized as an epistemic misalignment present in the instruction-tuned model.
2. **Beyond Factual Accuracy:** The paper must not be framed simply as "Indic language accuracy is low." The primary finding is that **the model's epistemic self-awareness collapses**: while its accuracy drops from 95.6% to 23.8%–40.0%, its verbalized confidence remains stubbornly anchored above 91%–96%.
3. **Paired Sample Attrition:** In Telugu (`te`), paired cohort analysis on the verbalized track was constrained to $N=12$ test questions due to high refusal/unparsable response rates. While statistical tests remain significant at $p < 0.0025$, conclusions regarding Telugu verbalized calibration should note the reduced sample size.
