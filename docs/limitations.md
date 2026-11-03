# Scientific Limitations & Scope

**Project:** Cross-Lingual Calibration Collapse Under Instruction-Tuning: Verbalized vs. Statistical Confidence, Answer Consistency and Mitigation  
**Primary Model:** Google Gemma 3 4B Instruct (`google/gemma-3-4b-it`)  
**Methodological Protocol:** Frozen Specification v4.5  

---

## 1. Model Scale & Family Specificity

Our empirical evaluations were conducted exclusively using **Google Gemma 3 4B Instruct** in 4-bit NF4 precision. While Gemma 3 4B is a state-of-the-art open-weights model representative of modern instruction-tuned architectures deployable on consumer and edge hardware (6 GB VRAM), the observed quantitative calibration degradation may vary across parameter scales:
- Larger parameter regimes (e.g., Gemma 3 12B/27B, LLaMA 3 70B) possess greater cross-lingual multilingual capacity that may alleviate raw accuracy drop, although literature suggests verbalized overconfidence persists across scale.
- Architectural differences (e.g., vocabulary size, tokenizer compression ratio for Indic scripts) influence token entropy and statistical confidence distributions.

---

## 2. Quantization Constraints (NF4 vs. FP16/BF16)

To enable rigorous local GPU execution on an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM), the model was loaded in **4-bit NormalFloat (NF4)** using BitsAndBytes quantization. 
- Prior literature indicates NF4 preserves $>98\%$ of full-precision benchmark performance.
- However, logit-level compression and dequantization overhead could introduce slight numerical perturbations in tail token probabilities, slightly shifting statistical confidence ($C_{\text{stat}}$) geometric means.
- Verbalized confidence ($C_{\text{verb}}$) relies on generated discrete tokens and is robust to minor logit shifts.

---

## 3. Causal Attribution Regarding Instruction-Tuning

> [!IMPORTANT]
> **Methodological Caveat on Causal Inference:**
> While our study observes severe verbalized miscalibration in instruction-tuned Gemma 3 4B-it, we do **not** claim formal causal proof that instruction-tuning is the sole isolated cause of cross-lingual calibration collapse.
> 
> Establishing strict causality would require evaluating a matched pre-trained base model (`google/gemma-3-4b`) under identical prompt constraints. Because base models cannot follow zero-shot verbalized confidence elicitation instructions without few-shot demonstrations, direct apples-to-apples verbalized comparison introduces confounding prompt effects. Our findings strictly characterize the empirical behavior of the instruction-tuned model.

---

## 4. Linguistic Coverage & Tokenization Asymmetry

The experimental scope focused on five languages: English (high-resource Latin script) and four major Indic languages (Hindi, Bengali, Telugu, Tamil) representing two distinct language families (Indo-Aryan and Dravidian) across three unique scripts (Devanagari, Bengali, Dravidian).
- Tokenizer efficiency varies dramatically across scripts: Indic words typically fragment into $2.5\times - 4.0\times$ more subword tokens than English equivalents.
- This fragmentation inflates token sequence length $K$, potentially depressing sequence-level statistical confidence ($C_{\text{stat}} = \exp(\frac{1}{K} \sum \ln p_i)$) compared to English.

---

## 5. Strict Propositional Verification vs. Semantic Nuance

Strict correctness was evaluated using atomic proposition claim matching against verified evidence spans:
- This ensures high reproducibility and prevents soft judge hallucination.
- However, highly verbose or poetically phrased valid native answers that express factual truth using alternate idioms or unlisted dialectal synonyms may occasionally be penalized by strict substring verification unless normalized.
- Mitigation Strategy B circumvents this by anchoring in English ground-truth entities before translation.
