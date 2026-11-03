"""
Mitigation Experiment Runner.
Executes systematic benchmarking of Baseline vs. Strategies A, B, C, D, E.
Selects optimal strategy strictly on P_val and evaluates held-out performance on P_test.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.model_engine import ModelEngine
from src.confidence import ConfidenceEngine
from src.calibration.recalibration import RecalibrationEngine
from src.calibration.metrics import compute_quantile_ece
from src.mitigation.strategies import (
    StrategyA_NativePromptOptimization,
    StrategyB_CrossLingualPivotAnswering,
    StrategyC_BilingualVerification,
    StrategyD_EvidenceGroundedRefinement,
    StrategyE_SelectiveAnswering,
    MitigationOutput,
    LANG_CONFIGS,
)


class MitigationExperimentRunner:
    """Orchestrates multi-strategy mitigation evaluation and validation selection."""

    def __init__(
        self,
        manifest_path: str = "data/full_150_questions.jsonl",
        output_dir: str = "outputs/mitigation_results",
        model_engine: Optional[ModelEngine] = None,
    ):
        self.manifest_path = manifest_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_engine = model_engine or ModelEngine.get_instance()
        self.confidence_engine = ConfidenceEngine(model_engine=self.model_engine)

        self.strategy_a = StrategyA_NativePromptOptimization(self.model_engine, self.confidence_engine)
        self.strategy_b = StrategyB_CrossLingualPivotAnswering(self.model_engine, self.confidence_engine)
        self.strategy_c = StrategyC_BilingualVerification(self.model_engine, self.confidence_engine)
        self.strategy_d = StrategyD_EvidenceGroundedRefinement(self.model_engine, self.confidence_engine)

        self.strategies = {
            "Strategy_A": self.strategy_a,
            "Strategy_B": self.strategy_b,
            "Strategy_C": self.strategy_c,
            "Strategy_D": self.strategy_d,
        }

        # Load questions
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.questions = [json.loads(line) for line in f if line.strip()]

        self.cal_questions = [q for q in self.questions if q["partition"] == "cal"]
        self.val_questions = [q for q in self.questions if q["partition"] == "val"]
        self.test_questions = [q for q in self.questions if q["partition"] == "test"]

    def run_strategy_on_partition(
        self,
        strategy_key: str,
        partition_name: str,
        questions: List[Dict[str, Any]],
        languages: List[str] = ["en", "hi", "bn", "te", "ta"],
    ) -> List[Dict[str, Any]]:
        """Executes a single strategy across a question partition."""
        strat = self.strategies[strategy_key]
        records_file = self.output_dir / f"{strategy_key}_{partition_name}_records.jsonl"

        completed = {}
        if records_file.exists():
            with open(records_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        completed[(r["question_id"], r["target_language"])] = r

        print(f"[{strategy_key} on {partition_name}] Loaded {len(completed)} existing records.")

        results = list(completed.values())
        with open(records_file, "a", encoding="utf-8") as f_out:
            for q in questions:
                qid = q["question_id"]
                for lang in languages:
                    key = (qid, lang)
                    if key in completed:
                        continue

                    out = strat.execute(q, lang)
                    row = {
                        "strategy": strategy_key,
                        "partition": partition_name,
                        "question_id": qid,
                        "domain": q["domain"],
                        "target_language": lang,
                        "final_answer": out.final_answer,
                        "raw_confidence": out.raw_confidence,
                        "calibrated_confidence": out.calibrated_confidence,
                        "statistical_confidence": out.statistical_confidence,
                        "is_correct": out.is_correct,
                        "latency_seconds": out.latency_seconds,
                        "intermediate_candidates": out.intermediate_candidates,
                        "verification_details": out.verification_details,
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_out.flush()
                    completed[key] = row
                    results.append(row)
        return results

    def select_best_mitigation_on_validation(
        self,
        languages: List[str] = ["hi", "bn", "te", "ta"],
    ) -> Dict[str, Any]:
        """
        Evaluates all strategies on P_val to choose the best mitigation strategy.
        Selection criterion: Highest Validation Accuracy, tie-broken by lowest Brier Score.
        """
        val_summary = {}
        for strat_key in self.strategies.keys():
            recs = self.run_strategy_on_partition(strat_key, "val", self.val_questions, languages)
            acc = float(np.mean([r["is_correct"] for r in recs]))
            # Brier Score against raw confidence
            confs = np.array([r["raw_confidence"] for r in recs])
            ys = np.array([1.0 if r["is_correct"] else 0.0 for r in recs])
            bs = float(np.mean((confs - ys)**2))
            val_summary[strat_key] = {
                "validation_accuracy": round(acc, 4),
                "validation_brier_score": round(bs, 4),
                "sample_count": len(recs),
            }

        # Find best
        best_strat = max(val_summary.keys(), key=lambda k: (val_summary[k]["validation_accuracy"], -val_summary[k]["validation_brier_score"]))
        return {
            "validation_scores": val_summary,
            "selected_best_strategy": best_strat,
        }
