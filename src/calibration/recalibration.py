"""
Phase 5 Post-Hoc Recalibration Engine.

Implements Section 10 of Phase 5 Methodological Specification v4.5:
1. Feature Standardization:
   z_i = logit(c_i + eps),  tilde{z}_i = (z_i - mu_{cal}) / sigma_{cal} (ddof=0)
2. Operational Candidates:
   - Model 1: Pooled Platt Scaling (w * tilde{z}_i + b)
   - Model 2: Language-Shift Platt Scaling (w * tilde{z}_i + b_0 + delta_l, with delta_en == 0)
3. Strictly Exploratory Benchmarks:
   - Model 3: Language-Specific Platt Scaling (w_l * tilde{z}_i + b_l)
     with Hessian weighting matrix W_l = diag(w_i * p_{i,l} * (1 - p_{i,l}))
   - Model 4: Isotonic Regression (PAVA)
4. Model Selection:
   Selection on P_val strictly between Model 1 and Model 2 minimizing Brier Score.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


@dataclass
class Standardizer:
    mu_cal: float = 0.0
    sigma_cal: float = 1.0
    epsilon: float = 1e-5
    is_fitted: bool = False

    def fit(self, confidences: np.ndarray) -> "Standardizer":
        conf = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
        # Logit transform with epsilon boundary protection
        logits = np.log((conf + self.epsilon) / (1.0 - conf + self.epsilon))
        self.mu_cal = float(np.mean(logits))
        self.sigma_cal = float(np.std(logits, ddof=0))
        if self.sigma_cal < 1e-6:
            self.sigma_cal = 1.0
        self.is_fitted = True
        return self

    def transform(self, confidences: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Standardizer must be fitted on calibration partition first.")
        conf = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
        logits = np.log((conf + self.epsilon) / (1.0 - conf + self.epsilon))
        return (logits - self.mu_cal) / self.sigma_cal


class Model1PooledPlatt:
    """
    Model 1: Pooled Platt Scaling (Operational Candidate).
    Global logistic calibration: p_hat = sigmoid(w * tilde{z} + b)
    L2 penalty: lambda / 2 * w^2 (lambda = 1.0)
    """

    def __init__(self, l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.w: float = 1.0
        self.b: float = 0.0
        self.is_fitted: bool = False

    def fit(self, z_tilde: np.ndarray, y: np.ndarray) -> "Model1PooledPlatt":
        z = np.asarray(z_tilde, dtype=np.float64)
        labels = np.asarray(y, dtype=np.float64)

        def objective(params: np.ndarray) -> float:
            w, b = params[0], params[1]
            p = sigmoid(w * z + b)
            # Clip for numerical stability in log
            eps = 1e-12
            p = np.clip(p, eps, 1.0 - eps)
            loss = -np.sum(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
            reg = 0.5 * self.l2_reg * (w ** 2)
            return float(loss + reg)

        res = minimize(objective, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
        self.w = float(res.x[0])
        self.b = float(res.x[1])
        self.is_fitted = True
        return self

    def predict_proba(self, z_tilde: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model 1 must be fitted before predict_proba.")
        return sigmoid(self.w * np.asarray(z_tilde, dtype=np.float64) + self.b)


class Model2LanguageShiftPlatt:
    """
    Model 2: Language-Shift Platt Scaling (Operational Candidate).
    Shared slope w across all conditions with condition-specific intercept shift delta_l:
    p_hat = sigmoid(w * tilde{z} + b_0 + delta_l), where delta_en == 0.
    L2 penalty: (lambda / 2) * (w^2 + sum_{l != en} delta_l^2)
    """

    def __init__(self, l2_reg: float = 1.0, ref_condition: str = "en"):
        self.l2_reg = l2_reg
        self.ref_condition = ref_condition
        self.w: float = 1.0
        self.b0: float = 0.0
        self.deltas: Dict[str, float] = {}
        self.conditions: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        z_tilde: np.ndarray,
        y: np.ndarray,
        condition_indices: List[str],
    ) -> "Model2LanguageShiftPlatt":
        z = np.asarray(z_tilde, dtype=np.float64)
        labels = np.asarray(y, dtype=np.float64)
        cond_list = list(condition_indices)

        unique_conds = sorted(list(set(cond_list)))
        if self.ref_condition not in unique_conds:
            unique_conds.insert(0, self.ref_condition)
        other_conds = [c for c in unique_conds if c != self.ref_condition]
        self.conditions = unique_conds

        # Map non-reference conditions to parameter indices
        # params layout: [w, b0, delta_1, delta_2, ..., delta_K]
        cond_to_param_idx = {c: i + 2 for i, c in enumerate(other_conds)}

        def objective(params: np.ndarray) -> float:
            w = params[0]
            b0 = params[1]
            deltas_vec = params[2:]

            # Construct effective intercepts
            shifts = np.zeros(len(z), dtype=np.float64)
            for c, idx in cond_to_param_idx.items():
                mask = [ci == c for ci in cond_list]
                shifts[mask] = params[idx]

            p = sigmoid(w * z + b0 + shifts)
            eps = 1e-12
            p = np.clip(p, eps, 1.0 - eps)
            loss = -np.sum(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
            reg = 0.5 * self.l2_reg * (w ** 2 + np.sum(deltas_vec ** 2))
            return float(loss + reg)

        init_params = np.zeros(2 + len(other_conds), dtype=np.float64)
        init_params[0] = 1.0  # initial w = 1.0

        res = minimize(objective, x0=init_params, method="L-BFGS-B")
        self.w = float(res.x[0])
        self.b0 = float(res.x[1])

        self.deltas = {self.ref_condition: 0.0}
        for c, idx in cond_to_param_idx.items():
            self.deltas[c] = float(res.x[idx])

        self.is_fitted = True
        return self

    def predict_proba(self, z_tilde: np.ndarray, condition: str) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model 2 must be fitted before predict_proba.")
        delta = self.deltas.get(condition, 0.0)
        return sigmoid(self.w * np.asarray(z_tilde, dtype=np.float64) + self.b0 + delta)


class Model3LanguageSpecificPlatt:
    """
    Model 3: Language-Specific Platt Scaling (Strictly Exploratory Benchmark).
    Independent slope and intercept per condition: p_hat = sigmoid(w_l * tilde{z} + b_l)
    Covariance estimate: S_hat_l = (X_l^T W_l X_l + diag(lambda, 0))^{-1}
    where W_l = diag(w_i * p_{i,l} * (1 - p_{i,l})) with uniform sample weights w_i = 1.0.
    """

    def __init__(self, l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.models: Dict[str, Dict[str, Any]] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        z_tilde: np.ndarray,
        y: np.ndarray,
        condition_indices: List[str],
    ) -> "Model3LanguageSpecificPlatt":
        z = np.asarray(z_tilde, dtype=np.float64)
        labels = np.asarray(y, dtype=np.float64)
        cond_list = list(condition_indices)
        unique_conds = sorted(list(set(cond_list)))

        for c in unique_conds:
            mask = np.array([ci == c for ci in cond_list], dtype=bool)
            z_c = z[mask]
            y_c = labels[mask]
            N_c = len(z_c)

            if N_c == 0:
                continue

            def obj_c(params: np.ndarray) -> float:
                w, b = params[0], params[1]
                p = sigmoid(w * z_c + b)
                eps = 1e-12
                p = np.clip(p, eps, 1.0 - eps)
                loss = -np.sum(y_c * np.log(p) + (1.0 - y_c) * np.log(1.0 - p))
                reg = 0.5 * self.l2_reg * (w ** 2)
                return float(loss + reg)

            res = minimize(obj_c, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
            w_c, b_c = float(res.x[0]), float(res.x[1])

            # Compute Hessian & Covariance Matrix
            # Design matrix X_l: N_c x 2, [z_c, 1]
            X_l = np.column_stack([z_c, np.ones(N_c, dtype=np.float64)])
            p_fitted = sigmoid(w_c * z_c + b_c)
            # W_l = diag(w_i * p * (1 - p)), w_i = 1.0
            W_diag = p_fitted * (1.0 - p_fitted)
            # X_l^T W_l X_l
            X_W_X = X_l.T @ (W_diag[:, None] * X_l)
            penalty_matrix = np.diag([self.l2_reg, 0.0])
            hessian = X_W_X + penalty_matrix
            cov_matrix = np.linalg.pinv(hessian)

            self.models[c] = {
                "w": w_c,
                "b": b_c,
                "covariance": cov_matrix,
                "n_samples": N_c,
            }

        self.is_fitted = True
        return self

    def predict_proba(self, z_tilde: np.ndarray, condition: str) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model 3 must be fitted before predict_proba.")
        m = self.models.get(condition)
        if not m:
            # Fallback to uncalibrated or identity sigmoid if unseen
            return sigmoid(np.asarray(z_tilde, dtype=np.float64))
        return sigmoid(m["w"] * np.asarray(z_tilde, dtype=np.float64) + m["b"])

    def get_parameter_covariance(self, condition: str) -> Optional[np.ndarray]:
        m = self.models.get(condition)
        return m["covariance"] if m else None


class Model4Isotonic:
    """
    Model 4: Isotonic Regression via PAVA (Strictly Exploratory Benchmark).
    Piecewise constant monotonic non-parametric calibration.
    """

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted: bool = False

    def fit(self, z_tilde: np.ndarray, y: np.ndarray) -> "Model4Isotonic":
        z = np.asarray(z_tilde, dtype=np.float64)
        labels = np.asarray(y, dtype=np.float64)
        self.iso.fit(z, labels)
        self.is_fitted = True
        return self

    def predict_proba(self, z_tilde: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model 4 must be fitted before predict_proba.")
        return np.clip(self.iso.predict(np.asarray(z_tilde, dtype=np.float64)), 0.0, 1.0)


class RecalibrationEngine:
    """
    Orchestrates fitting of Models 1-4 on P_cal, model selection on P_val,
    and out-of-sample inference on P_test.
    """

    def __init__(self):
        self.standardizer = Standardizer()
        self.model_1 = Model1PooledPlatt()
        self.model_2 = Model2LanguageShiftPlatt()
        self.model_3 = Model3LanguageSpecificPlatt()
        self.model_4 = Model4Isotonic()
        self.selected_operational_model: Optional[str] = None
        self.validation_brier_scores: Dict[str, float] = {}

    def fit_on_calibration(
        self,
        raw_confidences: np.ndarray,
        labels: np.ndarray,
        conditions: List[str],
    ) -> None:
        """Fits standardizer and all 4 calibration models on calibration partition (P_cal)."""
        # Step 1: Standardize logits on P_cal
        self.standardizer.fit(raw_confidences)
        z_tilde = self.standardizer.transform(raw_confidences)

        # Step 2: Fit Model 1
        self.model_1.fit(z_tilde, labels)

        # Step 3: Fit Model 2
        self.model_2.fit(z_tilde, labels, conditions)

        # Step 4: Fit Model 3 (Exploratory)
        self.model_3.fit(z_tilde, labels, conditions)

        # Step 5: Fit Model 4 (Exploratory)
        self.model_4.fit(z_tilde, labels)

    def select_model_on_validation(
        self,
        raw_confidences_val: np.ndarray,
        labels_val: np.ndarray,
        conditions_val: List[str],
    ) -> str:
        """
        Evaluates Model 1 vs Model 2 on validation partition (P_val) using Brier Score.
        Freezes selected operational model.
        """
        z_val = self.standardizer.transform(raw_confidences_val)
        y_val = np.asarray(labels_val, dtype=np.float64)

        # Model 1 BS
        p1 = self.model_1.predict_proba(z_val)
        bs1 = float(np.mean((p1 - y_val) ** 2))

        # Model 2 BS
        p2 = np.zeros_like(p1)
        for i, (zi, c) in enumerate(zip(z_val, conditions_val)):
            p2[i] = self.model_2.predict_proba(np.array([zi]), condition=c)[0]
        bs2 = float(np.mean((p2 - y_val) ** 2))

        self.validation_brier_scores = {
            "Model_1_Pooled_Platt": bs1,
            "Model_2_Language_Shift_Platt": bs2,
        }

        if bs2 < bs1:
            self.selected_operational_model = "Model_2_Language_Shift_Platt"
        else:
            self.selected_operational_model = "Model_1_Pooled_Platt"

        return self.selected_operational_model

    def predict_recalibrated(
        self,
        raw_confidences: np.ndarray,
        conditions: List[str],
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Predicts recalibrated probabilities for query data using chosen or selected model."""
        target_model = model_name or self.selected_operational_model or "Model_1_Pooled_Platt"
        z = self.standardizer.transform(raw_confidences)

        if target_model == "Model_1_Pooled_Platt":
            return self.model_1.predict_proba(z)
        elif target_model == "Model_2_Language_Shift_Platt":
            out = np.zeros(len(z), dtype=np.float64)
            for i, (zi, c) in enumerate(zip(z, conditions)):
                out[i] = self.model_2.predict_proba(np.array([zi]), condition=c)[0]
            return out
        elif target_model == "Model_3_Language_Specific_Platt":
            out = np.zeros(len(z), dtype=np.float64)
            for i, (zi, c) in enumerate(zip(z, conditions)):
                out[i] = self.model_3.predict_proba(np.array([zi]), condition=c)[0]
            return out
        elif target_model == "Model_4_Isotonic":
            return self.model_4.predict_proba(z)
        else:
            raise ValueError(f"Unknown calibration model: {target_model}")
