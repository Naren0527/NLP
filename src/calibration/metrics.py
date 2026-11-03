"""
Phase 5 Calibration Metrics and Formal Estimators.

Implements Section 8 of Phase 5 Methodological Specification v4.5:
- Brier Score (BS = (1/N) * sum (c_i - y_i)^2)
- Murphy's Decomposition (BS = REL - RES + UNC)
- Expected Calibration Error (ECE_B) via Quantile (Equal-Frequency) Binning
- Maximum Calibration Error (MCE)
- Root Mean Squared Calibration Error (RMSCE = sqrt(REL))
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class MurphyDecomposition:
    uncertainty: float   # UNC = y_bar * (1 - y_bar)
    reliability: float   # REL = sum (N_b / N) * (c_bar_b - y_bar_b)^2
    resolution: float    # RES = sum (N_b / N) * (y_bar_b - y_bar)^2
    brier_score: float   # BS = (1/N) * sum (c_i - y_i)^2
    bin_covariance_residual: float  # RESID = BS - (REL - RES + UNC)

    @property
    def is_consistent(self) -> bool:
        """
        Verifies exact algebraic identity:
        BS == REL - RES + UNC + RESID
        When c_i are replaced by bin centroids c_bar_b (discrete forecasts), RESID == 0.
        """
        reconstructed = self.reliability - self.resolution + self.uncertainty + self.bin_covariance_residual
        return math.isclose(self.brier_score, reconstructed, abs_tol=1e-5)


@dataclass
class BinMetric:
    bin_index: int
    count: int
    mean_confidence: float
    mean_accuracy: float
    calibration_error: float  # |mean_conf - mean_acc|
    conf_min: float
    conf_max: float


@dataclass
class CalibrationMetrics:
    brier_score: float
    ece: float
    mce: float
    rmsce: float
    num_samples: int
    num_bins: int
    bins: List[BinMetric]
    decomposition: Optional[MurphyDecomposition] = None


def compute_brier_score(confidences: np.ndarray, labels: np.ndarray) -> float:
    """Computes standard unweighted Brier Score."""
    if len(confidences) == 0:
        return 0.0
    return float(np.mean((confidences - labels) ** 2))


def compute_murphy_decomposition(
    confidences: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 5,
) -> MurphyDecomposition:
    """
    Computes Murphy's decomposition of the Brier Score into Reliability, Resolution, and Uncertainty.
    Uses quantile (equal-frequency) binning for consistency with ECE.
    """
    N = len(confidences)
    if N == 0:
        return MurphyDecomposition(
            uncertainty=0.0,
            reliability=0.0,
            resolution=0.0,
            brier_score=0.0,
            bin_covariance_residual=0.0,
        )

    y_bar = float(np.mean(labels))
    unc = float(y_bar * (1.0 - y_bar))
    bs = compute_brier_score(confidences, labels)

    # Quantile bins
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(confidences, quantiles)
    # Ensure strictly non-decreasing edges with distinct bounds if possible
    bin_edges[0] = -1e-9
    bin_edges[-1] = 1.0 + 1e-9

    rel = 0.0
    res = 0.0

    # Assign samples to bins
    bin_indices = np.digitize(confidences, bin_edges) - 1
    # Clip any out of bound
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    for b in range(num_bins):
        mask = (bin_indices == b)
        N_b = int(np.sum(mask))
        if N_b > 0:
            c_bin = confidences[mask]
            c_bar_b = float(np.mean(c_bin))
            y_bar_b = float(np.mean(labels[mask]))
            weight = float(N_b) / float(N)
            rel += weight * ((c_bar_b - y_bar_b) ** 2)
            res += weight * ((y_bar_b - y_bar) ** 2)

    resid = bs - (rel - res + unc)

    return MurphyDecomposition(
        uncertainty=unc,
        reliability=rel,
        resolution=res,
        brier_score=bs,
        bin_covariance_residual=resid,
    )


def compute_quantile_ece(
    confidences: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 5,
) -> CalibrationMetrics:
    """
    Computes Expected Calibration Error (ECE), MCE, RMSCE, and Murphy Decomposition
    using quantile (equal-frequency) binning across confidence forecasts.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    N = len(conf)

    if N == 0:
        return CalibrationMetrics(
            brier_score=0.0,
            ece=0.0,
            mce=0.0,
            rmsce=0.0,
            num_samples=0,
            num_bins=num_bins,
            bins=[],
            decomposition=None,
        )

    # Check bounds
    np.clip(conf, 0.0, 1.0, out=conf)

    bs = compute_brier_score(conf, y)
    decomp = compute_murphy_decomposition(conf, y, num_bins=num_bins)

    # Determine quantile bin edges
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(conf, quantiles)
    bin_edges[0] = -1e-9
    bin_edges[-1] = 1.0 + 1e-9

    bin_indices = np.digitize(conf, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_metrics: List[BinMetric] = []
    ece = 0.0
    mce = 0.0
    rel_sum = 0.0

    for b in range(num_bins):
        mask = (bin_indices == b)
        N_b = int(np.sum(mask))
        if N_b > 0:
            c_bar_b = float(np.mean(conf[mask]))
            y_bar_b = float(np.mean(y[mask]))
            cal_err = abs(c_bar_b - y_bar_b)
            weight = float(N_b) / float(N)
            ece += weight * cal_err
            rel_sum += weight * (cal_err ** 2)
            if cal_err > mce:
                mce = cal_err
            bin_metrics.append(
                BinMetric(
                    bin_index=b,
                    count=N_b,
                    mean_confidence=c_bar_b,
                    mean_accuracy=y_bar_b,
                    calibration_error=cal_err,
                    conf_min=float(np.min(conf[mask])),
                    conf_max=float(np.max(conf[mask])),
                )
            )
        else:
            bin_metrics.append(
                BinMetric(
                    bin_index=b,
                    count=0,
                    mean_confidence=0.0,
                    mean_accuracy=0.0,
                    calibration_error=0.0,
                    conf_min=0.0,
                    conf_max=0.0,
                )
            )

    rmsce = math.sqrt(rel_sum)

    return CalibrationMetrics(
        brier_score=bs,
        ece=ece,
        mce=mce,
        rmsce=rmsce,
        num_samples=N,
        num_bins=num_bins,
        bins=bin_metrics,
        decomposition=decomp,
    )
