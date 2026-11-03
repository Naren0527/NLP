"""
Phase 5 Statistical Inference and Multiplicity Control.

Implements Sections 6.3 and 11 of Phase 5 Methodological Specification v4.5:
1. Hamilton-Hare Largest-Remainder Method for Deterministic Stratified Random QC Allocation.
2. Two-Tier Multiplicity Architecture:
   - Level A: Within-Track Familywise Testing (K = 5 comparisons, alpha = 0.01, 99.0% Bootstrap CIs).
   - Level B: Global / Across-Track Testing (K = 20 comparisons, alpha = 0.0025, 99.75% Bootstrap CIs).
3. Paired Cluster-Bootstrap for Comparative Differences (Delta M = M_l - M_en).
"""

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np


class HamiltonHareAllocator:
    """
    Implements the Hamilton-Hare Largest-Remainder Method for proportional
    integer quota allocation across strata.
    """

    @staticmethod
    def allocate_sample(
        strata_counts: Dict[str, int],
        total_sample_size: int,
        seed: int = 20260330,
    ) -> Dict[str, int]:
        """
        Distributes total_sample_size across strata proportionally:
        1. q_s = total_sample_size * (N_s / N_total)
        2. m_s = floor(q_s)
        3. r_s = q_s - m_s
        4. Distribute 1 extra unit to top strata by remainder r_s until sum == total_sample_size.
        """
        N_total = sum(strata_counts.values())
        if N_total == 0 or total_sample_size <= 0:
            return {k: 0 for k in strata_counts}

        if total_sample_size >= N_total:
            # Saturated sample: allocate all
            return dict(strata_counts)

        quotas: Dict[str, float] = {}
        allocations: Dict[str, int] = {}
        remainders: Dict[str, float] = {}

        for s, count in strata_counts.items():
            q = (float(total_sample_size) * float(count)) / float(N_total)
            quotas[s] = q
            allocations[s] = math.floor(q)
            remainders[s] = q - allocations[s]

        allocated_so_far = sum(allocations.values())
        deficit = total_sample_size - allocated_so_far

        # Sort keys primarily by remainder descending, secondarily by stratum name ascending for deterministic tie-break
        sorted_strata = sorted(remainders.keys(), key=lambda s: (-remainders[s], s))

        for i in range(deficit):
            s = sorted_strata[i]
            allocations[s] += 1

        return allocations

    @staticmethod
    def sample_strata_indices(
        strata_elements: Dict[str, List[Any]],
        total_sample_size: int,
        seed: int = 20260330,
    ) -> Dict[str, List[Any]]:
        """
        Selects exact items from each stratum using Hamilton-Hare quota and fixed random seed.
        """
        counts = {s: len(elems) for s, elems in strata_elements.items()}
        quotas = HamiltonHareAllocator.allocate_sample(counts, total_sample_size, seed=seed)

        rng = np.random.RandomState(seed)
        sampled: Dict[str, List[Any]] = {}

        # Sort strata names for deterministic order of RNG draws
        for s in sorted(strata_elements.keys()):
            m = quotas[s]
            elems = list(strata_elements[s])
            if m <= 0:
                sampled[s] = []
            elif m >= len(elems):
                sampled[s] = list(elems)
            else:
                # Deterministic random choice without replacement
                indices = rng.choice(len(elems), size=m, replace=False)
                sampled[s] = [elems[idx] for idx in sorted(indices)]

        return sampled


@dataclass
class BootstrapResult:
    observed_diff: float
    ci_lower: float
    ci_upper: float
    fwer_tier: str  # 'Level_A_Within_Track' or 'Level_B_Global'
    alpha_comp: float
    coverage: float
    bootstrap_replicates: int
    is_significant: bool  # True if CI excludes 0.0


class MultiplicityController:
    """Two-tier FWER multiplicity thresholds."""
    LEVEL_A_K = 5
    LEVEL_A_ALPHA = 0.05 / 5     # 0.01 (99.0% simultaneous coverage)
    LEVEL_A_LOWER_QUANTILE = 0.005
    LEVEL_A_UPPER_QUANTILE = 0.995

    LEVEL_B_K = 20
    LEVEL_B_ALPHA = 0.05 / 20    # 0.0025 (99.75% simultaneous coverage)
    LEVEL_B_LOWER_QUANTILE = 0.00125
    LEVEL_B_UPPER_QUANTILE = 0.99875


class BootstrapInference:
    """
    Performs question-level cluster bootstrap inference for paired comparative metrics.
    """

    @staticmethod
    def test_paired_difference(
        question_ids: List[str],
        metric_fn: Callable[[List[str]], float],
        fwer_tier: str = "Level_A_Within_Track",
        n_bootstraps: int = 10000,
        seed: int = 20260330,
    ) -> BootstrapResult:
        """
        question_ids: list of unique semantic question ids in the paired cohort.
        metric_fn: function taking a list of resampled question_ids and returning Delta M.
        """
        unique_qids = np.array(sorted(list(set(question_ids))))
        N = len(unique_qids)

        if N == 0:
            return BootstrapResult(
                observed_diff=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                fwer_tier=fwer_tier,
                alpha_comp=0.0,
                coverage=0.0,
                bootstrap_replicates=0,
                is_significant=False,
            )

        observed = metric_fn(list(unique_qids))

        # Multiplicity parameters
        if fwer_tier == "Level_A_Within_Track":
            alpha = MultiplicityController.LEVEL_A_ALPHA
            q_low = MultiplicityController.LEVEL_A_LOWER_QUANTILE
            q_high = MultiplicityController.LEVEL_A_UPPER_QUANTILE
            cov = 0.99
        elif fwer_tier == "Level_B_Global":
            alpha = MultiplicityController.LEVEL_B_ALPHA
            q_low = MultiplicityController.LEVEL_B_LOWER_QUANTILE
            q_high = MultiplicityController.LEVEL_B_UPPER_QUANTILE
            cov = 0.9975
        else:
            raise ValueError(f"Unknown FWER tier: {fwer_tier}")

        rng = np.random.RandomState(seed)
        boot_diffs = np.zeros(n_bootstraps, dtype=np.float64)

        for b in range(n_bootstraps):
            # Resample question clusters with replacement
            sampled_idx = rng.choice(N, size=N, replace=True)
            resampled_qids = list(unique_qids[sampled_idx])
            boot_diffs[b] = metric_fn(resampled_qids)

        ci_lower = float(np.percentile(boot_diffs, q_low * 100.0))
        ci_upper = float(np.percentile(boot_diffs, q_high * 100.0))

        # Test whether 0 is strictly excluded from confidence interval
        is_significant = (ci_lower > 0.0) or (ci_upper < 0.0)

        return BootstrapResult(
            observed_diff=observed,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            fwer_tier=fwer_tier,
            alpha_comp=alpha,
            coverage=cov,
            bootstrap_replicates=n_bootstraps,
            is_significant=is_significant,
        )
