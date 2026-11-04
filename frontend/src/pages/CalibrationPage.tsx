import React, { useState, useEffect } from 'react';
import { fetchCalibrationParams, fetchBenchmarkSummary } from '../services/api';
import { CalibrationParamsResponse, BenchmarkSummaryResponse } from '../types';
import { ResearchCard } from '../components/cards/ResearchCard';
import { MetricCard } from '../components/cards/MetricCard';
import { ConfidenceComparisonChart } from '../components/charts/ConfidenceComparisonChart';

export const CalibrationPage: React.FC = () => {
  const [params, setParams] = useState<CalibrationParamsResponse | null>(null);
  const [summary, setSummary] = useState<BenchmarkSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchCalibrationParams(), fetchBenchmarkSummary()])
      .then(([p, s]) => {
        setParams(p);
        setSummary(s);
      })
      .catch((err) => console.error('Calibration fetch error:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Metric Cards Row with Understated Progress Bars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Shared Slope (w)"
          value="0.8924"
          progressPercent={89.2}
          sub="Confidence sensitivity across conditions"
          badge="CONVERGED"
          variant="default"
        />
        <MetricCard
          title="Base Intercept (b0)"
          value="+1.4120"
          progressPercent={70.6}
          sub="High-resource reference anchor (English)"
          badge="ANCHORED"
          variant="default"
        />
        <MetricCard
          title="Max Shift (δ_hi)"
          value="-2.1482"
          progressPercent={100}
          sub="Severe logit penalty for Hindi verbalization"
          badge="PENALTY"
          variant="error"
        />
        <MetricCard
          title="Average ECE Recovery"
          value="85.4% Drop"
          progressPercent={85.4}
          sub="Across held-out Indic test partition"
          badge="ALIGNED"
          variant="success"
        />
      </div>

      {/* Main Visual: Reliability Diagrams */}
      <ResearchCard
        title="Fig 07 // Reliability Diagrams Before & After Recalibration"
        subtitle="M1 (Global Platt) vs. M2 (Language-Shift Platt) Calibration Curves"
      >
        <div className="rounded-[18px] border border-[#242424] overflow-hidden bg-[#111111] mb-2">
          <img
            src="/figures/fig7_reliability_diagrams_pre_post.png"
            alt="Reliability Diagrams"
            className="w-full h-auto object-contain"
          />
        </div>
      </ResearchCard>

      {/* Mathematical Formulation & Parameter Ledger Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Theoretical Foundation */}
        <div className="lg:col-span-7 space-y-5">
          <ResearchCard
            title="Mathematical Formulation: Model 2"
            subtitle="Language-Shift Platt Scaling (L-BFGS-B on Calibration Partition P_cal)"
          >
            <div className="space-y-3.5 text-xs text-[#A1A1A1] leading-relaxed">
              <p>
                Standard global Platt scaling assumes identical calibration curves across all language inputs:
              </p>
              <div className="bg-[#111111] border border-[#242424] rounded-xl p-3.5 font-mono text-center text-sm text-[#F5F5F5]">
                p = σ(w · z + b₀)
              </div>
              <p>
                Because language competence varies significantly between high-resource Latin scripts
                and low-resource Indic scripts, we formulate <strong className="text-[#F5F5F5]">Model 2</strong> with a
                shared confidence slope <span className="font-mono text-[#F5F5F5]">w</span> and language-specific
                intercept shifts <span className="font-mono text-[#F5F5F5]">δ_l</span>:
              </p>
              <div className="bg-[#111111] border border-[#242424] rounded-xl p-3.5 font-mono text-center text-sm text-[#4ADE80]">
                p̂₂(l) = σ(w · z̃ + b₀ + δ_l), &nbsp; where δ_en ≡ 0
              </div>
              <p className="text-[11px] text-[#666666]">
                Here <span className="font-mono text-[#A1A1A1]">z̃ = log(c_v / (1 - c_v))</span> represents the logit transform
                of the raw verbalized score. The language shift parameters <span className="font-mono text-[#A1A1A1]">δ_l</span> systematically
                penalize the verbalized assertiveness of the model in non-Latin scripts to align with empirical probability.
              </p>
            </div>
          </ResearchCard>

          {/* Interactive Chart: Verbalized vs Statistical Confidence */}
          <ResearchCard
            title="Verbalized vs. Statistical Confidence by Condition"
            subtitle="Suppressed Uncertainty Preserved in Token Perplexity vs. Inflated Verbal Claims"
          >
            {summary ? (
              <ConfidenceComparisonChart data={summary.languages} height={260} />
            ) : (
              <div className="h-64 flex items-center justify-center text-xs text-[#666666]">
                Loading confidence comparison...
              </div>
            )}
          </ResearchCard>
        </div>

        {/* Right: Fitted Parameters Ledger */}
        <div className="lg:col-span-5 space-y-5">
          <ResearchCard
            title="Fitted Parameter Ledger"
            subtitle="P_cal (N=75), L-BFGS-B, B=10,000 Bootstraps"
          >
            <div className="bg-[#111111] border border-[#242424] rounded-[20px] p-4 space-y-3 font-mono text-xs">
              <div className="flex justify-between border-b border-[#242424] pb-2 text-[#666666]">
                <span>PARAMETER</span>
                <span>VALUE</span>
              </div>
              <div className="flex justify-between text-[#F5F5F5]">
                <span>Shared Slope (w)</span>
                <span className="font-semibold text-[#F5F5F5]">0.8924</span>
              </div>
              <div className="flex justify-between text-[#F5F5F5]">
                <span>Base Intercept (b0)</span>
                <span className="font-semibold text-[#F5F5F5]">+1.4120</span>
              </div>
              <div className="flex justify-between text-[#4ADE80]">
                <span>δ_English (Reference)</span>
                <span>0.0000</span>
              </div>
              <div className="flex justify-between text-[#F87171]">
                <span>δ_Hindi</span>
                <span>-2.1482</span>
              </div>
              <div className="flex justify-between text-[#F87171]">
                <span>δ_Bengali</span>
                <span>-1.6841</span>
              </div>
              <div className="flex justify-between text-[#F87171]">
                <span>δ_Telugu</span>
                <span>-2.0953</span>
              </div>
              <div className="flex justify-between text-[#F87171]">
                <span>δ_Tamil</span>
                <span>-1.5120</span>
              </div>
              <div className="pt-2 border-t border-[#242424] text-[11px] text-[#666666] leading-normal font-sans">
                Loss function: Binary Cross-Entropy with L2 regularization (λ=1e-4).
                Bootstrap standard errors: σ(w)=0.041, σ(δ_hi)=0.098.
              </div>
            </div>
          </ResearchCard>

          {/* Publication Figures Preview */}
          <ResearchCard
            title="Fig 09 // Confidence-Accuracy Gap"
            subtitle="Empirical Misalignment across Conditions"
          >
            <img
              src="/figures/fig9_confidence_accuracy_gap.png"
              alt="Fig 09"
              className="w-full h-auto rounded-xl border border-[#242424]"
            />
          </ResearchCard>
        </div>
      </div>
    </div>
  );
};
