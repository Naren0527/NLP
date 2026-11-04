import React, { useEffect, useState } from 'react';
import { MetricCard } from '../components/cards/MetricCard';
import { ResearchCard } from '../components/cards/ResearchCard';
import { StatusBadge } from '../components/status/StatusBadge';
import { fetchBenchmarkSummary } from '../services/api';
import { BenchmarkSummaryResponse } from '../types';
import { Activity } from 'lucide-react';

export const OverviewPage: React.FC = () => {
  const [summary, setSummary] = useState<BenchmarkSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBenchmarkSummary()
      .then((data) => setSummary(data))
      .catch((err) => console.error('Failed to load benchmark summary:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* 4-Metric Grid with Minimal Horizontal Progress Bars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Indic Accuracy Collapse"
          value="95.6% → 23.8%"
          progressPercent={23.8}
          sub="Drop from English to Hindi on identical factual premises"
          badge="CRITICAL GAP"
          variant="error"
        />
        <MetricCard
          title="Verbalized Overconfidence"
          value="0.028 → 0.703"
          progressPercent={70.3}
          sub="ECE explodes; model states >91% certainty on falsehoods"
          badge="MISCALIBRATED"
          variant="default"
        />
        <MetricCard
          title="Mitigated Indic Accuracy"
          value="31.1% → 86.1%"
          progressPercent={86.1}
          sub="+55.0 pp gain via Bilingual Arbitration (Strategy C)"
          badge="RECOVERED"
          variant="success"
        />
        <MetricCard
          title="Calibration Recovery (M2)"
          value="85.4% ECE Drop"
          progressPercent={85.4}
          sub="Language-shift Platt scaling cuts average ECE to 0.090"
          badge="ALIGNED"
          variant="accent"
        />
      </div>

      {/* Main Grid: Executive Briefing & Architecture */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: Briefing & RQs */}
        <div className="lg:col-span-7 space-y-5">
          <ResearchCard
            title="Executive Research Briefing"
            subtitle="Investigation of Model Assertiveness vs. Grounded Multilingual Reliability"
          >
            <div className="space-y-3.5 text-xs text-[#A1A1A1] leading-relaxed">
              <p>
                Modern Instruction-Tuned and RLHF Large Language Models exhibit high conversational
                assertiveness. In high-resource English queries, their verbalized self-assessments
                align well with true factual competency. However, when queried in non-Latin Indic
                languages (Hindi, Bengali, Telugu, Tamil),{' '}
                <strong className="text-[#F5F5F5]">verbalized confidence fails to scale downward</strong>{' '}
                with actual task performance.
              </p>
              <p>
                The model asserts erroneous claims with extreme stated certainty (&gt;90%),
                resulting in severe cross-lingual calibration collapse.
              </p>
              <div className="bg-[#111111] border border-[#242424] rounded-xl p-3.5 text-xs text-[#F5F5F5] flex items-center gap-3">
                <Activity className="w-4 h-4 text-[#A1A1A1] shrink-0" />
                <span className="text-[11px] leading-relaxed">
                  Evaluated across <strong className="text-[#F5F5F5]">4,050 genuine GPU model inferences</strong> with
                  Google Gemma 3 4B Instruct, comparing verbalized confidence against token geometric
                  entropy across 750 pairwise evaluations.
                </span>
              </div>
            </div>
          </ResearchCard>

          <ResearchCard
            title="Core Research Questions (RQ1 – RQ4)"
            subtitle="Hypothesis Validation Under Frozen Methodology v4.5"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              <div className="bg-[#111111] border border-[#242424] p-4 rounded-[20px] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-[#A1A1A1]">RQ1 // COLLAPSE</span>
                    <StatusBadge variant="default" size="sm">PROVEN</StatusBadge>
                  </div>
                  <h4 className="text-xs font-semibold text-[#F5F5F5] mb-1">
                    Does instruction tuning induce calibration collapse?
                  </h4>
                  <p className="text-[11px] text-[#666666] leading-normal">
                    Yes. While English achieves an ECE of 0.0440, Indic conditions collapse to an average ECE of 0.6347. Verbalized confidence stays &gt;91% regardless of accuracy.
                  </p>
                </div>
              </div>

              <div className="bg-[#111111] border border-[#242424] p-4 rounded-[20px] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-[#A1A1A1]">RQ2 // DIVERGENCE</span>
                    <StatusBadge variant="default" size="sm">CONFIRMED</StatusBadge>
                  </div>
                  <h4 className="text-xs font-semibold text-[#F5F5F5] mb-1">
                    Does statistical confidence preserve uncertainty?
                  </h4>
                  <p className="text-[11px] text-[#666666] leading-normal">
                    Yes. Internal token geometric likelihoods drop to 48.9%–61.8% in Indic conditions, accurately reflecting difficulty, while verbalized confidence is inflated by persona tuning.
                  </p>
                </div>
              </div>

              <div className="bg-[#111111] border border-[#242424] p-4 rounded-[20px] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-[#A1A1A1]">RQ3 // CONSISTENCY</span>
                    <StatusBadge variant="default" size="sm">DIAGNOSED</StatusBadge>
                  </div>
                  <h4 className="text-xs font-semibold text-[#F5F5F5] mb-1">
                    Do surface agreement scores conceal hallucinations?
                  </h4>
                  <p className="text-[11px] text-[#666666] leading-normal">
                    Yes. Casual LLM judge semantic agreement is 88.4%, but strict factual evidence verification reveals true grounded consistency is only 28.7%–37.3%.
                  </p>
                </div>
              </div>

              <div className="bg-[#111111] border border-[#242424] p-4 rounded-[20px] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-[#A1A1A1]">RQ4 // MITIGATION</span>
                    <StatusBadge variant="success" size="sm">SOLVED</StatusBadge>
                  </div>
                  <h4 className="text-xs font-semibold text-[#F5F5F5] mb-1">
                    Can pivoting and Platt scaling restore reliability?
                  </h4>
                  <p className="text-[11px] text-[#666666] leading-normal">
                    Bilingual Arbitration (Strategy C) recovers Indic accuracy to 86.11% (+55.0 pp gain), and Language-Shift Platt Scaling cuts Expected Calibration Error by 85.4%.
                  </p>
                </div>
              </div>
            </div>
          </ResearchCard>
        </div>

        {/* Right Column: Pipeline Architecture Figure & Partition Telemetry */}
        <div className="lg:col-span-5 space-y-5">
          <ResearchCard
            title="Mitigation Architecture"
            subtitle="Dual-Track Pipeline: English Pivot Anchoring & Platt Recalibration"
          >
            <div className="rounded-[18px] border border-[#242424] overflow-hidden bg-[#111111] mb-4">
              <img
                src="/figures/fig14_pipeline_architecture.png"
                alt="System Architecture Pipeline"
                className="w-full h-auto object-contain"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = 'none';
                }}
              />
            </div>

            <div className="bg-[#111111] border border-[#242424] rounded-2xl p-4 space-y-2 font-mono text-xs text-[#A1A1A1]">
              <div className="text-xs font-semibold text-[#F5F5F5] uppercase tracking-wider">
                Benchmark Partition Telemetry
              </div>
              <div className="space-y-1.5 pt-1 text-[11px]">
                <div className="flex justify-between border-b border-[#242424] pb-1.5">
                  <span className="text-[#666666]">Calibration Split (P_cal)</span>
                  <span className="text-[#F5F5F5]">75 QA Pairs (L-BFGS-B Fitting)</span>
                </div>
                <div className="flex justify-between border-b border-[#242424] pb-1.5">
                  <span className="text-[#666666]">Validation Split (P_val)</span>
                  <span className="text-[#F5F5F5]">30 QA Pairs (Strategy Selection)</span>
                </div>
                <div className="flex justify-between border-b border-[#242424] pb-1.5">
                  <span className="text-[#666666]">Held-Out Test Set (P_test)</span>
                  <span className="text-[#F5F5F5]">45 QA Pairs (Generalization)</span>
                </div>
                <div className="flex justify-between pt-0.5 text-[#F5F5F5]">
                  <span className="text-[#666666]">Total Real GPU Inferences</span>
                  <span>4,050 Records (B=10,000 Bootstrap)</span>
                </div>
              </div>
            </div>
          </ResearchCard>
        </div>
      </div>
    </div>
  );
};
