import React, { useState, useEffect } from 'react';
import { fetchBenchmarkSummary } from '../services/api';
import { BenchmarkSummaryResponse } from '../types';
import { ResearchCard } from '../components/cards/ResearchCard';
import { MetricCard } from '../components/cards/MetricCard';
import { AccuracyBarChart } from '../components/charts/AccuracyBarChart';
import { CalibrationBarChart } from '../components/charts/CalibrationBarChart';
import { formatPercent, formatPp, formatDecimal } from '../utils/formatters';

export const BenchmarkPage: React.FC = () => {
  const [data, setData] = useState<BenchmarkSummaryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchBenchmarkSummary()
      .then((res) => setData(res))
      .catch((err) => console.error('Benchmark fetch error:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* 4 Summary Metric Cards with Understated Progress Bars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Indic Mean Baseline"
          value={formatPercent(data?.indic_aggregate.baseline_accuracy || 0.311)}
          progressPercent={(data?.indic_aggregate.baseline_accuracy || 0.311) * 100}
          sub="Average accuracy across Hindi, Bengali, Telugu, Tamil"
          badge="COLLAPSED"
          variant="error"
        />
        <MetricCard
          title="Indic Mean Mitigated"
          value={formatPercent(data?.indic_aggregate.mitigated_accuracy || 0.861)}
          progressPercent={(data?.indic_aggregate.mitigated_accuracy || 0.861) * 100}
          sub="Strategy C Bilingual Arbitration on P_test"
          badge="+55.0 PP GAIN"
          variant="success"
        />
        <MetricCard
          title="Baseline Indic ECE"
          value={formatDecimal(data?.indic_aggregate.baseline_ece || 0.6347, 4)}
          progressPercent={(data?.indic_aggregate.baseline_ece || 0.6347) * 100}
          sub="Massive verbalized overconfidence error"
          badge="SEVERE"
          variant="default"
        />
        <MetricCard
          title="Mitigated Indic ECE"
          value={formatDecimal(data?.indic_aggregate.mitigated_ece || 0.4666, 4)}
          progressPercent={(data?.indic_aggregate.mitigated_ece || 0.4666) * 100}
          sub="Pre-Platt; drops to 0.090 with Model 2"
          badge="IMPROVED"
          variant="accent"
        />
      </div>

      {/* Comprehensive Quantitative Benchmark Table */}
      <ResearchCard
        title="Held-Out Test Set (P_test, N=45) Quantitative Metrics"
        subtitle="Full Empirical Breakdown across English and Indic Language Conditions"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[#242424] text-[#666666] uppercase tracking-wider font-mono">
                <th className="py-2.5 px-3">Condition</th>
                <th className="py-2.5 px-3">Script</th>
                <th className="py-2.5 px-3 text-right">Baseline Acc</th>
                <th className="py-2.5 px-3 text-right">Mitigated Acc</th>
                <th className="py-2.5 px-3 text-right">Acc Gain</th>
                <th className="py-2.5 px-3 text-right">Rel Error Red</th>
                <th className="py-2.5 px-3 text-right">Baseline ECE</th>
                <th className="py-2.5 px-3 text-right">Mitigated ECE</th>
                <th className="py-2.5 px-3 text-right">ECE Reduction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#242424] font-mono">
              {data?.languages.map((row) => (
                <tr
                  key={row.code}
                  className="hover:bg-[#111111] transition-colors text-[#F5F5F5]"
                >
                  <td className="py-3 px-3 font-semibold text-[#F5F5F5]">
                    {row.language.toUpperCase()} ({row.code})
                  </td>
                  <td className="py-3 px-3 text-[#A1A1A1] font-sans">
                    {row.script}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A1A1A1]">
                    {formatPercent(row.baseline_accuracy)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#4ADE80] font-semibold">
                    {formatPercent(row.mitigated_accuracy)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#4ADE80]">
                    {formatPp(row.accuracy_gain_pp)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A1A1A1]">
                    {row.code === 'en' ? '—' : `${row.relative_error_reduction.toFixed(1)}%`}
                  </td>
                  <td className="py-3 px-3 text-right text-[#F87171]">
                    {row.baseline_ece.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A1A1A1]">
                    {row.mitigated_ece.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#4ADE80]">
                    {row.ece_reduction.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ResearchCard>

      {/* Interactive Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ResearchCard
          title="Baseline vs. Mitigated Accuracy"
          subtitle="Test Split Generalization (N=45 per condition)"
        >
          {data ? (
            <AccuracyBarChart data={data.languages} height={280} />
          ) : (
            <div className="h-64 flex items-center justify-center text-xs text-[#666666]">
              Loading chart data...
            </div>
          )}
        </ResearchCard>

        <ResearchCard
          title="Expected Calibration Error (ECE)"
          subtitle="Pre vs. Post Mitigation across Languages"
        >
          {data ? (
            <CalibrationBarChart data={data.languages} height={280} />
          ) : (
            <div className="h-64 flex items-center justify-center text-xs text-[#666666]">
              Loading chart data...
            </div>
          )}
        </ResearchCard>
      </div>

      {/* Publication Figures Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        <ResearchCard
          title="Fig 01 // Baseline vs Mitigated"
          subtitle="Test Split (N=45) Accuracy Breakdown"
        >
          <img
            src="/figures/fig1_baseline_vs_mitigated_accuracy.png"
            alt="Fig 01"
            className="w-full h-auto rounded-xl border border-[#242424]"
          />
        </ResearchCard>

        <ResearchCard
          title="Fig 02 // Absolute Accuracy Gain"
          subtitle="Percentage Point Recovery per Language"
        >
          <img
            src="/figures/fig2_overall_accuracy_improvement.png"
            alt="Fig 02"
            className="w-full h-auto rounded-xl border border-[#242424]"
          />
        </ResearchCard>

        <ResearchCard
          title="Fig 03 // Relative Error Reduction"
          subtitle="Error Elimination Rate across Indic Conditions"
        >
          <img
            src="/figures/fig3_relative_error_reduction.png"
            alt="Fig 03"
            className="w-full h-auto rounded-xl border border-[#242424]"
          />
        </ResearchCard>
      </div>
    </div>
  );
};
