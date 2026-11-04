import React, { useState, useEffect } from 'react';
import { fetchStrategiesBenchmarks } from '../services/api';
import { StrategiesBenchmarksResponse } from '../types';
import { ResearchCard } from '../components/cards/ResearchCard';
import { MetricCard } from '../components/cards/MetricCard';
import { CoverageRiskLineChart } from '../components/charts/CoverageRiskLineChart';
import { StatusBadge } from '../components/status/StatusBadge';
import { formatPercent, formatDecimal } from '../utils/formatters';

export const StrategiesPage: React.FC = () => {
  const [data, setData] = useState<StrategiesBenchmarksResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStrategiesBenchmarks()
      .then((res) => setData(res))
      .catch((err) => console.error('Strategies fetch error:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Top Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Validation Winner"
          value="Strategy C"
          progressPercent={80}
          sub="Bilingual Verification & Arbitration"
          badge="SELECTED"
          variant="success"
        />
        <MetricCard
          title="Peak Validation Acc"
          value="80.0%"
          progressPercent={80.0}
          sub="Lowest Brier score (0.1797) on P_val"
          badge="TOP SCORE"
          variant="accent"
        />
        <MetricCard
          title="Optimal Cutoff (τ)"
          value="τ = 0.55"
          progressPercent={55}
          sub="Yields 95.6% accuracy at 82% coverage"
          badge="BALANCED"
          variant="default"
        />
        <MetricCard
          title="Zero-Risk Frontier"
          value="100.0% Acc"
          progressPercent={100}
          sub="At τ = 0.90 with 31% coverage"
          badge="SAFETY LIMIT"
          variant="default"
        />
      </div>

      {/* Stage A: Validation Strategy Selection Table */}
      <ResearchCard
        title="Stage A: Validation Strategy Selection (P_val, N=30)"
        subtitle="Empirical Tournament across Architectural Mitigation Candidates"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse font-mono">
            <thead>
              <tr className="border-b border-[#242424] text-[#666666] uppercase tracking-wider">
                <th className="py-2.5 px-3">Candidate Strategy</th>
                <th className="py-2.5 px-3 text-right">Validation Accuracy</th>
                <th className="py-2.5 px-3 text-right">Validation Brier</th>
                <th className="py-2.5 px-3 text-right">Avg Latency</th>
                <th className="py-2.5 px-3 text-right">Selection Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#242424] text-[#F5F5F5]">
              {data?.validation_strategy_selection.map((item) => (
                <tr key={item.strategy} className="hover:bg-[#111111] transition-colors">
                  <td className="py-3 px-3 font-semibold text-[#F5F5F5]">
                    {item.strategy}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A1A1A1]">
                    {formatPercent(item.validation_accuracy)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#A1A1A1]">
                    {formatDecimal(item.validation_brier, 4)}
                  </td>
                  <td className="py-3 px-3 text-right text-[#666666]">
                    {item.average_latency_sec.toFixed(1)}s
                  </td>
                  <td className="py-3 px-3 text-right">
                    <StatusBadge
                      variant={
                        item.status.includes('WINNER')
                          ? 'success'
                          : item.status.includes('ACCEPTED')
                          ? 'accent'
                          : 'default'
                      }
                      size="sm"
                    >
                      {item.status}
                    </StatusBadge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ResearchCard>

      {/* Interactive Recharts: Strategy E Coverage-Risk Frontier */}
      <ResearchCard
        title="Strategy E: Selective Answering Coverage-Risk Frontier"
        subtitle="Simulated Trade-Off Between Coverage Rate and Selective Risk across Threshold Cutoffs (τ ∈ [0.10, 0.90])"
      >
        {data ? (
          <CoverageRiskLineChart data={data.strategy_e_frontier} height={300} />
        ) : (
          <div className="h-64 flex items-center justify-center text-xs text-[#666666]">
            Loading frontier data...
          </div>
        )}
      </ResearchCard>

      {/* Publication Figures 2x2 Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <ResearchCard
          title="Fig 10 // Accuracy vs. Coverage Curve"
          subtitle="Selective Answering Frontier (Strategy E)"
        >
          <img
            src="/figures/fig10_accuracy_vs_coverage_curve.png"
            alt="Fig 10"
            className="w-full h-auto rounded-xl border border-[#242424]"
          />
        </ResearchCard>

        <ResearchCard
          title="Fig 11 // Selective Risk vs. Coverage"
          subtitle="Risk Metric Minimization Function"
        >
          <img
            src="/figures/fig11_selective_risk_vs_coverage.png"
            alt="Fig 11"
            className="w-full h-auto rounded-xl border border-[#242424]"
          />
        </ResearchCard>

        <ResearchCard
          title="Fig 12 // Strategy Comparison Heatmap"
          subtitle="P_val Multi-Metric Performance Matrix"
        >
          <img
            src="/figures/fig12_strategy_comparison_heatmap.png"
            alt="Fig 12"
            className="w-full h-auto rounded-xl border border-[#242424]"
          />
        </ResearchCard>

        <ResearchCard
          title="Fig 13 // Inference Latency Benchmark"
          subtitle="Execution Cost per Sample (Seconds)"
        >
          <img
            src="/figures/fig13_inference_latency_comparison.png"
            alt="Fig 13"
            className="w-full h-auto rounded-xl border border-[#242424]"
          />
        </ResearchCard>
      </div>
    </div>
  );
};
