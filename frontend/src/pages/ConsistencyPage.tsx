import React, { useState, useEffect } from 'react';
import { fetchConsistencyAudit } from '../services/api';
import { ConsistencyAuditResponse } from '../types';
import { ResearchCard } from '../components/cards/ResearchCard';
import { MetricCard } from '../components/cards/MetricCard';
import { StatusBadge } from '../components/status/StatusBadge';
import { formatPercent } from '../utils/formatters';

export const ConsistencyPage: React.FC = () => {
  const [audit, setAudit] = useState<ConsistencyAuditResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchConsistencyAudit()
      .then((res) => setAudit(res))
      .catch((err) => console.error('Consistency audit error:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* 2 Big Headline Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <MetricCard
          title="LLM Judge Surface Agreement"
          value="88.4%"
          progressPercent={88.4}
          sub="Automated LLM judges score pairs as agreeing based on conversational fluency"
          badge="SURFACE ILLUSION"
          variant="default"
        />
        <MetricCard
          title="True Grounded Consistency"
          value="28.7% – 37.3%"
          progressPercent={33.0}
          sub="Factual grounding verified against atomic gold evidence claims"
          badge="VERIFIED GROUNDING"
          variant="success"
        />
      </div>

      {/* Main Visual: Cross-Lingual Consistency Figure */}
      <ResearchCard
        title="Fig 08 // Cross-Lingual Semantic Agreement Before & After Mitigation"
        subtitle="750 Pairwise Comparisons Across Language Dyads"
      >
        <div className="rounded-[18px] border border-[#242424] overflow-hidden bg-[#111111]">
          <img
            src="/figures/fig8_cross_lingual_consistency.png"
            alt="Cross-Lingual Consistency"
            className="w-full h-auto object-contain"
          />
        </div>
      </ResearchCard>

      {/* Language Dyad Comparison Table */}
      <ResearchCard
        title="Pairwise Language Dyad Consistency Audit"
        subtitle="Symmetrical Evaluation across English-Indic Language Pairs"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse font-mono">
            <thead>
              <tr className="border-b border-[#242424] text-[#666666] uppercase tracking-wider">
                <th className="py-2.5 px-3">Language Dyad</th>
                <th className="py-2.5 px-3 text-right">Surface LLM Judge Agreement</th>
                <th className="py-2.5 px-3 text-right">Baseline Grounded Consistency</th>
                <th className="py-2.5 px-3 text-right">Mitigated Grounded Consistency</th>
                <th className="py-2.5 px-3 text-right">Factual Recovery Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#242424] text-[#F5F5F5]">
              {audit?.language_pairs.map((p) => {
                const delta = (p.mitigated_consistency - p.grounded_consistency) * 100;
                return (
                  <tr key={p.pair} className="hover:bg-[#111111] transition-colors">
                    <td className="py-3 px-3 font-semibold text-[#F5F5F5]">
                      {p.pair}
                    </td>
                    <td className="py-3 px-3 text-right text-[#A1A1A1]">
                      {formatPercent(p.surface_agreement)}
                    </td>
                    <td className="py-3 px-3 text-right text-[#F87171] font-semibold">
                      {formatPercent(p.grounded_consistency)}
                    </td>
                    <td className="py-3 px-3 text-right text-[#4ADE80] font-semibold">
                      {formatPercent(p.mitigated_consistency)}
                    </td>
                    <td className="py-3 px-3 text-right text-[#4ADE80]">
                      +{delta.toFixed(1)} pp
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ResearchCard>

      {/* Qualitative Error Taxonomy */}
      <ResearchCard
        title="Qualitative Error Taxonomy: Why LLM Judges Fail"
        subtitle="Empirical Diagnosis of Failure Modes in Cross-Lingual Evaluation"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {audit?.error_taxonomy.map((diag) => (
            <div
              key={diag.id}
              className="bg-[#111111] border border-[#242424] p-5 rounded-[20px] flex flex-col justify-between space-y-3"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-mono text-[#A1A1A1]">{diag.id}</span>
                  <StatusBadge
                    variant={diag.severity === 'CRITICAL' ? 'error' : 'default'}
                    size="sm"
                  >
                    {diag.prevalence} PREVALENCE
                  </StatusBadge>
                </div>
                <h4 className="text-xs font-semibold text-[#F5F5F5] mb-1.5 leading-snug">
                  {diag.title}
                </h4>
                <p className="text-xs text-[#666666] leading-relaxed font-sans">
                  {diag.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </ResearchCard>
    </div>
  );
};
