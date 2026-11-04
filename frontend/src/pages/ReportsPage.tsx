import React, { useState, useEffect } from 'react';
import { fetchReportsContent } from '../services/api';
import { ReportsContentResponse } from '../types';
import { ResearchCard } from '../components/cards/ResearchCard';
import { StatusBadge } from '../components/status/StatusBadge';
import {
  FileText,
  Download,
  BookOpen,
  Scale,
  BarChart,
  AlertTriangle,
} from 'lucide-react';

export const ReportsPage: React.FC = () => {
  const [reports, setReports] = useState<ReportsContentResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'paper' | 'methodology' | 'results' | 'limitations'>(
    'paper'
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchReportsContent()
      .then((data) => setReports(data))
      .catch((err) => console.error('Failed to load reports:', err))
      .finally(() => setLoading(false));
  }, []);

  const tabs = [
    { id: 'paper', label: 'Final Research Paper', icon: FileText, key: 'final_paper' as const },
    { id: 'methodology', label: 'Formal Methodology', icon: Scale, key: 'methodology' as const },
    { id: 'results', label: 'Detailed Results', icon: BarChart, key: 'results' as const },
    { id: 'limitations', label: 'Limitations & Scope', icon: AlertTriangle, key: 'limitations' as const },
  ];

  return (
    <div className="space-y-6">
      {/* Artifact Export Header Card */}
      <ResearchCard
        title="Artifact Export Console"
        subtitle="Download Official Research Ledgers, Raw Empirical Outputs & Scientific Artifacts"
      >
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <a
            href="/api/reports/download/research_metrics.csv"
            download
            className="flex items-center justify-between bg-[#111111] hover:bg-[#1A1A1A] border border-[#242424] hover:border-[#333333] p-3.5 rounded-[16px] transition-all text-xs font-mono group"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#1A1A1A] group-hover:bg-[#202020] border border-[#242424] flex items-center justify-center transition-colors">
                <Download className="w-3.5 h-3.5 text-[#F5F5F5] group-hover:translate-y-0.5 transition-transform" />
              </div>
              <div>
                <div className="text-[#F5F5F5] font-medium tracking-tight">RESEARCH_METRICS.CSV</div>
                <div className="text-[10px] text-[#A1A1A1]">Full Quantitative Ledger</div>
              </div>
            </div>
            <StatusBadge variant="default" size="sm">CSV</StatusBadge>
          </a>

          <a
            href="/api/reports/download/metrics_summary.json"
            download
            className="flex items-center justify-between bg-[#111111] hover:bg-[#1A1A1A] border border-[#242424] hover:border-[#333333] p-3.5 rounded-[16px] transition-all text-xs font-mono group"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#1A1A1A] group-hover:bg-[#202020] border border-[#242424] flex items-center justify-center transition-colors">
                <Download className="w-3.5 h-3.5 text-[#F5F5F5] group-hover:translate-y-0.5 transition-transform" />
              </div>
              <div>
                <div className="text-[#F5F5F5] font-medium tracking-tight">METRICS_SUMMARY.JSON</div>
                <div className="text-[10px] text-[#A1A1A1]">Urgent Aggregate Results</div>
              </div>
            </div>
            <StatusBadge variant="default" size="sm">JSON</StatusBadge>
          </a>

          <a
            href="/api/reports/download/final_research_report.md"
            download
            className="flex items-center justify-between bg-[#111111] hover:bg-[#1A1A1A] border border-[#242424] hover:border-[#333333] p-3.5 rounded-[16px] transition-all text-xs font-mono group"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#1A1A1A] group-hover:bg-[#202020] border border-[#242424] flex items-center justify-center transition-colors">
                <Download className="w-3.5 h-3.5 text-[#F5F5F5] group-hover:translate-y-0.5 transition-transform" />
              </div>
              <div>
                <div className="text-[#F5F5F5] font-medium tracking-tight">FINAL_REPORT.MD</div>
                <div className="text-[10px] text-[#A1A1A1]">Full Scientific Dossier</div>
              </div>
            </div>
            <StatusBadge variant="accent" size="sm">MD</StatusBadge>
          </a>
        </div>
      </ResearchCard>

      {/* Interactive Pill Tabs */}
      <div className="flex items-center justify-start">
        <div className="inline-flex items-center bg-[#0A0A0A] border border-[#242424] p-1 rounded-full overflow-x-auto max-w-full">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-[#202020] text-[#F5F5F5] border border-[#333333] shadow-sm'
                    : 'text-[#A1A1A1] hover:text-[#F5F5F5] border border-transparent'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-[#F5F5F5]' : 'text-[#666666]'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Document Reader Container */}
      <ResearchCard
        title={tabs.find((t) => t.id === activeTab)?.label}
        subtitle="Peer-Review Research Documentation & Empirical Ledger"
      >
        {loading ? (
          <div className="py-20 text-center text-xs text-[#A1A1A1]">
            Loading research document...
          </div>
        ) : (
          <div className="bg-[#050505] border border-[#242424] rounded-[16px] p-6 max-h-[700px] overflow-y-auto">
            <div className="prose prose-invert max-w-none text-xs text-[#A1A1A1] leading-relaxed whitespace-pre-wrap font-mono selection:bg-[#202020] selection:text-[#F5F5F5]">
              {reports ? (
                activeTab === 'paper' ? (
                  reports.final_paper
                ) : activeTab === 'methodology' ? (
                  reports.methodology
                ) : activeTab === 'results' ? (
                  reports.results
                ) : (
                  reports.limitations
                )
              ) : (
                'Document not available.'
              )}
            </div>
          </div>
        )}
      </ResearchCard>
    </div>
  );
};
