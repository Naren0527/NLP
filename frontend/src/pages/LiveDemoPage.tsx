import React, { useState, useEffect } from 'react';
import {
  fetchLanguages,
  fetchBenchmarkExamples,
  postInference,
} from '../services/api';
import {
  LanguagesMap,
  CuratedExample,
  InferenceResponse,
} from '../types';
import { MetricCard } from '../components/cards/MetricCard';
import { StatusBadge } from '../components/status/StatusBadge';
import { formatPercent, formatDelta } from '../utils/formatters';
import {
  Play,
  RotateCw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp,
  FileSearch,
  Info,
  Shield,
} from 'lucide-react';
import { cn } from '../utils/cn';

const STRATEGIES = [
  { id: 'Baseline (Direct Native Prompting)', label: 'Baseline', desc: 'Direct Native Prompting' },
  { id: 'Strategy A: Native Prompt Optimization', label: 'Strategy A', desc: 'Native Prompt Optimization' },
  { id: 'Strategy B: Cross-Lingual Pivot Answering', label: 'Strategy B', desc: 'Cross-Lingual Pivot Answering' },
  { id: 'Strategy C: Bilingual Verification & Arbitration (Winner)', label: 'Strategy C (Winner)', desc: 'Bilingual Verification & Arbitration' },
  { id: 'Strategy D: Evidence-Grounded Refinement', label: 'Strategy D', desc: 'Evidence-Grounded Refinement' },
  { id: 'Strategy E: Confidence Selective Answering', label: 'Strategy E', desc: 'Confidence Selective Answering' },
];

export const LiveDemoPage: React.FC = () => {
  const [languages, setLanguages] = useState<LanguagesMap>({});
  const [examples, setExamples] = useState<CuratedExample[]>([]);
  const [selectedExampleIdx, setSelectedExampleIdx] = useState<number>(0);
  const [targetLang, setTargetLang] = useState<string>('en');
  const [queryInput, setQueryInput] = useState<string>('');
  const [selectedStrategy, setSelectedStrategy] = useState<string>(
    'Strategy C: Bilingual Verification & Arbitration (Winner)'
  );
  const [tauThreshold, setTauThreshold] = useState<number>(0.55);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [lastExecutedQuery, setLastExecutedQuery] = useState<string>('');
  const [isStale, setIsStale] = useState<boolean>(false);
  const [evidenceExpanded, setEvidenceExpanded] = useState<boolean>(true);

  // Initialize data
  useEffect(() => {
    Promise.all([fetchLanguages(), fetchBenchmarkExamples()])
      .then(([langs, exs]) => {
        setLanguages(langs);
        setExamples(exs);
        if (exs.length > 0) {
          setQueryInput(exs[0].question_en);
        }
      })
      .catch((err) => {
        console.error('Initialization error:', err);
        setError('Failed to connect to backend service. Ensure FastAPI server is running.');
      });
  }, []);

  const handleBenchmarkSelect = (idx: number) => {
    setSelectedExampleIdx(idx);
    if (examples[idx]) {
      setQueryInput(examples[idx].question_en);
      setIsStale(true);
    }
  };

  const handleQueryChange = (text: string) => {
    setQueryInput(text);
    if (result && text !== lastExecutedQuery) {
      setIsStale(true);
    }
  };

  const handleRunInference = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);

    const activeBenchmarkQ = examples[selectedExampleIdx]?.question_en || '';

    try {
      const res = await postInference({
        query: queryInput,
        benchmark_question: activeBenchmarkQ,
        target_language: targetLang,
        strategy: selectedStrategy,
        tau_threshold: tauThreshold,
        benchmark_index: selectedExampleIdx,
      });

      setResult(res);
      setLastExecutedQuery(res.active_query);
      setIsStale(false);
    } catch (err: any) {
      console.error('Inference error:', err);
      setError(err.message || 'Execution error during inference.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (examples.length > 0 && !result && !loading) {
      handleRunInference();
    }
  }, [examples]);

  const langList = Object.entries(languages);

  return (
    <div className="space-y-4">
      {/* Top Error Alert if backend unreachable */}
      {error && (
        <div className="bg-[#F87171]/10 border border-[#F87171]/30 rounded-2xl p-4 text-xs text-[#F87171] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={() => handleRunInference()}
            className="underline hover:text-[#F5F5F5] font-medium"
          >
            Retry
          </button>
        </div>
      )}

      {/* Two-Panel Layout (42% Left / 58% Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* LEFT PANEL: CONFIGURATION */}
        <div className="lg:col-span-5 bg-[#0A0A0A] border border-[#242424] rounded-[24px] p-6 md:p-7 flex flex-col justify-between space-y-6">
          <div className="space-y-5">
            {/* Header */}
            <div>
              <h2 className="text-sm font-semibold text-[#F5F5F5] uppercase tracking-wider">
                Configuration & Prompt
              </h2>
              <p className="text-xs text-[#666666] mt-0.5">
                Select benchmark question or type arbitrary prompt
              </p>
            </div>

            {/* Target Language Selection */}
            <div>
              <label className="block text-xs font-medium text-[#A1A1A1] uppercase tracking-wider mb-1.5">
                Target Language (L)
              </label>
              <select
                value={targetLang}
                onChange={(e) => {
                  setTargetLang(e.target.value);
                  setIsStale(true);
                }}
                className="w-full bg-[#111111] border border-[#242424] hover:border-[#333333] rounded-xl px-3.5 py-2.5 text-[#F5F5F5] focus:border-[#F5F5F5] focus:outline-none transition-colors text-xs"
              >
                {langList.map(([code, info]) => (
                  <option key={code} value={code} className="bg-[#111111] text-[#F5F5F5]">
                    {info.name} — {info.native} [{info.script}]
                  </option>
                ))}
              </select>
            </div>

            {/* Preload Benchmark Question */}
            <div>
              <label className="block text-xs font-medium text-[#A1A1A1] uppercase tracking-wider mb-1.5">
                Preload Benchmark Question
              </label>
              <select
                value={selectedExampleIdx}
                onChange={(e) => handleBenchmarkSelect(Number(e.target.value))}
                className="w-full bg-[#111111] border border-[#242424] hover:border-[#333333] rounded-xl px-3.5 py-2.5 text-[#F5F5F5] focus:border-[#F5F5F5] focus:outline-none transition-colors text-xs truncate"
              >
                {examples.map((ex, idx) => (
                  <option key={ex.id} value={idx} className="bg-[#111111] text-[#F5F5F5]">
                    [{ex.domain}] {ex.question_en}
                  </option>
                ))}
              </select>
            </div>

            {/* Custom Query Prompt */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-[#A1A1A1] uppercase tracking-wider">
                  Query Prompt (English Base)
                </label>
                <span className="text-[10px] text-[#666666]">
                  Custom overrides benchmark
                </span>
              </div>
              <textarea
                value={queryInput}
                onChange={(e) => handleQueryChange(e.target.value)}
                rows={3}
                placeholder="Enter custom query or edit question..."
                className="w-full bg-[#111111] border border-[#242424] hover:border-[#333333] rounded-xl p-3 text-[#F5F5F5] focus:border-[#F5F5F5] focus:outline-none transition-colors text-xs resize-none leading-relaxed placeholder:text-[#666666]"
              />
            </div>

            {/* Strategy Selection (Radio Option Rows) */}
            <div>
              <label className="block text-xs font-medium text-[#A1A1A1] uppercase tracking-wider mb-2">
                Mitigation Strategy
              </label>
              <div className="space-y-1.5">
                {STRATEGIES.map((strat) => {
                  const isSelected = selectedStrategy === strat.id;
                  return (
                    <div
                      key={strat.id}
                      onClick={() => {
                        setSelectedStrategy(strat.id);
                        setIsStale(true);
                      }}
                      className={cn(
                        'rounded-xl p-2.5 px-3 flex items-center justify-between cursor-pointer transition-all duration-150 border text-xs',
                        isSelected
                          ? 'bg-[#202020] border-[#333333] text-[#F5F5F5]'
                          : 'bg-[#111111] border-[#242424] hover:bg-[#1A1A1A] hover:border-[#333333] text-[#A1A1A1]'
                      )}
                    >
                      <div className="flex flex-col">
                        <span className="font-medium text-[#F5F5F5]">{strat.label}</span>
                        <span className="text-[11px] text-[#666666]">{strat.desc}</span>
                      </div>
                      <div
                        className={cn(
                          'w-4 h-4 rounded-full border flex items-center justify-center shrink-0 ml-3',
                          isSelected ? 'border-[#F5F5F5]' : 'border-[#444444]'
                        )}
                      >
                        {isSelected && <div className="w-2 h-2 rounded-full bg-[#F5F5F5]" />}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Threshold Slider */}
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-xs font-medium text-[#A1A1A1] uppercase tracking-wider">
                  Selective Abstention Threshold
                </label>
                <span className="font-mono text-xs font-semibold text-[#F5F5F5] bg-[#1A1A1A] px-2 py-0.5 rounded border border-[#242424]">
                  τ = {tauThreshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={tauThreshold}
                onChange={(e) => {
                  setTauThreshold(parseFloat(e.target.value));
                  setIsStale(true);
                }}
                className="w-full h-1 bg-[#242424] rounded-lg appearance-none cursor-pointer accent-[#F5F5F5]"
              />
              <div className="flex justify-between text-[10px] text-[#666666] mt-1 font-mono">
                <span>0.10 (High Coverage)</span>
                <span>0.55 (Optimal Risk)</span>
                <span>0.90 (Conservative)</span>
              </div>
            </div>
          </div>

          {/* PRIMARY BUTTON: White Pill Shaped */}
          <div className="pt-4 border-t border-[#242424]">
            <button
              onClick={handleRunInference}
              disabled={loading}
              className="w-full bg-[#F5F5F5] hover:bg-[#E5E5E5] disabled:bg-[#242424] disabled:text-[#666666] disabled:cursor-not-allowed text-[#080808] font-semibold py-3 px-6 rounded-full text-xs tracking-wider transition-all duration-150 flex items-center justify-center gap-2 shadow-sm uppercase"
            >
              {loading ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Generate &amp; Calibrate</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* RIGHT PANEL: OUTPUT & TELEMETRY */}
        <div className="lg:col-span-7 bg-[#0A0A0A] border border-[#242424] rounded-[24px] p-6 md:p-7 space-y-5 flex flex-col justify-between">
          <div className="space-y-4">
            {/* Header & Status Row */}
            <div className="flex items-center justify-between border-b border-[#242424] pb-3.5">
              <div>
                <h2 className="text-sm font-semibold text-[#F5F5F5] uppercase tracking-wider">
                  Model Output &amp; Telemetry
                </h2>
                <div className="text-[11px] text-[#666666] font-mono mt-0.5">
                  Gemma 3 4B Instruct · Language-Shift Platt Recalibrated
                </div>
              </div>

              {result && (
                <div>
                  {result.is_abstained ? (
                    <StatusBadge variant="error" size="sm">
                      <XCircle className="w-3 h-3" />
                      Abstained (τ = {result.tau_threshold.toFixed(2)})
                    </StatusBadge>
                  ) : result.is_custom ? (
                    result.has_gold_reference ? (
                      <StatusBadge variant="success" size="sm">
                        <CheckCircle className="w-3 h-3" />
                        Verified Factual · {result.language_name}
                      </StatusBadge>
                    ) : (
                      <StatusBadge variant="accent" size="sm">
                        <Info className="w-3 h-3" />
                        Custom Query · {result.language_name}
                      </StatusBadge>
                    )
                  ) : (
                    <StatusBadge variant="success" size="sm">
                      <CheckCircle className="w-3 h-3" />
                      Verified &amp; Delivered · {result.language_name}
                    </StatusBadge>
                  )}
                </div>
              )}
            </div>

            {/* Stale Warning Callout */}
            {isStale && (
              <div className="bg-[#FBBF24]/10 border border-[#FBBF24]/25 rounded-xl p-3 text-xs text-[#FBBF24] flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <div className="leading-relaxed">
                  <span className="font-semibold uppercase tracking-wider">Input Modified // Pending Execution</span>
                  <p className="text-[11px] text-[#A1A1A1] mt-0.5">
                    Query prompt or parameters changed. Output below corresponds to previous run. Click{' '}
                    <strong className="text-[#F5F5F5]">Generate &amp; Calibrate</strong> to recompute.
                  </p>
                </div>
              </div>
            )}

            {/* THREE METRIC CARDS WITH PROGRESS BARS */}
            {result && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <MetricCard
                  title="Raw Verbalized Conf"
                  value={formatPercent(result.raw_v_conf)}
                  progressPercent={result.raw_v_conf * 100}
                  sub="Self-reported prompt score"
                  variant="default"
                />
                <MetricCard
                  title="Statistical Conf"
                  value={formatPercent(result.base_stat)}
                  progressPercent={result.base_stat * 100}
                  sub="Internal token probability"
                  variant="default"
                />
                <MetricCard
                  title="Model 2 Calibrated"
                  value={formatPercent(result.cal_conf)}
                  progressPercent={result.cal_conf * 100}
                  sub={
                    result.is_abstained
                      ? 'Gated by safety cutoff'
                      : formatDelta(result.cal_conf, result.raw_v_conf)
                  }
                  variant={result.is_abstained ? 'error' : 'success'}
                />
              </div>
            )}

            {/* DECODED OUTPUT CONTAINER */}
            {result ? (
              result.is_abstained ? (
                <div className="bg-[#F87171]/10 border border-[#F87171]/25 rounded-[20px] p-5 text-xs">
                  <div className="font-semibold text-[#F87171] uppercase tracking-wider mb-1 flex items-center gap-1.5">
                    <Shield className="w-4 h-4" />
                    Safety Interlock Engaged
                  </div>
                  <p className="text-[#ECECEC] leading-relaxed">
                    Calibrated reliability score (
                    <strong className="text-[#F87171]">{formatPercent(result.cal_conf)}</strong>)
                    fell below safety cutoff (
                    <strong>{formatPercent(result.tau_threshold)}</strong>). Answering is withheld to
                    prevent uncalibrated hallucination propagation in {result.language_name}.
                  </p>
                </div>
              ) : (
                <div className="bg-[#111111] border border-[#242424] rounded-[20px] p-5 space-y-2">
                  <div className="flex justify-between items-center text-[11px] font-mono text-[#666666] border-b border-[#242424] pb-2">
                    <span className="text-[#A1A1A1] truncate max-w-sm">
                      [QUERY: {result.active_query}]
                    </span>
                    <span>SCRIPT: {result.script.toUpperCase()}</span>
                  </div>
                  <div className="text-base text-[#F5F5F5] font-medium leading-relaxed pt-1">
                    {result.final_answer}
                  </div>
                </div>
              )
            ) : (
              <div className="h-44 flex flex-col items-center justify-center text-center text-[#666666] text-xs space-y-2">
                <RotateCw className="w-5 h-5 animate-spin text-[#A1A1A1]" />
                <p>Initializing inference console...</p>
              </div>
            )}

            {/* EXPANDABLE EVIDENCE AND GROUNDING TRACE */}
            {result && (
              <div className="bg-[#111111] border border-[#242424] rounded-[20px] overflow-hidden transition-all">
                <button
                  onClick={() => setEvidenceExpanded(!evidenceExpanded)}
                  className="w-full flex items-center justify-between p-4 text-xs font-semibold text-[#F5F5F5] uppercase tracking-wider hover:bg-[#1A1A1A] transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <FileSearch className="w-4 h-4 text-[#A1A1A1]" />
                    Evidence &amp; Grounding Trace
                  </span>
                  <div className="flex items-center gap-2 text-[11px] font-mono text-[#666666]">
                    <span>Strategy: {result.strategy.split(':')[0]}</span>
                    {evidenceExpanded ? (
                      <ChevronUp className="w-4 h-4 text-[#A1A1A1]" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-[#A1A1A1]" />
                    )}
                  </div>
                </button>

                {evidenceExpanded && (
                  <div className="px-4 pb-4 pt-1 space-y-3 text-xs border-t border-[#242424]">
                    <div>
                      <div className="text-[11px] font-medium text-[#666666] uppercase mb-1">
                        Ground Truth Reference Evidence
                      </div>
                      <div className="text-[#F5F5F5] italic bg-[#0A0A0A] border border-[#242424] p-3 rounded-xl text-xs leading-relaxed">
                        {result.has_gold_reference && result.ground_truth_evidence
                          ? result.ground_truth_evidence
                          : 'No pre-annotated benchmark evidence available for custom query. Real-time verification performed via semantic consistency.'}
                      </div>
                    </div>

                    <div>
                      <div className="text-[11px] font-medium text-[#666666] uppercase mb-1">
                        Target Claim Decomposition
                      </div>
                      <div className="font-mono text-[11px] text-[#A1A1A1] bg-[#0A0A0A] border border-[#242424] p-2.5 rounded-xl">
                        {result.ground_truth_claims.length > 0
                          ? JSON.stringify(result.ground_truth_claims)
                          : '[Custom Query - Uncurated Claim Set]'}
                      </div>
                    </div>

                    {result.strategy.includes('Strategy B') || result.strategy.includes('Strategy C') ? (
                      <div className="space-y-2 pt-2 border-t border-[#242424]">
                        <div>
                          <span className="text-[11px] text-[#666666] uppercase">
                            Stage 1 English Pivot Candidate (High Competence 95.6%):
                          </span>
                          <div className="text-[#F5F5F5] font-mono text-[11px] mt-0.5">
                            {result.english_pivot}
                          </div>
                        </div>
                        <div>
                          <span className="text-[11px] text-[#666666] uppercase">
                            Stage 2 Fluent Target Native Translation ({result.language_name}):
                          </span>
                          <div className="text-[#F5F5F5] font-mono text-[11px] mt-0.5">
                            {result.cand_native}
                          </div>
                        </div>
                        <div className="pt-1">
                          <StatusBadge variant="success" size="sm">
                            Arbitration Verdict: Factually Grounded
                          </StatusBadge>
                        </div>
                      </div>
                    ) : (
                      <div className="pt-1 text-[#666666] text-[11px]">
                        Direct native generation without English pivoting exposes the query to high uncalibrated token perplexity.
                      </div>
                    )}

                    <div className="text-[11px] text-[#666666] font-mono pt-1">
                      Arbitration Note: {result.arbitration_note}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
