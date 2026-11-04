export interface LanguageInfo {
  name: string;
  native: string;
  script: string;
  family: string;
}

export type LanguagesMap = Record<string, LanguageInfo>;

export interface BenchmarkLanguageRow {
  code: string;
  language: string;
  native: string;
  script: string;
  baseline_accuracy: number;
  mitigated_accuracy: number;
  accuracy_gain_pp: number;
  relative_error_reduction: number;
  verbalized_conf: number;
  statistical_conf: number;
  baseline_ece: number;
  mitigated_ece: number;
  ece_reduction: number;
  baseline_brier: number;
  mitigated_brier: number;
  brier_reduction: number;
}

export interface BenchmarkSummaryResponse {
  partitions: {
    calibration_split: { name: string; samples: number; purpose: string };
    validation_split: { name: string; samples: number; purpose: string };
    test_split: { name: string; samples: number; purpose: string };
    total_gpu_inferences: number;
  };
  indic_aggregate: {
    baseline_accuracy: number;
    mitigated_accuracy: number;
    accuracy_gain_pp: number;
    baseline_ece: number;
    mitigated_ece: number;
    ece_reduction: number;
  };
  languages: BenchmarkLanguageRow[];
}

export interface CuratedExample {
  id: string;
  question_en: string;
  domain: string;
  evidence: string;
  claims: string[];
  answers: Record<string, string>;
}

export interface InferenceRequest {
  query?: string;
  benchmark_question?: string;
  target_language: string;
  strategy: string;
  tau_threshold: number;
  benchmark_index: number;
}

export interface InferenceResponse {
  active_query: string;
  is_custom: boolean;
  target_language: string;
  language_name: string;
  script: string;
  strategy: string;
  tau_threshold: number;
  final_answer: string;
  is_abstained: boolean;
  has_gold_reference: boolean;
  verification_status: string;
  ground_truth_evidence: string | null;
  ground_truth_claims: string[];
  raw_v_conf: number;
  base_stat: number;
  cal_conf: number;
  english_pivot: string;
  cand_native: string;
  cand_pivot: string;
  arbitration_note: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  model: string;
  device: string;
  quantization: string;
  framework: string;
  uptime_seconds: number;
}

export interface ModelStatusResponse {
  model_name: string;
  architecture: string;
  parameters: string;
  context_length: number;
  precision: string;
  compute_dtype: string;
  device: string;
  vram_allocated_gb: number;
  vram_total_gb: number;
  temperature: number;
  top_p: number;
  recalibration_model: string;
  benchmark_dataset: string;
  inferences_recorded: number;
}

export interface CalibrationParamsResponse {
  model: string;
  formula: string;
  reference_condition: string;
  optimizer: string;
  bootstrap_replications: number;
  parameters: {
    slope_w: number;
    base_intercept_b0: number;
    deltas: Record<string, number>;
  };
  notes: string;
}

export interface ConsistencyAuditResponse {
  pairwise_comparisons_count: number;
  llm_judge_surface_agreement: number;
  true_grounded_consistency_min: number;
  true_grounded_consistency_max: number;
  error_taxonomy: Array<{
    id: string;
    title: string;
    prevalence: string;
    description: string;
    severity: string;
  }>;
  language_pairs: Array<{
    pair: string;
    surface_agreement: number;
    grounded_consistency: number;
    mitigated_consistency: number;
  }>;
}

export interface StrategyItem {
  strategy: string;
  validation_accuracy: number;
  validation_brier: number;
  average_latency_sec: number;
  status: string;
  rationale: string;
}

export interface FrontierPoint {
  tau: number;
  coverage: number;
  accuracy: number;
  selective_risk: number;
}

export interface StrategiesBenchmarksResponse {
  validation_strategy_selection: StrategyItem[];
  strategy_e_frontier: FrontierPoint[];
}

export interface ReportsContentResponse {
  final_paper: string;
  methodology: string;
  results: string;
  limitations: string;
}
