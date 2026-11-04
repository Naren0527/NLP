import {
  HealthResponse,
  ModelStatusResponse,
  LanguagesMap,
  BenchmarkSummaryResponse,
  CuratedExample,
  InferenceRequest,
  InferenceResponse,
  CalibrationParamsResponse,
  ConsistencyAuditResponse,
  StrategiesBenchmarksResponse,
  ReportsContentResponse,
} from '../types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatusResponse> {
  const res = await fetch(`${API_BASE}/model/status`);
  if (!res.ok) throw new Error(`Model status failed: ${res.statusText}`);
  return res.json();
}

export async function fetchLanguages(): Promise<LanguagesMap> {
  const res = await fetch(`${API_BASE}/languages`);
  if (!res.ok) throw new Error(`Failed to fetch languages: ${res.statusText}`);
  return res.json();
}

export async function fetchBenchmarkSummary(): Promise<BenchmarkSummaryResponse> {
  const res = await fetch(`${API_BASE}/benchmark/summary`);
  if (!res.ok) throw new Error(`Failed to fetch benchmark summary: ${res.statusText}`);
  return res.json();
}

export async function fetchBenchmarkExamples(): Promise<CuratedExample[]> {
  const res = await fetch(`${API_BASE}/benchmark/examples`);
  if (!res.ok) throw new Error(`Failed to fetch benchmark examples: ${res.statusText}`);
  return res.json();
}

export async function postInference(payload: InferenceRequest): Promise<InferenceResponse> {
  const res = await fetch(`${API_BASE}/inference`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Inference failed (${res.status}): ${errText}`);
  }
  return res.json();
}

export async function fetchCalibrationParams(): Promise<CalibrationParamsResponse> {
  const res = await fetch(`${API_BASE}/calibration/parameters`);
  if (!res.ok) throw new Error(`Failed to fetch calibration params: ${res.statusText}`);
  return res.json();
}

export async function fetchConsistencyAudit(): Promise<ConsistencyAuditResponse> {
  const res = await fetch(`${API_BASE}/consistency/audit`);
  if (!res.ok) throw new Error(`Failed to fetch consistency audit: ${res.statusText}`);
  return res.json();
}

export async function fetchStrategiesBenchmarks(): Promise<StrategiesBenchmarksResponse> {
  const res = await fetch(`${API_BASE}/strategies/benchmarks`);
  if (!res.ok) throw new Error(`Failed to fetch strategies benchmarks: ${res.statusText}`);
  return res.json();
}

export async function fetchReportsContent(): Promise<ReportsContentResponse> {
  const res = await fetch(`${API_BASE}/reports/content`);
  if (!res.ok) throw new Error(`Failed to fetch reports content: ${res.statusText}`);
  return res.json();
}
