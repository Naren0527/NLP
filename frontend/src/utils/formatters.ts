export function formatPercent(value: number | undefined | null, decimals = 1): string {
  if (value === undefined || value === null || isNaN(value)) return '0.0%';
  const num = value > 1 ? value : value * 100;
  return `${num.toFixed(decimals)}%`;
}

export function formatPp(value: number | undefined | null, decimals = 1): string {
  if (value === undefined || value === null || isNaN(value)) return '+0.0 pp';
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(decimals)} pp`;
}

export function formatDelta(calVal: number, rawVal: number): string {
  const c = calVal > 1 ? calVal : calVal * 100;
  const r = rawVal > 1 ? rawVal : rawVal * 100;
  const d = c - r;
  const prefix = d > 0 ? '+' : '';
  return `${prefix}${d.toFixed(1)}% vs. Raw Verbalized`;
}

export function formatDecimal(value: number | undefined | null, decimals = 4): string {
  if (value === undefined || value === null || isNaN(value)) return '0.0000';
  return value.toFixed(decimals);
}
