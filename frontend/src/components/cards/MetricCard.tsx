import React from 'react';
import { cn } from '../../utils/cn';

interface MetricCardProps {
  title: string;
  value: string;
  progressPercent?: number; // 0 to 100
  sub?: string;
  badge?: string;
  variant?: 'default' | 'accent' | 'success' | 'warning' | 'error';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  progressPercent,
  sub,
  badge,
  variant = 'default',
  className,
}) => {
  // If progressPercent is not explicitly passed, try to parse from value string if it ends with %
  let pct = progressPercent;
  if (pct === undefined && value.includes('%')) {
    const parsed = parseFloat(value.replace(/[^0-9.]/g, ''));
    if (!isNaN(parsed)) {
      pct = Math.min(100, Math.max(0, parsed));
    }
  }

  const progressBarColor = {
    default: 'bg-[#A1A1A1]',
    accent: 'bg-[#F5F5F5]',
    success: 'bg-[#4ADE80]',
    warning: 'bg-[#FBBF24]',
    error: 'bg-[#F87171]',
  };

  return (
    <div
      className={cn(
        'bg-[#111111] border border-[#242424] hover:border-[#333333] rounded-[20px] p-4 flex flex-col justify-between transition-colors duration-150',
        className
      )}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="text-[11px] font-medium text-[#A1A1A1] uppercase tracking-wider truncate">
          {title}
        </span>
        {badge && (
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#1A1A1A] border border-[#242424] text-[#A1A1A1] tracking-wide shrink-0">
            {badge}
          </span>
        )}
      </div>

      <div className="text-2xl font-semibold text-[#F5F5F5] tracking-tight my-0.5">
        {value}
      </div>

      {pct !== undefined && (
        <div className="w-full h-1 bg-[#1A1A1A] rounded-full overflow-hidden my-2">
          <div
            className={cn('h-full rounded-full transition-all duration-300', progressBarColor[variant])}
            style={{ width: `${pct}%` }}
          />
        </div>
      )}

      {sub && (
        <div className="text-[11px] text-[#666666] leading-relaxed truncate" title={sub}>
          {sub}
        </div>
      )}
    </div>
  );
};
