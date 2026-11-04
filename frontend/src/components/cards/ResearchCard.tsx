import React from 'react';
import { cn } from '../../utils/cn';

interface ResearchCardProps {
  title?: string;
  subtitle?: string;
  badge?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  headerAction?: React.ReactNode;
}

export const ResearchCard: React.FC<ResearchCardProps> = ({
  title,
  subtitle,
  badge,
  children,
  className,
  headerAction,
}) => {
  return (
    <div
      className={cn(
        'bg-[#0A0A0A] border border-[#242424] rounded-[24px] p-6 md:p-7 transition-colors duration-150',
        className
      )}
    >
      {(title || subtitle || badge || headerAction) && (
        <div className="flex items-start justify-between gap-4 pb-4 mb-5 border-b border-[#242424]">
          <div>
            <div className="flex items-center gap-2.5">
              {title && (
                <h3 className="text-sm font-semibold text-[#F5F5F5] tracking-wide uppercase">
                  {title}
                </h3>
              )}
              {badge}
            </div>
            {subtitle && (
              <p className="text-xs text-[#666666] mt-1 leading-relaxed">
                {subtitle}
              </p>
            )}
          </div>
          {headerAction && <div className="shrink-0">{headerAction}</div>}
        </div>
      )}
      <div>{children}</div>
    </div>
  );
};
