import React from 'react';
import { cn } from '../../utils/cn';

interface StatusBadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'accent';
  className?: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  children,
  variant = 'default',
  className,
  size = 'md',
}) => {
  const variantStyles = {
    default: 'bg-[#1A1A1A] text-[#A1A1A1] border-[#242424]',
    success: 'bg-[#4ADE80]/10 text-[#4ADE80] border-[#4ADE80]/20',
    warning: 'bg-[#FBBF24]/10 text-[#FBBF24] border-[#FBBF24]/20',
    error: 'bg-[#F87171]/10 text-[#F87171] border-[#F87171]/20',
    accent: 'bg-[#202020] text-[#F5F5F5] border-[#333333]',
  };

  const sizeStyles = {
    sm: 'text-[11px] px-2.5 py-0.5',
    md: 'text-xs px-3 py-1',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 font-medium rounded-full border tracking-wide font-sans',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
    >
      {children}
    </span>
  );
};
