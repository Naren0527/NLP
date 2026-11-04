import React from 'react';
import { cn } from '../../utils/cn';

interface ConnectionIndicatorProps {
  status: 'online' | 'offline' | 'checking';
  label?: string;
  className?: string;
}

export const ConnectionIndicator: React.FC<ConnectionIndicatorProps> = ({
  status,
  label,
  className,
}) => {
  const isOnline = status === 'online';
  const isChecking = status === 'checking';

  return (
    <div className={cn('inline-flex items-center gap-2 text-xs font-mono', className)}>
      <span className="relative flex h-2 w-2">
        {isOnline && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4ADE80] opacity-75" />
        )}
        <span
          className={cn(
            'relative inline-flex rounded-full h-2 w-2',
            isOnline ? 'bg-[#4ADE80]' : isChecking ? 'bg-[#FBBF24]' : 'bg-[#F87171]'
          )}
        />
      </span>
      <span className="text-[#A3A3A3]">
        {label || (isOnline ? 'BACKEND CONNECTED' : isChecking ? 'CONNECTING...' : 'BACKEND OFFLINE')}
      </span>
    </div>
  );
};
