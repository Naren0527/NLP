import React from 'react';
import { ConnectionIndicator } from '../status/ConnectionIndicator';

interface HeaderProps {
  pageTitle: string;
  pageSubtitle?: string;
  backendOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  pageTitle,
  pageSubtitle,
  backendOnline,
}) => {
  return (
    <header className="h-16 bg-[#212121] border-b border-[#383838] px-6 flex items-center justify-between sticky top-0 z-20 shrink-0">
      <div className="flex items-center gap-3">
        <div>
          <h2 className="text-sm font-semibold text-[#ECECEC] tracking-wide uppercase">
            {pageTitle}
          </h2>
          {pageSubtitle && (
            <p className="text-xs text-[#737373] truncate hidden sm:block">
              {pageSubtitle}
            </p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden md:flex items-center gap-2 text-xs font-mono text-[#737373] bg-[#262626] border border-[#383838] px-2.5 py-1 rounded">
          <span className="text-[#ECECEC]">GEMMA 3 4B-IT</span>
          <span className="text-[#383838]">|</span>
          <span>RTX 4050</span>
          <span className="text-[#383838]">|</span>
          <span className="text-[#8B8CF7]">NF4</span>
        </div>
        <ConnectionIndicator status={backendOnline ? 'online' : 'offline'} />
      </div>
    </header>
  );
};
