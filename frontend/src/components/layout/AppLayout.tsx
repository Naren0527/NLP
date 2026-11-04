import React, { useState, useEffect } from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { fetchHealth } from '../../services/api';
import { cn } from '../../utils/cn';
import { Activity } from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: 'Overview', exact: true },
  { to: '/live-demo', label: 'Live Demo' },
  { to: '/benchmark', label: 'Benchmark' },
  { to: '/calibration', label: 'Calibration' },
  { to: '/consistency', label: 'Consistency' },
  { to: '/strategies', label: 'Strategies' },
  { to: '/reports', label: 'Reports' },
];

export const AppLayout: React.FC = () => {
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const location = useLocation();

  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      try {
        await fetchHealth();
        if (mounted) setBackendOnline(true);
      } catch {
        if (mounted) setBackendOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen w-full bg-[#050505] text-[#F5F5F5] flex flex-col font-sans antialiased selection:bg-[#333333] selection:text-[#F5F5F5]">
      {/* Top Header & Floating Navigation Area */}
      <header className="sticky top-0 z-30 w-full pt-4 pb-2 px-4 md:px-8 bg-[#050505]/95 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
          {/* Left: Minimal Brand */}
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-6 h-6 rounded-full bg-[#1A1A1A] border border-[#242424] flex items-center justify-center text-[#F5F5F5]">
              <Activity className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-xs font-semibold text-[#F5F5F5] tracking-tight leading-none">
                Cross-Lingual Calibration
              </div>
              <div className="text-[10px] text-[#666666] tracking-wider uppercase leading-none mt-0.5">
                Research Workspace
              </div>
            </div>
          </div>

          {/* Center: Floating Pill Navigation Capsule */}
          <nav className="flex justify-center shrink-0">
            <div className="bg-[#0A0A0A] border border-[#242424] rounded-full p-1 shadow-xl flex items-center gap-0.5 overflow-x-auto max-w-[calc(100vw-2rem)]">
              {NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.exact}
                  className={({ isActive }) =>
                    cn(
                      'text-xs font-medium px-3.5 py-1.5 rounded-full whitespace-nowrap transition-all duration-150',
                      isActive
                        ? 'bg-[#202020] text-[#F5F5F5] shadow-sm'
                        : 'text-[#A1A1A1] hover:text-[#F5F5F5] hover:bg-[#1A1A1A]'
                    )
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          </nav>

          {/* Right: Minimal Status Telemetry */}
          <div className="hidden lg:flex items-center gap-2.5 shrink-0">
            <div className="flex items-center gap-2 bg-[#0A0A0A] border border-[#242424] px-3 py-1 rounded-full text-[11px] font-mono text-[#A1A1A1]">
              <span className="flex items-center gap-1.5">
                <span
                  className={cn(
                    'w-1.5 h-1.5 rounded-full',
                    backendOnline ? 'bg-[#4ADE80]' : 'bg-[#F87171]'
                  )}
                />
                <span className="text-[#F5F5F5]">Gemma 3 4B</span>
              </span>
              <span className="text-[#333333]">·</span>
              <span>RTX 4050</span>
              <span className="text-[#333333]">·</span>
              <span className="text-[#666666]">{backendOnline ? 'Online' : 'Offline'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-4 md:px-6 py-4 md:py-6">
        <Outlet context={{ backendOnline }} />
      </main>
    </div>
  );
};
