import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  PlayCircle,
  BarChart3,
  Sliders,
  ShieldCheck,
  Layers,
  FileText,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  Activity,
} from 'lucide-react';
import { cn } from '../../utils/cn';

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  backendOnline: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onToggleCollapse,
  backendOnline,
}) => {
  const navItems = [
    { to: '/', label: 'Overview', icon: LayoutDashboard, exact: true },
    { to: '/live-demo', label: 'Live Demo', icon: PlayCircle },
    { to: '/benchmark', label: 'Benchmark', icon: BarChart3 },
    { to: '/calibration', label: 'Calibration', icon: Sliders },
    { to: '/consistency', label: 'Consistency', icon: ShieldCheck },
    { to: '/strategies', label: 'Strategies', icon: Layers },
    { to: '/reports', label: 'Reports', icon: FileText },
  ];

  return (
    <aside
      className={cn(
        'bg-[#171717] border-r border-[#383838] flex flex-col justify-between transition-all duration-200 select-none z-30 shrink-0 h-screen sticky top-0',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Brand Header */}
      <div>
        <div className="h-16 flex items-center justify-between px-4 border-b border-[#383838]">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-8 h-8 rounded bg-[#8B8CF7]/15 border border-[#8B8CF7]/30 flex items-center justify-center shrink-0 text-[#8B8CF7]">
              <Activity className="w-4 h-4" />
            </div>
            {!collapsed && (
              <div className="min-w-0">
                <h1 className="text-sm font-semibold text-[#ECECEC] truncate tracking-tight">
                  Cross-Lingual
                </h1>
                <p className="text-[11px] font-medium text-[#737373] tracking-wide uppercase truncate">
                  Research Workspace
                </p>
              </div>
            )}
          </div>
          <button
            onClick={onToggleCollapse}
            className="text-[#737373] hover:text-[#ECECEC] p-1 rounded hover:bg-[#262626] transition-colors"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.exact}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors duration-150 group relative',
                    isActive
                      ? 'bg-[#262626] text-[#ECECEC] border border-[#383838]'
                      : 'text-[#A3A3A3] hover:text-[#ECECEC] hover:bg-[#262626]/60'
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon
                      className={cn(
                        'w-4 h-4 shrink-0 transition-colors',
                        isActive ? 'text-[#8B8CF7]' : 'text-[#737373] group-hover:text-[#A3A3A3]'
                      )}
                    />
                    {!collapsed && <span className="truncate">{item.label}</span>}
                    {isActive && (
                      <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 rounded-r bg-[#8B8CF7]" />
                    )}
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Bottom Telemetry Card */}
      <div className="p-3 border-t border-[#383838]">
        {!collapsed ? (
          <div className="bg-[#212121] rounded-md p-3 border border-[#383838] space-y-2 text-xs font-mono">
            <div className="flex items-center justify-between text-[#737373]">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-[#8B8CF7]" />
                MODEL
              </span>
              <span className="text-[#ECECEC]">Gemma 3 4B</span>
            </div>
            <div className="flex items-center justify-between text-[#737373]">
              <span className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-[#8B8CF7]" />
                INFERENCES
              </span>
              <span className="text-[#ECECEC]">4,050 GPU</span>
            </div>
            <div className="flex items-center justify-between pt-1 border-t border-[#383838]/60 text-[11px]">
              <span className="text-[#737373]">STATUS</span>
              <span className={cn('flex items-center gap-1 font-semibold', backendOnline ? 'text-[#4ADE80]' : 'text-[#F87171]')}>
                <span className={cn('w-1.5 h-1.5 rounded-full', backendOnline ? 'bg-[#4ADE80]' : 'bg-[#F87171]')} />
                {backendOnline ? 'ONLINE' : 'DISCONNECTED'}
              </span>
            </div>
          </div>
        ) : (
          <div className="flex justify-center" title={backendOnline ? 'Backend Online' : 'Backend Disconnected'}>
            <span className={cn('w-2 h-2 rounded-full', backendOnline ? 'bg-[#4ADE80]' : 'bg-[#F87171]')} />
          </div>
        )}
      </div>
    </aside>
  );
};
