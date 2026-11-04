import React from 'react';
import { NavLink } from 'react-router-dom';
import { cn } from '../../utils/cn';

const NAV_ITEMS = [
  { to: '/', label: 'Overview', exact: true },
  { to: '/live-demo', label: 'Live Demo' },
  { to: '/benchmark', label: 'Benchmark' },
  { to: '/calibration', label: 'Calibration' },
  { to: '/consistency', label: 'Consistency' },
  { to: '/strategies', label: 'Strategies' },
  { to: '/reports', label: 'Reports' },
];

export const FloatingPillNav: React.FC = () => {
  return (
    <nav className="flex justify-center w-full px-4 mb-6">
      <div className="bg-[#0A0A0A] border border-[#242424] rounded-full p-1.5 shadow-2xl flex items-center gap-1 max-w-full overflow-x-auto scrollbar-none">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.exact}
            className={({ isActive }) =>
              cn(
                'text-xs font-medium px-4 py-1.5 rounded-full whitespace-nowrap transition-all duration-150',
                isActive
                  ? 'bg-[#202020] text-[#F5F5F5] shadow-inner'
                  : 'text-[#A1A1A1] hover:text-[#F5F5F5] hover:bg-[#1A1A1A]'
              )
            }
          >
            {item.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
};
