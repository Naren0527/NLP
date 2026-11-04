import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { BenchmarkLanguageRow } from '../../types';

interface CalibrationBarChartProps {
  data: BenchmarkLanguageRow[];
  height?: number;
}

export const CalibrationBarChart: React.FC<CalibrationBarChartProps> = ({
  data,
  height = 300,
}) => {
  const chartData = data.map((d) => ({
    language: `${d.language} (${d.code})`,
    'Baseline ECE': +d.baseline_ece.toFixed(4),
    'Mitigated ECE': +d.mitigated_ece.toFixed(4),
  }));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333333" vertical={false} />
          <XAxis
            dataKey="language"
            stroke="#737373"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#383838' }}
          />
          <YAxis
            stroke="#737373"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#383838' }}
            domain={[0, 0.8]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#262626',
              borderColor: '#383838',
              borderRadius: '6px',
              color: '#ECECEC',
              fontSize: '12px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)',
            }}
            formatter={(value: any) => [value, 'Expected Calibration Error']}
            labelStyle={{ color: '#ECECEC', fontWeight: 600, marginBottom: '4px' }}
          />
          <Legend
            wrapperStyle={{ fontSize: '12px', paddingTop: '12px', color: '#A3A3A3' }}
            iconType="circle"
          />
          <Bar
            dataKey="Baseline ECE"
            fill="#F87171"
            radius={[3, 3, 0, 0]}
            maxBarSize={40}
          />
          <Bar
            dataKey="Mitigated ECE"
            fill="#4ADE80"
            radius={[3, 3, 0, 0]}
            maxBarSize={40}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
