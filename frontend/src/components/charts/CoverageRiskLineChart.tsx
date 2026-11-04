import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { FrontierPoint } from '../../types';

interface CoverageRiskLineChartProps {
  data: FrontierPoint[];
  height?: number;
}

export const CoverageRiskLineChart: React.FC<CoverageRiskLineChartProps> = ({
  data,
  height = 300,
}) => {
  const chartData = data.map((d) => ({
    threshold: `τ=${d.tau.toFixed(2)}`,
    'Coverage Rate': +(d.coverage * 100).toFixed(1),
    'Accuracy on Covered': +(d.accuracy * 100).toFixed(1),
    'Selective Risk': +(d.selective_risk * 100).toFixed(1),
  }));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333333" vertical={false} />
          <XAxis
            dataKey="threshold"
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
            domain={[0, 100]}
            unit="%"
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
            formatter={(value: any) => [`${value}%`]}
            labelStyle={{ color: '#ECECEC', fontWeight: 600, marginBottom: '4px' }}
          />
          <Legend
            wrapperStyle={{ fontSize: '12px', paddingTop: '12px', color: '#A3A3A3' }}
            iconType="circle"
          />
          <Line
            type="monotone"
            dataKey="Coverage Rate"
            stroke="#8B8CF7"
            strokeWidth={2}
            dot={{ r: 3, fill: '#8B8CF7' }}
          />
          <Line
            type="monotone"
            dataKey="Accuracy on Covered"
            stroke="#4ADE80"
            strokeWidth={2}
            dot={{ r: 3, fill: '#4ADE80' }}
          />
          <Line
            type="monotone"
            dataKey="Selective Risk"
            stroke="#F87171"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={{ r: 3, fill: '#F87171' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
