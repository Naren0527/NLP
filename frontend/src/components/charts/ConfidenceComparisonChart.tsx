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

interface ConfidenceComparisonChartProps {
  data: BenchmarkLanguageRow[];
  height?: number;
}

export const ConfidenceComparisonChart: React.FC<ConfidenceComparisonChartProps> = ({
  data,
  height = 300,
}) => {
  const chartData = data.map((d) => ({
    language: `${d.language} (${d.code})`,
    'Verbalized Confidence': +(d.verbalized_conf * 100).toFixed(1),
    'Statistical Confidence': +(d.statistical_conf * 100).toFixed(1),
    'Actual Accuracy': +(d.baseline_accuracy * 100).toFixed(1),
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
          <Bar
            dataKey="Verbalized Confidence"
            fill="#A78BFA"
            radius={[3, 3, 0, 0]}
            maxBarSize={30}
          />
          <Bar
            dataKey="Statistical Confidence"
            fill="#8B8CF7"
            radius={[3, 3, 0, 0]}
            maxBarSize={30}
          />
          <Bar
            dataKey="Actual Accuracy"
            fill="#F87171"
            radius={[3, 3, 0, 0]}
            maxBarSize={30}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
