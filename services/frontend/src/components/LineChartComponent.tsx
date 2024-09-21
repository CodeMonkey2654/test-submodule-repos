// src/components/LineChartComponent.tsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface LineChartProps {
  data: Array<{ [key: string]: any }>;
  dataKey: string;
  xKey: string;
  title?: string;
  tooltipFormatter?: (value: any) => string;
  labelFormatter?: (label: any) => string;
  strokeColor?: string;
  strokeWidth?: number;
  dot?: boolean;
  activeDot?: boolean | object;
}

const LineChartComponent: React.FC<LineChartProps> = ({
  data,
  dataKey,
  xKey,
  title,
  tooltipFormatter,
  labelFormatter,
  strokeColor = '#FF0000', // Default to red
  strokeWidth = 2,
  dot = true,
  activeDot = { r: 5 },
}) => {
  return (
    <div className="w-full h-64">
      {title && <h3 className="text-lg font-semibold mb-2 text-secondary">{title}</h3>}
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <XAxis
            dataKey={xKey}
            tickFormatter={labelFormatter}
            stroke="#000000"
            tick={{ fill: '#333333' }}
          />
          <YAxis stroke="#000000" tick={{ fill: '#333333' }} />
          <Tooltip
            labelFormatter={labelFormatter}
            formatter={tooltipFormatter ? tooltipFormatter : undefined}
            contentStyle={{ backgroundColor: '#FFFFFF', border: '1px solid #000000', color: '#333333' }}
            labelStyle={{ color: '#000000' }}
          />
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            dot={dot}
            activeDot={activeDot}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LineChartComponent;
