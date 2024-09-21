import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from 'recharts';

interface LineConfig {
  dataKey: string;
  strokeColor: string;
  strokeWidth?: number;
  dot?: boolean;
  activeDot?: boolean | object;
}

interface MultiLineChartProps {
  data: Array<{ [key: string]: any }>;
  xKey: string;
  lines: LineConfig[];
  title?: string;
  tooltipFormatter?: (value: any, name: string, props: any) => React.ReactNode;
  labelFormatter?: (label: any) => string;
  showCartesianGrid?: boolean;
  showLegend?: boolean;
  legendLayout?: 'horizontal' | 'vertical';
  legendVerticalAlign?: 'top' | 'middle' | 'bottom';
  legendAlign?: 'left' | 'center' | 'right';
}

const MultiLineChartComponent: React.FC<MultiLineChartProps> = ({
  data,
  xKey,
  lines,
  title,
  tooltipFormatter,
  labelFormatter,
  showCartesianGrid = false,
  showLegend = false,
  legendLayout = 'horizontal',
  legendVerticalAlign = 'top',
  legendAlign = 'right',
}) => {
  return (
    <div className="w-full h-64">
      {title && <h3 className="text-lg font-semibold mb-2 text-secondary">{title}</h3>}
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          {showCartesianGrid && <CartesianGrid strokeDasharray="3 3" />}
          <XAxis
            dataKey={xKey}
            tickFormatter={labelFormatter}
            stroke="#000000"
            tick={{ fill: '#333333' }}
          />
          <YAxis stroke="#000000" tick={{ fill: '#333333' }} />
          <Tooltip
            labelFormatter={labelFormatter}
            formatter={tooltipFormatter}
            contentStyle={{ backgroundColor: '#FFFFFF', border: '1px solid #000000', color: '#333333' }}
            labelStyle={{ color: '#000000' }}
          />
          {showLegend && (
            <Legend layout={legendLayout} verticalAlign={legendVerticalAlign} align={legendAlign} />
          )}
          {lines.map((line) => (
            <Line
              key={line.dataKey}
              type="monotone"
              dataKey={line.dataKey}
              stroke={line.strokeColor}
              strokeWidth={line.strokeWidth || 2}
              dot={line.dot !== undefined ? line.dot : true}
              activeDot={line.activeDot || { r: 5 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MultiLineChartComponent;
