import React from 'react';
import { ResponsiveContainer } from 'recharts';
import LineChartComponent from './LineChartComponent'; // Import the new component
import { TelemetryData } from '../types';

interface TelemetryChartProps {
  data: TelemetryData[];
  metric: 'powerLevel' | 'temperature'; // Extend as needed
  color: string;
  label: string;
}

const TelemetryChart: React.FC<TelemetryChartProps> = ({ data, metric, color, label }) => {
  const chartData = data.map((d) => ({
    timestamp: new Date(d.timestamp).toLocaleTimeString(),
    value: d[metric],
  }));

  return (
    <div className="w-full h-64">
      <h3 className="text-lg font-medium mb-2">{label}</h3>
      <ResponsiveContainer>
        <LineChartComponent 
          data={chartData} 
          dataKey="value" 
          xKey="timestamp" 
          strokeColor={color} 
          tooltipFormatter={(value) => `${value}`} 
          labelFormatter={(label) => label} 
          dot={false} 
        />
      </ResponsiveContainer>
    </div>
  );
};

export default TelemetryChart;
