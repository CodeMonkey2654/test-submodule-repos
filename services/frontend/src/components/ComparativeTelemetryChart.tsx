// src/components/ComparativeTelemetryChart.tsx
import React from 'react';
import { Card, MultiLineChartComponent } from './index';
import { ComparativeData } from '../types';

interface ComparativeTelemetryChartProps {
  data: ComparativeData[];
  metric: 'powerLevel' | 'temperature'; // Extend as needed
  label: string;
}

const ComparativeTelemetryChart: React.FC<ComparativeTelemetryChartProps> = ({ data, metric, label }) => {
  // Combine data from multiple runs based on timestamps
  const combinedData: { timestamp: string; [key: string]: number | string }[] = [];

  data.forEach((run) => {
    run.telemetry.forEach((d) => {
      const timestamp = new Date(d.timestamp).toLocaleTimeString();
      const existing = combinedData.find((item) => item.timestamp === timestamp);
      if (existing) {
        existing[run.runId] = d[metric];
      } else {
        combinedData.push({
          timestamp,
          [run.runId]: d[metric],
        });
      }
    });
  });

  // Prepare line configurations
  const lines = data.map((run, index) => ({
    dataKey: run.runId,
    strokeColor: getColor(index),
    strokeWidth: 2,
    dot: false,
    activeDot: { r: 5, fill: getColor(index) },
  }));

  return (
    <Card title={`${label} Comparison`}>
      <MultiLineChartComponent
        data={combinedData}
        xKey="timestamp"
        lines={lines}
        title={`${label} Comparison`}
        showCartesianGrid={true}
        showLegend={true}
        legendLayout="horizontal"
        legendVerticalAlign="top"
        legendAlign="right"
      />
    </Card>
  );
};

// Helper function to get color based on index, aligning with red, white, black scheme
const getColor = (index: number): string => {
  const colors = ['#FF0000', '#000000', '#FF4D4D', '#333333', '#FF7300', '#0088FE']; // Adjusted colors
  return colors[index % colors.length];
};

export default ComparativeTelemetryChart;
