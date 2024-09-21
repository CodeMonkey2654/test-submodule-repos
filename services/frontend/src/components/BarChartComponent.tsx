import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface BarChartProps {
  data: Array<{ [key: string]: any }>;
  dataKey: string;
  xKey: string;
  title?: string;
}

const BarChartComponent: React.FC<BarChartProps> = ({ data, dataKey, xKey, title }) => {
  return (
    <div className="bg-background shadow-md rounded-md p-4">
      {title && <h3 className="text-lg font-semibold mb-2 text-secondary">{title}</h3>}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <XAxis dataKey={xKey} stroke="#000" />
          <YAxis stroke="#000" />
          <Tooltip />
          <Bar dataKey={dataKey} fill="#FF0000" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BarChartComponent;
