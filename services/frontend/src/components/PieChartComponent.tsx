import React from 'react';
import { PieChart, Pie, Tooltip, Cell, ResponsiveContainer } from 'recharts';

interface PieChartProps {
  data: Array<{ [key: string]: any }>;
  dataKey: string;
  nameKey: string;
  title?: string;
}

const COLORS = ['#FF0000', '#000000', '#FF4D4D', '#333333'];

const PieChartComponent: React.FC<PieChartProps> = ({ data, dataKey, nameKey, title }) => {
  return (
    <div className="bg-background shadow-md rounded-md p-4">
      {title && <h3 className="text-lg font-semibold mb-2 text-secondary">{title}</h3>}
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            dataKey={dataKey}
            nameKey={nameKey}
            cx="50%"
            cy="50%"
            outerRadius={100}
            fill="#FF0000"
            label
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]}>{entry[nameKey]}</Cell>
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PieChartComponent;
