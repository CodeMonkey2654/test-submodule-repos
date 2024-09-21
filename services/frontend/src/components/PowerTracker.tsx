import { FC, useEffect, useState } from 'react';
import { Card, Spinner } from './index'; // Importing from your component library
import LineChartComponent from './LineChartComponent'; // Import the new component

interface PowerDataPoint {
  time: number;
  value: number;
}

const PowerTracker: FC = () => {
  const [powerConsumptionData, setPowerConsumptionData] = useState<PowerDataPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // Simulate initial data loading
    const loadData = setTimeout(() => {
      setLoading(false);
    }, 1000);

    // Update data every second
    const interval = setInterval(() => {
      setPowerConsumptionData(prevData => {
        const newData = [...prevData, { time: Date.now(), value: Math.random() * 100 }];
        return newData.slice(-20); // Keep only the last 20 data points
      });
    }, 1000);

    return () => {
      clearInterval(interval);
      clearTimeout(loadData);
    };
  }, []);

  if (loading) {
    return (
      <Card title="Power Consumption Tracker" className="flex justify-center items-center w-full h-64">
        <Spinner />
      </Card>
    );
  }

  return (
    <Card title="Power Consumption Tracker" className="w-full h-64">
      <LineChartComponent 
        data={powerConsumptionData} 
        dataKey="value" 
        xKey="time" 
        tooltipFormatter={(value) => `${value.toFixed(2)} W`} 
        labelFormatter={(label) => new Date(label).toLocaleTimeString()} 
        strokeColor="#DC2626" 
        activeDot={{ r: 8, fill: '#DC2626' }} 
        dot={false} 
      />
    </Card>
  );
}

export default PowerTracker;