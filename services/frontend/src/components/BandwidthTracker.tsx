import { FC, useEffect, useState } from 'react';
import { Card, LineChartComponent, Spinner } from './index';

interface BandwidthData {
  time: number;
  value: number;
}

const BandwidthTracker: FC = () => {
  const [bandwidthData, setBandwidthData] = useState<BandwidthData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // Simulate data fetching delay
    const timeout = setTimeout(() => {
      setLoading(false);
    }, 1000);

    const interval = setInterval(() => {
      setBandwidthData(prevData => {
        const newData = [...prevData, { time: Date.now(), value: Math.random() * 100 }];
        return newData.slice(-20); // Keep only the last 20 data points
      });
    }, 1000);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, []);

  if (loading) {
    return (
      <Card title="Bandwidth Tracker">
        <div className="flex justify-center items-center h-64">
          <Spinner />
        </div>
      </Card>
    );
  }

  return (
    <Card title="Bandwidth Tracker">
      <LineChartComponent
        data={bandwidthData}
        dataKey="value"
        xKey="time"
        title="Real-Time Bandwidth Usage"
        tooltipFormatter={(value: number) => `${value.toFixed(2)} Mbps`}
        labelFormatter={(time: number) => new Date(time).toLocaleTimeString()}
        strokeColor="#DC2626" // Red color as per the scheme
        strokeWidth={2}
        dot={false}
        activeDot={{ r: 8, fill: '#DC2626' }}
      />
    </Card>
  );
};

export default BandwidthTracker;
