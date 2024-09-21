// src/pages/AnalyticsPage.tsx

import React, { useState, useContext } from 'react';
import { useQuery } from '@tanstack/react-query';
import { UserContext } from '../contexts/user-context';
import client from '../client';
import { TelemetryData, Run, ComparativeData } from '../types';
import TelemetryChart from '../components/TelemetryChart';
import ComparativeTelemetryChart from '../components/ComparativeTelemetryChart';
import TimeRangeFilter from '../components/TimeRangeFilter';
import RunSelector from '../components/RunSelector';

const AnalyticsPage: React.FC = () => {
  const { user } = useContext(UserContext);
  const [timeRange, setTimeRange] = useState<string>('Last Hour');
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);

  // Fetch available runs
  const { data: runs, isLoading: runsLoading, error: runsError } = useQuery<Run[]>({
    queryKey: ['runs'], // Unique key for the query (with no parameters)
    queryFn: async () => {
      const response = await client.get('/runs'); // Replace with actual endpoint
      return response.data;
    },
  });
  // Fetch telemetry data based on time range and user role
  const { data: telemetryData, isLoading: telemetryLoading, error: telemetryError } = useQuery<TelemetryData[]>({
        queryKey: ['telemetry', timeRange, user?.role], // Unique key for the query
        queryFn: async () => {
          const params: Record<string, string> = { timeRange };
          if (user?.role === 'mechanical') {
            params.subsystem = 'mechanical';
          } else if (user?.role === 'electrical') {
            params.subsystem = 'electrical';
          } else if (user?.role === 'controls') {
            params.subsystem = 'software';
          }
          const response = await client.get('/telemetry', { params });
          return response.data;
        },
        enabled: !!user,
    });
  // Fetch comparative data for selected runs
  const { data: comparativeData, isLoading: comparativeLoading, error: comparativeError } = useQuery<ComparativeData[]>({
    queryKey: ['comparativeTelemetry', selectedRuns],
    queryFn: async () => {
      if (selectedRuns.length === 0) return [];
      const response = await client.post('/telemetry/comparative', { runIds: selectedRuns });
      return response.data;
    },
    enabled: selectedRuns.length > 0,
  });

  if (runsLoading || telemetryLoading) {
    return <div className="p-4">Loading...</div>;
  }

  if (runsError || telemetryError) {
    return <div className="p-4">Error loading data.</div>;
  }

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <header className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
        {/* You can add user info or logout button here */}
      </header>

      <main className="space-y-6">
        {/* Filters Section */}
        <section className="bg-white shadow rounded-lg p-4">
          <h2 className="text-2xl font-semibold mb-4">Filters</h2>
          <div className="flex flex-col md:flex-row md:space-x-4">
            <TimeRangeFilter selectedRange={timeRange} onChange={setTimeRange} />
            {/* Add more filters if needed */}
          </div>
        </section>

        {/* Telemetry Charts Section */}
        <section className="bg-white shadow rounded-lg p-4">
          <h2 className="text-2xl font-semibold mb-4">Telemetry Data</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <TelemetryChart
              data={telemetryData || []}
              metric="powerLevel"
              color="#8884d8"
              label="Power Level (%)"
            />
            <TelemetryChart
              data={telemetryData || []}
              metric="temperature"
              color="#82ca9d"
              label="Temperature (°C)"
            />
            {/* Add more TelemetryChart components as needed */}
          </div>
        </section>

        {/* Comparative Analysis Section */}
        <section className="bg-white shadow rounded-lg p-4">
          <h2 className="text-2xl font-semibold mb-4">Comparative Run Analysis</h2>
          <RunSelector
            runs={runs || []}
            selectedRuns={selectedRuns}
            onChange={setSelectedRuns}
          />
          {comparativeLoading && <p>Loading comparative data...</p>}
          {comparativeError && <p>Error loading comparative data.</p>}
          {comparativeData && comparativeData.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ComparativeTelemetryChart
                data={comparativeData}
                metric="powerLevel"
                label="Power Level (%)"
              />
              <ComparativeTelemetryChart
                data={comparativeData}
                metric="temperature"
                label="Temperature (°C)"
              />
              {/* Add more ComparativeTelemetryChart components as needed */}
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default AnalyticsPage;
