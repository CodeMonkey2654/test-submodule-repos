import React, { useEffect, useState, useContext } from 'react';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import {jwtDecode} from 'jwt-decode';
import { UserContext } from '../contexts/user-context';
import { Link } from '@tanstack/react-router';

interface TelemetryData {
  timestamp: string;
  powerLevel: number;
  temperature: number;
  // Add other telemetry fields as needed
}

interface JWTToken {
  role: 'Mechanical Engineer' | 'Electrical Engineer' | 'Computer Science Major';
  // Add other JWT fields as needed
}

const MainDashboard: React.FC = () => {
  const { user, token } = useContext(UserContext);
  const [telemetry, setTelemetry] = useState<TelemetryData[]>([]);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  // Decode JWT to get user role
  const decodedToken: JWTToken | null = token ? jwtDecode<JWTToken>(token) : null;
  const userRole = decodedToken?.role || 'Guest';
  console.log(userRole);
  console.log(socket);

  // Establish WebSocket connection
  useEffect(() => {
    const ws = new WebSocket('wss://your-backend-url/ws/telemetry'); // Replace with your WebSocket URL

    ws.onopen = () => {
      console.log('WebSocket connection established');
    };

    ws.onmessage = (event) => {
      const data: TelemetryData = JSON.parse(event.data);
      setTelemetry((prev) => [...prev, data].slice(-50)); // Keep last 50 data points
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setSocket(ws);

    // Cleanup on unmount
    return () => {
      ws.close();
    };
  }, []);

  // Example: Fetch overall rover status (replace with actual API call)
  const { data: roverStatus, isLoading, error } = useQuery({
      queryKey: ['roverStatus'],
      queryFn: async () => {
        const response = await fetch('/api/rover/status', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      }
  });

  // Prepare data for Recharts
  const chartData = telemetry.map(data => ({
    timestamp: new Date(data.timestamp).toLocaleTimeString(),
    powerLevel: data.powerLevel,
    temperature: data.temperature,
  }));

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <header className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Lunar Rover Dashboard</h1>
        <div>
          <span className="mr-4">Welcome, {user?.name || 'User'}</span>
          <Link
            to="/login"
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            Logout
          </Link>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Analytics Card */}
        <div className="bg-white shadow rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-4">Analytics</h2>
          {isLoading ? (
            <p>Loading rover status...</p>
          ) : error ? (
            <p>Error loading data.</p>
          ) : (
            <div>
              <p><strong>Location:</strong> {roverStatus.location.x}, {roverStatus.location.y}</p>
              <p><strong>Power Level:</strong> {roverStatus.powerLevel}%</p>
              <p><strong>Temperature:</strong> {roverStatus.temperature}°C</p>
              {/* Add more status fields as needed */}
            </div>
          )}
          <Link
            to="/analytics"
            className="mt-4 inline-block text-blue-500 hover:underline"
          >
            View Detailed Analytics
          </Link>
        </div>

        {/* Controls Card */}
        <div className="bg-white shadow rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-4">Controls</h2>
          <p>Use the controls to maneuver the rover.</p>
          <Link
            to="/controller"
            className="mt-4 inline-block text-blue-500 hover:underline"
          >
            Go to Controller
          </Link>
        </div>

        {/* Help Card */}
        <div className="bg-white shadow rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-4">Help & Support</h2>
          <p>Access documentation, tutorials, and support resources.</p>
          <Link
            to="/help"
            className="mt-4 inline-block text-blue-500 hover:underline"
          >
            Get Help
          </Link>
        </div>

        {/* Real-Time Telemetry Chart */}
        <div className="col-span-1 lg:col-span-3 bg-white shadow rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-4">Real-Time Telemetry</h2>
          <LineChart width={500} height={300} data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="powerLevel" stroke="#8884d8" activeDot={{ r: 8 }} />
            <Line type="monotone" dataKey="temperature" stroke="#82ca9d" />
          </LineChart>
        </div>

        {/* Additional Links (Conditional based on role) */}
      </main>
    </div>
  );
};

export default MainDashboard;
