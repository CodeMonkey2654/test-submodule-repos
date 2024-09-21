import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import axios from 'axios';

// Assuming New Relic is loaded globally
declare global {
  interface Window {
    newrelic?: any;
  }
}

interface LoginResponse {
  token: string;
  user: {
    id: number;
    name: string;
    email: string;
  };
}

const login = async (credentials: { email: string; password: string }): Promise<LoginResponse> => {
  const response = await axios.post('/api/login', credentials);
  return response.data;
};

const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const mutation = useMutation({
    mutationKey: ['login'],
    mutationFn: login,
    onSuccess: (data) => {
      console.log('Login successful:', data);
    },
    onError: (error: any) => {
      if (window.newrelic) {
        window.newrelic.noticeError(error);
      }
      console.error('Login failed:', error);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate({ email, password });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-secondary">
      <div className="bg-accent p-8 rounded shadow-md w-full max-w-md">
        <h2 className="text-2xl font-bold mb-6 text-secondary">Login</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-secondary mb-1">
              Email
            </label>
            <input
              type="email"
              id="email"
              className="w-full px-3 py-2 border border-secondary rounded focus:outline-none focus:ring-2 focus:ring-primary"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-secondary mb-1">
              Password
            </label>
            <input
              type="password"
              id="password"
              className="w-full px-3 py-2 border border-secondary rounded focus:outline-none focus:ring-2 focus:ring-primary"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="********"
            />
          </div>
          {mutation.isError && (
            <div className="text-red-500 text-sm">
              An error occurred during login. Please try again.
            </div>
          )}
          <button
            type="submit"
            className={`w-full py-2 px-4 bg-primary text-accent rounded hover:bg-red-600 transition-colors ${
              mutation.isPending ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;