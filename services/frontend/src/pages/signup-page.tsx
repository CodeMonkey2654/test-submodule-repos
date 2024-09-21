import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import axios from 'axios';

// Assuming New Relic is loaded globally
declare global {
  interface Window {
    newrelic?: any;
  }
}

interface SignupResponse {
  token: string;
  user: {
    id: number;
    name: string;
    email: string;
    role: 'controls' | 'electrical' | 'mechanical';
    admin: boolean;
  };
}

interface SignupData {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
  role: 'controls' | 'electrical' | 'mechanical';
  admin: boolean;
}

const signup = async (data: SignupData): Promise<SignupResponse> => {
  const response = await axios.post('/api/signup', data);
  return response.data;
};

const SignupPage: React.FC = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [role, setRole] = useState<'controls' | 'electrical' | 'mechanical'>('controls');
  const [errorMessage, setErrorMessage] = useState('');

  const admin = role === 'controls';

  const mutation = useMutation({
    mutationKey: ['signup'],
    mutationFn: signup,
    onSuccess: (data) => {
      // Handle successful signup, e.g., save token, redirect
      console.log('Signup successful:', data);
    },
    onError: (error: any) => {
      // Log error to New Relic
      if (window.newrelic) {
        window.newrelic.noticeError(error);
      }
      console.error('Signup failed:', error);
      setErrorMessage('Signup failed. Please try again.');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage('');

    if (password !== confirmPassword) {
      setErrorMessage('Passwords do not match.');
      return;
    }

    mutation.mutate({ name, email, password, confirmPassword, role, admin });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-secondary">
      <div className="bg-accent p-8 rounded shadow-md w-full max-w-md">
        <h2 className="text-2xl font-bold mb-6 text-secondary">Sign Up</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="name" className="block text-secondary mb-1">
              Name
            </label>
            <input
              type="text"
              id="name"
              className="w-full px-3 py-2 border border-secondary rounded focus:outline-none focus:ring-2 focus:ring-primary"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              placeholder="Your Full Name"
            />
          </div>
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
          <div>
            <label htmlFor="confirmPassword" className="block text-secondary mb-1">
              Confirm Password
            </label>
            <input
              type="password"
              id="confirmPassword"
              className="w-full px-3 py-2 border border-secondary rounded focus:outline-none focus:ring-2 focus:ring-primary"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              placeholder="********"
            />
          <div>
            <label htmlFor="team" className="block text-secondary mb-1">
              Team
            </label>
            <select
              id="team"
              className="w-full px-3 py-2 border border-secondary rounded focus:outline-none focus:ring-2 focus:ring-primary"
              value={role}
              onChange={(e) => setRole(e.target.value as "controls" | "electrical" | "mechanical")}
              required
            >
              <option value="">Select an option</option>
              <option value="controls">Controls</option>
              <option value="electrical">Electrical</option>
              <option value="mechanical">Mechanical</option>
            </select>
          </div>
          </div>
          {errorMessage && (
            <div className="text-red-500 text-sm">
              {errorMessage}
            </div>
          )}
          <button
            type="submit"
            className={`w-full py-2 px-4 bg-primary text-accent rounded hover:bg-red-600 transition-colors ${
              mutation.isPending ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? 'Signing up...' : 'Sign Up'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default SignupPage;