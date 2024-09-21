import axios from 'axios';

const client = axios.create({
  baseURL: 'https://your-backend-url/api', // Replace with your backend API URL
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to include the token in headers
client.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token'); // Adjust storage as needed
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export default client;
