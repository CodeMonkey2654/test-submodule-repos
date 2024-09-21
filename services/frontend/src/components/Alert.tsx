import React from 'react';
import { FaCheckCircle, FaExclamationTriangle, FaInfoCircle } from 'react-icons/fa';

interface AlertProps {
  type: 'success' | 'error' | 'info';
  message: string;
}

const Alert: React.FC<AlertProps> = ({ type, message }) => {
  const types = {
    success: {
      icon: <FaCheckCircle className="text-primary" />,
      bg: 'bg-red-100',
      text: 'text-primary',
    },
    error: {
      icon: <FaExclamationTriangle className="text-secondary" />,
      bg: 'bg-red-100',
      text: 'text-secondary',
    },
    info: {
      icon: <FaInfoCircle className="text-primary" />,
      bg: 'bg-red-100',
      text: 'text-primary',
    },
  };

  return (
    <div className={`flex items-center p-4 ${types[type].bg} rounded-md`}>
      <div className="mr-3">{types[type].icon}</div>
      <div className={`${types[type].text}`}>{message}</div>
    </div>
  );
};

export default Alert;
