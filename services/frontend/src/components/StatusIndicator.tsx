import React from 'react';

interface StatusIndicatorProps {
  status: 'active' | 'idle' | 'error';
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status }) => {
  const statusStyles = {
    active: 'bg-primary',
    idle: 'bg-gray-500',
    error: 'bg-black',
  };

  const statusText = {
    active: 'Active',
    idle: 'Idle',
    error: 'Error',
  };

  return (
    <div className="flex items-center">
      <span className={`h-3 w-3 rounded-full ${statusStyles[status]} mr-2`}></span>
      <span className="text-secondary">{statusText[status]}</span>
    </div>
  );
};

export default StatusIndicator;
