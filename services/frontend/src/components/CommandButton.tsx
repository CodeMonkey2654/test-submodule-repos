import React from 'react';
import { FaPlay, FaStop, FaPause } from 'react-icons/fa';

interface CommandButtonProps {
  command: 'start' | 'stop' | 'pause';
  onClick: () => void;
}

const CommandButton: React.FC<CommandButtonProps> = ({ command, onClick }) => {
  const getIcon = () => {
    switch (command) {
      case 'start':
        return <FaPlay />;
      case 'stop':
        return <FaStop />;
      case 'pause':
        return <FaPause />;
      default:
        return null;
    }
  };

  const getLabel = () => {
    switch (command) {
      case 'start':
        return 'Start';
      case 'stop':
        return 'Stop';
      case 'pause':
        return 'Pause';
      default:
        return '';
    }
  };

  const colors = {
    start: 'bg-primary',
    stop: 'bg-secondary',
    pause: 'bg-gray-500',
  };

  return (
    <button
      onClick={onClick}
      className={`flex items-center px-4 py-2 ${colors[command]} text-white rounded-md hover:opacity-80 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-${command === 'start' ? 'primary' : 'secondary'}`}
    >
      {getIcon()}
      <span className="ml-2">{getLabel()}</span>
    </button>
  );
};

export default CommandButton;
