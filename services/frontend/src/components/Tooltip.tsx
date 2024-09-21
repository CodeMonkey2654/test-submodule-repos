import React from 'react';

interface TooltipProps {
  message: string;
  children: React.ReactNode;
}

const Tooltip: React.FC<TooltipProps> = ({ message, children }) => {
  return (
    <div className="relative flex items-center group">
      {children}
      <div className="absolute bottom-full mb-2 hidden group-hover:flex justify-center">
        <span className="bg-black text-white text-xs rounded py-1 px-2">{message}</span>
        <div className="w-3 h-3 bg-black rotate-45 transform -mt-1"></div>
      </div>
    </div>
  );
};

export default Tooltip;
