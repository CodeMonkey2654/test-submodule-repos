// src/components/DPad.tsx
import React from 'react';
import { FaArrowUp, FaArrowDown, FaArrowLeft, FaArrowRight } from 'react-icons/fa';
import Button from './Button';

interface DPadProps {
  onDirectionPress: (direction: 'up' | 'down' | 'left' | 'right') => void;
  onDirectionRelease?: (direction: 'up' | 'down' | 'left' | 'right') => void;
  className?: string;
}

const DPad: React.FC<DPadProps> = ({ onDirectionPress, onDirectionRelease, className = '' }) => {
  const handleMouseDown = (direction: 'up' | 'down' | 'left' | 'right') => () => {
    onDirectionPress(direction);
  };

  const handleMouseUp = (direction: 'up' | 'down' | 'left' | 'right') => () => {
    if (onDirectionRelease) {
      onDirectionRelease(direction);
    }
  };

  const handleTouchStart = (direction: 'up' | 'down' | 'left' | 'right') => () => {
    onDirectionPress(direction);
  };

  const handleTouchEnd = (direction: 'up' | 'down' | 'left' | 'right') => () => {
    if (onDirectionRelease) {
      onDirectionRelease(direction);
    }
  };

  return (
    <div className={`flex flex-col items-center ${className}`}>
      {/* Up Button */}
      <Button
        variant="secondary"
        onMouseDown={handleMouseDown('up')}
        onMouseUp={handleMouseUp('up')}
        onTouchStart={handleTouchStart('up')}
        onTouchEnd={handleTouchEnd('up')}
        aria-label="Move Up"
        className="mb-2"
      >
        <FaArrowUp />
      </Button>
      
      <div className="flex">
        {/* Left Button */}
        <Button
          variant="secondary"
          onMouseDown={handleMouseDown('left')}
          onMouseUp={handleMouseUp('left')}
          onTouchStart={handleTouchStart('left')}
          onTouchEnd={handleTouchEnd('left')}
          aria-label="Move Left"
          className="mr-2"
        >
          <FaArrowLeft />
        </Button>
        
        {/* Right Button */}
        <Button
          variant="secondary"
          onMouseDown={handleMouseDown('right')}
          onMouseUp={handleMouseUp('right')}
          onTouchStart={handleTouchStart('right')}
          onTouchEnd={handleTouchEnd('right')}
          aria-label="Move Right"
          className="ml-2"
        >
          <FaArrowRight />
        </Button>
      </div>
      
      {/* Down Button */}
      <Button
        variant="secondary"
        onMouseDown={handleMouseDown('down')}
        onMouseUp={handleMouseUp('down')}
        onTouchStart={handleTouchStart('down')}
        onTouchEnd={handleTouchEnd('down')}
        aria-label="Move Down"
        className="mt-2"
      >
        <FaArrowDown />
      </Button>
    </div>
  );
};

export default DPad;
