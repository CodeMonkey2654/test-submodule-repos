import React, { useState, useRef, useEffect } from 'react';

interface JoystickProps {
  onMove: (direction: string) => void;
}

const Joystick: React.FC<JoystickProps> = ({ onMove }) => {
  const joystickRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleMouseUp = () => {
    setDragging(false);
    setPosition({ x: 0, y: 0 });
    onMove('neutral');
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (dragging && joystickRef.current) {
      const rect = joystickRef.current.getBoundingClientRect();
      let x = e.clientX - rect.left - rect.width / 2;
      let y = e.clientY - rect.top - rect.height / 2;

      const maxDistance = rect.width / 2 - 20;
      const distance = Math.sqrt(x * x + y * y);
      if (distance > maxDistance) {
        x = (x / distance) * maxDistance;
        y = (y / distance) * maxDistance;
      }

      setPosition({ x, y });

      // Determine direction based on position
      if (y < -10) onMove('up');
      else if (y > 10) onMove('down');
      if (x < -10) onMove('left');
      else if (x > 10) onMove('right');
      if (Math.abs(x) <= 10 && Math.abs(y) <= 10) onMove('neutral');
    }
  };

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  });

  return (
    <div className="w-48 h-48 bg-gray-300 rounded-full relative flex justify-center items-center controller-joystick-container">
      <div
        ref={joystickRef}
        className="w-20 h-20 bg-secondary rounded-full absolute cursor-pointer transition-transform duration-100 controller-joystick"
        style={{ transform: `translate(${position.x}px, ${position.y}px)` }}
        onMouseDown={handleMouseDown}
      />
    </div>
  );
};

export default Joystick;
