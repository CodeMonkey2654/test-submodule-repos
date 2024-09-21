import React from 'react';
import { FaRobot } from 'react-icons/fa';
import Button from './Button';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-secondary text-background px-4 py-3 flex justify-between items-center shadow-md">
      <div className="flex items-center">
        <FaRobot className="text-primary mr-2" size={24} />
        <span className="font-bold text-xl">Robotics Dashboard</span>
      </div>
      <div>
        <Button variant="secondary">Logout</Button>
      </div>
    </nav>
  );
};

export default Navbar;
