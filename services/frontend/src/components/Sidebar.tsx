import React from 'react';
import { FaHome, FaChartBar, FaRobot, FaCog } from 'react-icons/fa';
import Tooltip from './Tooltip';

interface SidebarItemProps {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
}

const SidebarItem: React.FC<SidebarItemProps> = ({ icon, label, active = false }) => {
  return (
    <Tooltip message={label}>
      <div
        className={`flex items-center p-3 rounded-md cursor-pointer ${
          active
            ? 'bg-primary text-white'
            : 'text-secondary hover:bg-gray-200'
        }`}
      >
        {icon}
      </div>
    </Tooltip>
  );
};

const Sidebar: React.FC = () => {
  const menuItems = [
    { icon: <FaHome size={20} />, label: 'Home' },
    { icon: <FaChartBar size={20} />, label: 'Analytics' },
    { icon: <FaRobot size={20} />, label: 'Robotics' },
    { icon: <FaCog size={20} />, label: 'Settings' },
  ];

  return (
    <div className="w-20 bg-background h-screen shadow-lg flex flex-col items-center py-4">
      <div className="mb-8">
        <FaRobot className="text-primary" size={30} />
      </div>
      <div className="flex-1 space-y-4">
        {menuItems.map((item, index) => (
          <SidebarItem key={index} icon={item.icon} label={item.label} />
        ))}
      </div>
    </div>
  );
};

export default Sidebar;
