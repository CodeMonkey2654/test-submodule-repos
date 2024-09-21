import React, { useState } from 'react';

interface Tab {
  label: string;
  content: React.ReactNode;
}

interface TabsProps {
  tabs: Tab[];
}

const Tabs: React.FC<TabsProps> = ({ tabs }) => {
  const [active, setActive] = useState(0);

  return (
    <div>
      <div className="flex border-b border-secondary">
        {tabs.map((tab, idx) => (
          <button
            key={idx}
            onClick={() => setActive(idx)}
            className={`py-2 px-4 -mb-px text-sm font-medium ${
              active === idx
                ? 'border-b-2 border-primary text-primary'
                : 'text-secondary hover:text-primary'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="p-4">
        {tabs[active].content}
      </div>
    </div>
  );
};

export default Tabs;
