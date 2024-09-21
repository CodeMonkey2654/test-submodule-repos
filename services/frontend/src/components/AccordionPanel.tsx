// src/components/AccordionPanel.tsx
import React from 'react';
import { useAccordionContext } from './AccordionContext';

interface AccordionPanelProps {
  children: React.ReactNode;
  index: number;
}

const AccordionPanel: React.FC<AccordionPanelProps> = ({ children, index }) => {
  const context = useAccordionContext();

  if (!context) {
    throw new Error("AccordionPanel must be used within an Accordion");
  }

  const { expandedItems } = context;
  const isExpanded = expandedItems.includes(index);

  return (
    <div
      id={`accordion-panel-${index}`}
      className={`px-4 pb-4 transition-max-height duration-300 overflow-hidden ${isExpanded ? 'max-h-screen' : 'max-h-0'}`}
      aria-hidden={!isExpanded}
    >
      {isExpanded && <div className="text-gray-700">{children}</div>}
    </div>
  );
};

export default AccordionPanel;
