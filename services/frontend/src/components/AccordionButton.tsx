import React from 'react';
import { useAccordionContext } from './AccordionContext';
import { FaChevronDown, FaChevronUp } from 'react-icons/fa';

interface AccordionButtonProps {
  children: React.ReactNode;
  index: number;
}

const AccordionButton: React.FC<AccordionButtonProps> = ({ children, index }) => {
  const context = useAccordionContext();

  if (!context) {
    throw new Error("AccordionButton must be used within an Accordion");
  }

  const { expandedItems, toggleItem } = context;
  const isExpanded = expandedItems.includes(index);

  const handleClick = () => {
    toggleItem(index);
  };

  return (
    <button
      onClick={handleClick}
      className="w-full flex justify-between items-center p-4 text-left text-primary focus:outline-none focus:ring-2 focus:ring-primary"
      aria-expanded={isExpanded}
      aria-controls={`accordion-panel-${index}`}
    >
      <span className="font-medium">{children}</span>
      {isExpanded ? <FaChevronUp /> : <FaChevronDown />}
    </button>
  );
};

export default AccordionButton;
