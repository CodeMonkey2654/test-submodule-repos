import React, { createContext, useState, ReactNode } from 'react';

interface AccordionContextProps {
  expandedItems: number[];
  toggleItem: (index: number) => void;
  allowMultiple: boolean;
}

export const AccordionContext = createContext<AccordionContextProps | undefined>(undefined);

interface AccordionProps {
  children: ReactNode;
  allowMultiple?: boolean;
}

const Accordion: React.FC<AccordionProps> = ({ children, allowMultiple = false }) => {
  const [expandedItems, setExpandedItems] = useState<number[]>([]);

  const toggleItem = (index: number) => {
    if (expandedItems.includes(index)) {
      setExpandedItems(expandedItems.filter((item) => item !== index));
    } else {
      setExpandedItems(
        allowMultiple ? [...expandedItems, index] : [index]
      );
    }
  };

  return (
    <AccordionContext.Provider value={{ expandedItems, toggleItem, allowMultiple }}>
      <div className="w-full">
        {children}
      </div>
    </AccordionContext.Provider>
  );
};

export default Accordion;
