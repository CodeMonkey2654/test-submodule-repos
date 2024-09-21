// src/components/AccordionItem.tsx
import React, { ReactNode } from 'react';

interface AccordionItemProps {
  children: ReactNode;
  index: number;
}

const AccordionItem: React.FC<AccordionItemProps> = ({ children}) => {
  return (
    <div className="border-b border-gray-300">
      {children}
    </div>
  );
};

export default AccordionItem;
