// src/components/AccordionContext.tsx
import { useContext } from 'react';
import { AccordionContext } from './Accordion';

export const useAccordionContext = () => {
  const context = useContext(AccordionContext);
  if (!context) {
    throw new Error("Accordion components must be used within an Accordion");
  }
  return context;
};
