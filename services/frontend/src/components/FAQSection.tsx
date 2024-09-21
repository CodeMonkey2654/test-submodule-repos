// src/components/FAQSection.tsx
import React from 'react';
import { Accordion, AccordionItem, AccordionButton, AccordionPanel, Card } from './index';

interface FAQSectionProps {
  faqItems: { question: string; answer: string }[];
  allowMultiple?: boolean;
}

const FAQSection: React.FC<FAQSectionProps> = ({ faqItems, allowMultiple = false }) => {
  return (
    <Card title="Frequently Asked Questions">
      <Accordion allowMultiple={allowMultiple}>
        {faqItems.map((item, index) => (
          <AccordionItem key={index} index={index}>
            <AccordionButton index={index}>{item.question}</AccordionButton>
            <AccordionPanel index={index}>{item.answer}</AccordionPanel>
          </AccordionItem>
        ))}
      </Accordion>
    </Card>
  );
};

export default FAQSection;
