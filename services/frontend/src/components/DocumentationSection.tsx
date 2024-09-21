// src/components/DocumentationSection.tsx
import React from 'react';
import { Card, Link, List } from './index';

interface DocumentationSectionProps {
  documentationLinks: { title: string; url: string }[];
}

const DocumentationSection: React.FC<DocumentationSectionProps> = ({ documentationLinks }) => {
  const listItems = documentationLinks.map((doc, index) => (
    <Link key={index} href={doc.url}>
      {doc.title}
    </Link>
  ));

  return (
    <Card title="Documentation">
      <List items={listItems} type="disc" />
    </Card>
  );
};

export default DocumentationSection;
