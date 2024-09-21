import React from 'react';

interface ListProps {
  items: React.ReactNode[];
  type?: 'disc' | 'circle' | 'decimal';
  className?: string;
}

const List: React.FC<ListProps> = ({ items, type = 'disc', className = '' }) => {
  const listStyle = {
    disc: 'list-disc',
    circle: 'list-circle',
    decimal: 'list-decimal',
  };

  return (
    <ul className={`${listStyle[type]} list-inside space-y-2 ${className}`}>
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
};

export default List;
