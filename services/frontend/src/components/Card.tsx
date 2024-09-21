import React from 'react';

interface CardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

const Card: React.FC<CardProps> = ({ title, children, className }) => {
  return (
    <div className={`bg-background shadow-md rounded-md p-4 ${className ? className : ""}`}>
      <h3 className="text-lg font-semibold mb-2 text-secondary">{title}</h3>
      {children}
    </div>
  );
};

export default Card;
