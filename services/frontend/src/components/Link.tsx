import React from 'react';

interface LinkProps {
  href: string;
  children: React.ReactNode;
  target?: string;
  rel?: string;
  className?: string;
}

const Link: React.FC<LinkProps> = ({
  href,
  children,
  target = '_blank',
  rel = 'noopener noreferrer',
  className = '',
}) => {
  return (
    <a
      href={href}
      target={target}
      rel={rel}
      className={`text-primary hover:underline ${className}`}
    >
      {children}
    </a>
  );
};

export default Link;
