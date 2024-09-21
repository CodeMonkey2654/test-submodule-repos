// src/components/Button.tsx
import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
  active?: boolean;
  onMouseDown?: () => void;
  onMouseUp?: () => void;
  onTouchStart?: () => void;
  onTouchEnd?: () => void;
  ariaLabel?: string;
}

const Button: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  disabled = false,
  type = 'button',
  className = '',
  active = false,
  onMouseDown,
  onMouseUp,
  onTouchStart,
  onTouchEnd,
  ariaLabel,
}) => {
  const baseClasses =
    'px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 flex items-center justify-center transition-colors duration-200';
  const variants = {
    primary: 'bg-primary text-white hover:bg-accent focus:ring-primary',
    secondary: 'bg-secondary text-white hover:bg-gray-700 focus:ring-secondary',
  };
  const activeClasses = active ? 'opacity-75' : '';
  return (
    <button
      type={type}
      onClick={onClick}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onTouchStart={onTouchStart}
      onTouchEnd={onTouchEnd}
      className={`${baseClasses} ${variants[variant]} ${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      } ${activeClasses} ${className}`}
      disabled={disabled}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
};

export default Button;
