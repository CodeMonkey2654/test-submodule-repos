import React from 'react';

interface TextareaProps {
  label: string;
  id: string;
  name: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  required?: boolean;
  rows?: number;
}

const Textarea: React.FC<TextareaProps> = ({
  label,
  id,
  name,
  value,
  onChange,
  required = false,
  rows = 4,
}) => {
  return (
    <div>
      <Textarea
        label={label}
        id={id}
        name={name}
        required={required}
        rows={rows}
        value={value}
        onChange={onChange}
      ></Textarea>
    </div>
  );
};

export default Textarea;
