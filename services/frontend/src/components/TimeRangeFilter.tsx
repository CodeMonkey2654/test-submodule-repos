import React from 'react';

interface TimeRangeFilterProps {
  selectedRange: string;
  onChange: (range: string) => void;
}

const TimeRangeFilter: React.FC<TimeRangeFilterProps> = ({ selectedRange, onChange }) => {
  const options = ['Last Hour', 'Last Day', 'Last Week', 'All Time'];

  return (
    <div className="mb-4">
      <label htmlFor="timeRange" className="block text-sm font-medium text-gray-700 mb-1">
        Select Time Range:
      </label>
      <select
        id="timeRange"
        value={selectedRange}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
};

export default TimeRangeFilter;
