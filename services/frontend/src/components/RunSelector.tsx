import React from 'react';
import { Run } from '../types';

interface RunSelectorProps {
  runs: Run[];
  selectedRuns: string[];
  onChange: (selectedRuns: string[]) => void;
}

const RunSelector: React.FC<RunSelectorProps> = ({ runs, selectedRuns, onChange }) => {
  const handleRunSelection = (runId: string) => {
    const updatedSelection = selectedRuns.includes(runId)
      ? selectedRuns.filter(id => id !== runId)
      : [...selectedRuns, runId];
    onChange(updatedSelection);
  };

  return (
    <div className="mb-4">
      <h3 className="text-lg font-semibold mb-2">Select Runs to Compare:</h3>
      <div className="flex flex-wrap gap-2">
        {runs.map(run => (
          <button
            key={run.id}
            onClick={() => handleRunSelection(run.id)}
            className={`px-3 py-1 rounded ${
              selectedRuns.includes(run.id)
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700'
            }`}
          >
            {run.name || `Run ${run.id}`}
          </button>
        ))}
      </div>
    </div>
  );
};

export default RunSelector;
