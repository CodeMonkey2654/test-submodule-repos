import React from 'react';

interface DataTableProps {
  columns: string[];
  data: Array<{ [key: string]: any }>;
}

const DataTable: React.FC<DataTableProps> = ({ columns, data }) => {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full bg-background">
        <thead>
          <tr>
            {columns.map((col, idx) => (
              <th key={idx} className="py-2 px-4 bg-secondary text-white text-left text-sm uppercase tracking-wider">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx} className="border-b">
              {columns.map((col, cidx) => (
                <td key={cidx} className="py-2 px-4 text-sm text-secondary">
                  {row[col]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default DataTable;
