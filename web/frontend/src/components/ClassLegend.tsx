import { List } from 'lucide-react';
import type { ClassInfo } from '../types';

const CLASS_COLORS = [
  'rgba(31, 119, 180, 0.7)',
  'rgba(255, 127, 14, 0.7)',
  'rgba(44, 160, 44, 0.7)',
  'rgba(214, 39, 40, 0.7)',
  'rgba(148, 103, 189, 0.7)',
  'rgba(140, 86, 75, 0.7)',
  'rgba(227, 119, 194, 0.7)',
  'rgba(127, 127, 127, 0.7)',
];

interface ClassLegendProps {
  classes: ClassInfo[];
}

export function ClassLegend({ classes }: ClassLegendProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center gap-2 mb-4">
        <List className="w-5 h-5 text-gray-600" />
        <h3 className="font-semibold text-gray-800">Classes</h3>
      </div>

      <div className="space-y-2">
        {classes.map((cls) => (
          <div key={cls.id} className="flex items-center gap-3">
            <div
              className="w-6 h-6 rounded border border-gray-300"
              style={{
                backgroundColor: CLASS_COLORS[cls.id] || '#ccc',
              }}
            />
            <span className="text-sm text-gray-700">{cls.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
