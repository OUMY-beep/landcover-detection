import { useState } from 'react';
import { ChevronDown, ChevronRight, List } from 'lucide-react';
import type { ClassInfo } from '../types';
import { translate, translateClassName, type Language } from '../lib/i18n';

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
  compact?: boolean;
  collapsible?: boolean;
  language: Language;
}

export function ClassLegend({ classes, compact = false, collapsible = false, language }: ClassLegendProps) {
  const [isOpen, setIsOpen] = useState(!collapsible);

  return (
    <div className="border border-slate-300 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <button
        type="button"
        onClick={() => collapsible && setIsOpen(!isOpen)}
        className={`flex w-full items-center justify-between gap-2 px-2.5 py-2 text-left ${
          collapsible ? 'cursor-pointer' : 'cursor-default'
        }`}
        aria-expanded={collapsible ? isOpen : undefined}
      >
        <span className="flex items-center gap-2">
        <List className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        <h3 className="font-semibold text-slate-800 dark:text-slate-100">{translate(language, 'landCoverClasses')}</h3>
        </span>
        {collapsible && (isOpen ? <ChevronDown className="h-4 w-4 text-slate-500" /> : <ChevronRight className="h-4 w-4 text-slate-500" />)}
      </button>

      {isOpen && <div className={`border-t border-slate-200 px-2.5 py-2 dark:border-slate-700 ${
        compact ? 'grid grid-cols-2 gap-x-2 gap-y-1.5' : 'space-y-1.5'
      }`}>
        {classes.map((cls) => (
          <div key={cls.id} className="flex min-w-0 items-center gap-1.5">
            <div
              className="h-4 w-4 shrink-0 rounded-sm border border-gray-300"
              style={{
                backgroundColor: CLASS_COLORS[cls.id] || '#ccc',
              }}
            />
            <span className="truncate text-xs leading-tight text-slate-700 dark:text-slate-200">{translateClassName(language, cls.name)}</span>
          </div>
        ))}
      </div>}
    </div>
  );
}
