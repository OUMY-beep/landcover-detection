import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { ClassInfo } from '../types';
import { translate, translateClassName, type Language } from '../lib/i18n';

type CorrectionTool = 'point' | 'brush' | 'rectangle';

interface CorrectionPanelProps {
  classes: ClassInfo[];
  classId: number;
  tool: CorrectionTool;
  correctionMode: boolean;
  saving: boolean;
  canCorrect: boolean;
  language: Language;
  onClassChange: (classId: number) => void;
  onToolChange: (tool: CorrectionTool) => void;
  onToggle: () => void;
}

export function CorrectionPanel({
  classes,
  classId,
  tool,
  correctionMode,
  saving,
  canCorrect,
  language,
  onClassChange,
  onToolChange,
  onToggle,
}: CorrectionPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const t = (key: Parameters<typeof translate>[1], values?: Record<string, string | number>) => translate(language, key, values);

  return (
    <div className="border border-slate-300 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between gap-2 px-2.5 py-2 text-left"
        aria-expanded={isOpen}
      >
        <span>
          <span className="block text-sm font-semibold text-slate-800 dark:text-slate-100">{t('manualCorrection')}</span>
          {!isOpen && <span className="block text-xs text-slate-500 dark:text-slate-400">{t('correctionSummary')}</span>}
        </span>
        {isOpen ? <ChevronDown className="h-4 w-4 text-slate-500" /> : <ChevronRight className="h-4 w-4 text-slate-500" />}
      </button>

      {isOpen && (
        <div className="space-y-2 border-t border-slate-200 px-2.5 py-2 dark:border-slate-700">
          <p className="text-xs text-slate-500 dark:text-slate-400">{t('correctionInstructions')}</p>
          <select
            value={classId}
            onChange={(event) => onClassChange(Number(event.target.value))}
            disabled={!classes.length || saving}
            className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-700 disabled:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:disabled:bg-slate-800"
            aria-label={t('correctClass')}
          >
            {classes.map((cls) => <option key={cls.id} value={cls.id}>{translateClassName(language, cls.name)}</option>)}
          </select>

          <div className="grid grid-cols-3 gap-1" aria-label={t('correctionTool')}>
            {(['point', 'brush', 'rectangle'] as const).map((item) => (
              <button
                key={item}
                type="button"
                disabled={saving}
                onClick={() => onToolChange(item)}
                className={`rounded px-1 py-1.5 text-xs font-medium disabled:opacity-50 ${
                  tool === item ? 'bg-slate-700 text-white dark:bg-blue-600' : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
                }`}
              >
                {t(item)}
              </button>
            ))}
          </div>

          <button
            type="button"
            disabled={!canCorrect || saving}
            onClick={onToggle}
            className={`w-full rounded px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
              correctionMode ? 'bg-amber-500 text-white' : 'bg-blue-600 text-white'
            }`}
          >
            {saving ? t('savingCorrection') : correctionMode ? t('finishCorrecting') : t('useTool', { tool: t(tool) })}
          </button>

          {correctionMode && !saving && (
            <p className="text-xs text-amber-700 dark:text-amber-300">
              {tool === 'point' && t('clickToCorrect')}
              {tool === 'brush' && t('dragToPaint')}
              {tool === 'rectangle' && t('dragToSelect')}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
