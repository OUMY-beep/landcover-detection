import { Cpu } from 'lucide-react';
import { translate, type Language } from '../lib/i18n';

type ModelType = 'segformer' | 'segearth' | 'hybrid';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onSelect: (model: ModelType) => void;
  language: Language;
}

export function ModelSelector({ selectedModel, onSelect, language }: ModelSelectorProps) {
  const t = (key: Parameters<typeof translate>[1]) => translate(language, key);

  return (
    <div className="border border-slate-300 bg-white p-2 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-2 flex items-center gap-2">
        <Cpu className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        <h3 className="font-semibold text-slate-800 dark:text-slate-100">{t('segmentationModel')}</h3>
      </div>

      <div className="flex flex-col gap-1">
        <button
          onClick={() => onSelect('segformer')}
          className={`rounded px-2.5 py-1 text-left leading-tight transition-colors ${
            selectedModel === 'segformer'
              ? 'bg-blue-500 text-white'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
          }`}
        >
          <div className="font-medium">SegFormer</div>
          <div className="text-xs opacity-80">{t('transformerBased')}</div>
        </button>

        <button
          onClick={() => onSelect('segearth')}
          className={`rounded px-2.5 py-1 text-left leading-tight transition-colors ${
            selectedModel === 'segearth'
              ? 'bg-teal-500 text-white'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
          }`}
        >
          <div className="font-medium">{t('generalImagery')}</div>
          <div className="text-xs opacity-80">{t('generalImageryDescription')}</div>
        </button>

        <button
          onClick={() => onSelect('hybrid')}
          className={`rounded px-2.5 py-1 text-left leading-tight transition-colors ${
            selectedModel === 'hybrid'
              ? 'bg-indigo-500 text-white'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
          }`}
        >
          <div className="font-medium">{t('hybridQuality')}</div>
          <div className="text-xs opacity-80">{t('hybridDescription')}</div>
        </button>

      </div>
    </div>
  );
}
