import { SlidersHorizontal } from 'lucide-react';

interface ControlPanelProps {
  postprocess: boolean;
  onTogglePostprocess: () => void;
  useTTA: boolean;
  onToggleTTA: () => void;
  useSegEarthVerify: boolean;
  onToggleSegEarthVerify: () => void;
  useBackgroundVerify: boolean;
  onToggleBackgroundVerify: () => void;
  useAdvanced: boolean;
  onToggleAdvanced: () => void;
  disabled?: boolean;
}

export function ControlPanel({
  postprocess,
  onTogglePostprocess,
  useTTA,
  onToggleTTA,
  useSegEarthVerify,
  onToggleSegEarthVerify,
  useBackgroundVerify,
  onToggleBackgroundVerify,
  useAdvanced,
  onToggleAdvanced,
  disabled = false,
}: ControlPanelProps) {
  return (
    <div className={`rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900 ${
      disabled ? 'opacity-60' : ''
    }`}>
      <div className="flex items-center gap-2 mb-4">
        <SlidersHorizontal className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        <h3 className="font-semibold text-slate-800 dark:text-slate-100">Refinement controls</h3>
      </div>

      {!disabled && (
        <p className="mb-3 rounded-lg bg-blue-50 px-3 py-2 text-xs text-blue-700 dark:bg-blue-950/50 dark:text-blue-200">
          Standard SegFormer refinements are enabled by default.
        </p>
      )}

      {disabled && (
        <p className="mb-3 rounded-lg bg-slate-100 px-3 py-2 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          Available with SegFormer only.
        </p>
      )}

      <div className="space-y-3">
        <label className="flex items-center justify-between cursor-pointer">
          <div>
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Test Time Augmentation</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Improve accuracy with multiple transforms</p>
          </div>
          <button
            disabled={disabled}
            onClick={onToggleTTA}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              useTTA ? 'bg-green-500' : 'bg-slate-300 dark:bg-slate-700'
            }`}
          >
            <div
              className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                useTTA ? 'left-7' : 'left-1'
              }`}
            />
          </button>
        </label>

        <label className="flex items-center justify-between cursor-pointer">
          <div>
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">SegEarth Verification</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Use SegEarth-OV to check obvious misses</p>
          </div>
          <button
            disabled={disabled}
            onClick={onToggleSegEarthVerify}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              useSegEarthVerify ? 'bg-sky-500' : 'bg-slate-300 dark:bg-slate-700'
            }`}
          >
            <div
              className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                useSegEarthVerify ? 'left-7' : 'left-1'
              }`}
            />
          </button>
        </label>

        <label className="flex items-center justify-between cursor-pointer">
          <div>
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Background Verify</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Only revisit pixels predicted as background</p>
          </div>
          <button
            disabled={disabled}
            onClick={onToggleBackgroundVerify}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              useBackgroundVerify ? 'bg-cyan-500' : 'bg-slate-300 dark:bg-slate-700'
            }`}
          >
            <div
              className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                useBackgroundVerify ? 'left-7' : 'left-1'
              }`}
            />
          </button>
        </label>

        <label className="flex items-center justify-between cursor-pointer">
          <div>
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Advanced Refinement</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Conservative cleanup of clear isolated errors</p>
          </div>
          <button
            disabled={disabled}
            onClick={onToggleAdvanced}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              useAdvanced ? 'bg-purple-500' : 'bg-slate-300 dark:bg-slate-700'
            }`}
          >
            <div
              className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                useAdvanced ? 'left-7' : 'left-1'
              }`}
            />
          </button>
        </label>

        <label className="flex items-center justify-between cursor-pointer">
          <div>
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Morphological</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Apply morphological operations</p>
          </div>
          <button
            disabled={disabled}
            onClick={onTogglePostprocess}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              postprocess ? 'bg-blue-500' : 'bg-slate-300 dark:bg-slate-700'
            }`}
          >
            <div
              className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                postprocess ? 'left-7' : 'left-1'
              }`}
            />
          </button>
        </label>
      </div>
    </div>
  );
}
