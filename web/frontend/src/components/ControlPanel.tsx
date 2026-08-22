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
}: ControlPanelProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center gap-2 mb-4">
        <SlidersHorizontal className="w-5 h-5 text-gray-600" />
        <h3 className="font-semibold text-gray-800">Controls</h3>
      </div>

      <div className="space-y-3">
        <label className="flex items-center justify-between cursor-pointer">
          <div>
            <p className="text-sm font-medium text-gray-700">Test Time Augmentation</p>
            <p className="text-xs text-gray-500">Improve accuracy with multiple transforms</p>
          </div>
          <button
            onClick={onToggleTTA}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              useTTA ? 'bg-green-500' : 'bg-gray-300'
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
            <p className="text-sm font-medium text-gray-700">SegEarth Verification</p>
            <p className="text-xs text-gray-500">Use SegEarth-OV to check obvious misses</p>
          </div>
          <button
            onClick={onToggleSegEarthVerify}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              useSegEarthVerify ? 'bg-sky-500' : 'bg-gray-300'
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
            <p className="text-sm font-medium text-gray-700">Background Verify</p>
            <p className="text-xs text-gray-500">Only revisit pixels predicted as background</p>
          </div>
          <button
            onClick={onToggleBackgroundVerify}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              useBackgroundVerify ? 'bg-cyan-500' : 'bg-gray-300'
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
            <p className="text-sm font-medium text-gray-700">Advanced Refinement</p>
            <p className="text-xs text-gray-500">Conservative cleanup of clear isolated errors</p>
          </div>
          <button
            onClick={onToggleAdvanced}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              useAdvanced ? 'bg-purple-500' : 'bg-gray-300'
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
            <p className="text-sm font-medium text-gray-700">Morphological</p>
            <p className="text-xs text-gray-500">Apply morphological operations</p>
          </div>
          <button
            onClick={onTogglePostprocess}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              postprocess ? 'bg-blue-500' : 'bg-gray-300'
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
