import { SlidersHorizontal } from 'lucide-react';

interface ControlPanelProps {
  postprocess: boolean;
  onTogglePostprocess: () => void;
}

export function ControlPanel({ postprocess, onTogglePostprocess }: ControlPanelProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center gap-2 mb-4">
        <SlidersHorizontal className="w-5 h-5 text-gray-600" />
        <h3 className="font-semibold text-gray-800">Controls</h3>
      </div>

      <div className="space-y-3">
        <label className="flex items-center justify-between cursor-pointer">
          <div>
            <p className="text-sm font-medium text-gray-700">Post-processing</p>
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
