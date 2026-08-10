import { Cpu } from 'lucide-react';

type ModelType = 'unet' | 'segformer' | 'compare';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onSelect: (model: ModelType) => void;
}

export function ModelSelector({ selectedModel, onSelect }: ModelSelectorProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-5 h-5 text-gray-600" />
        <h3 className="font-semibold text-gray-800">Model Selection</h3>
      </div>

      <div className="flex flex-col gap-2">
        <button
          onClick={() => onSelect('unet')}
          className={`px-4 py-2 rounded-lg text-left transition-colors ${
            selectedModel === 'unet'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <div className="font-medium">UNet</div>
          <div className="text-xs opacity-80">U-Net architecture</div>
        </button>

        <button
          onClick={() => onSelect('segformer')}
          className={`px-4 py-2 rounded-lg text-left transition-colors ${
            selectedModel === 'segformer'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <div className="font-medium">SegFormer</div>
          <div className="text-xs opacity-80">Transformer-based</div>
        </button>

        <button
          onClick={() => onSelect('compare')}
          className={`px-4 py-2 rounded-lg text-left transition-colors ${
            selectedModel === 'compare'
              ? 'bg-purple-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <div className="font-medium">Compare</div>
          <div className="text-xs opacity-80">Side-by-side comparison</div>
        </button>
      </div>
    </div>
  );
}
