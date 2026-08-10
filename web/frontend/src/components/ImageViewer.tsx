import { useState, useRef, useEffect } from 'react';
import { Layers, Eye, EyeOff } from 'lucide-react';

interface ImageViewerProps {
  imageUrl: string;
  maskUrl?: string;
  predictionUrl?: string;
  showMask?: boolean;
  showPrediction?: boolean;
  onToggleMask?: () => void;
  onTogglePrediction?: () => void;
}

export function ImageViewer({
  imageUrl,
  maskUrl,
  predictionUrl,
  showMask = true,
  showPrediction = true,
  onToggleMask,
  onTogglePrediction,
}: ImageViewerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    setIsLoading(true);
    setError(null);
  }, [imageUrl, maskUrl, predictionUrl]);

  const handleImageLoad = () => {
    setIsLoading(false);
  };

  const handleImageError = () => {
    setError('Failed to load image');
    setIsLoading(false);
  };

  if (!imageUrl) {
    return (
      <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden flex items-center justify-center">
        <p className="text-gray-400">No image loaded</p>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      {/* Base satellite image */}
      <img
        src={imageUrl}
        alt="Satellite image"
        className="absolute inset-0 w-full h-full object-contain"
        onLoad={handleImageLoad}
        onError={handleImageError}
      />

      {/* Ground truth mask overlay */}
      {maskUrl && showMask && (
        <img
          src={maskUrl}
          alt="Ground truth mask"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
      )}

      {/* Prediction mask overlay */}
      {predictionUrl && showPrediction && (
        <img
          src={predictionUrl}
          alt="Prediction mask"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-900/80">
          <p className="text-white">{error}</p>
        </div>
      )}

      {/* Layer controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 bg-black/50 p-2 rounded-lg backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-white" />
          <span className="text-white text-xs font-medium">Layers</span>
        </div>
        {maskUrl && onToggleMask && (
          <button
            onClick={onToggleMask}
            className={`flex items-center gap-2 px-2 py-1 rounded text-xs ${
              showMask ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
            }`}
          >
            {showMask ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            <span>GT Mask</span>
          </button>
        )}
        {predictionUrl && onTogglePrediction && (
          <button
            onClick={onTogglePrediction}
            className={`flex items-center gap-2 px-2 py-1 rounded text-xs ${
              showPrediction ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
            }`}
          >
            {showPrediction ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            <span>Prediction</span>
          </button>
        )}
      </div>
    </div>
  );
}
