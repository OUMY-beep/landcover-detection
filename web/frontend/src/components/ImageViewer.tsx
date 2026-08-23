import { useState, useRef, useEffect, type MouseEvent, type ReactNode } from 'react';
import { Eye, EyeOff, Layers, Maximize2, Minimize2, Download } from 'lucide-react';
import { translate, type Language } from '../lib/i18n';

interface ImageViewerProps {
  imageUrl: string;
  maskUrl?: string;
  predictionUrl?: string;
  showMask?: boolean;
  showPrediction?: boolean;
  onToggleMask?: () => void;
  onTogglePrediction?: () => void;
  correctionMode?: boolean;
  correctionTool?: 'point' | 'brush' | 'rectangle';
  onPredictionClick?: (x: number, y: number) => void;
  onPredictionAreaSelect?: (x1: number, y1: number, x2: number, y2: number) => void;
  onPredictionBrushSelect?: (points: Array<{ x: number; y: number }>) => void;
  fullscreenControls?: ReactNode;
  selectedFile?: File | null;
  selectedModel?: string;
  onExport?: () => void;
  isExporting?: boolean;
  language: Language;
}

interface SelectionPoint {
  x: number;
  y: number;
  displayX: number;
  displayY: number;
}

export function ImageViewer({
  imageUrl,
  maskUrl,
  predictionUrl,
  showMask = true,
  showPrediction = true,
  onToggleMask,
  onTogglePrediction,
  correctionMode = false,
  correctionTool = 'rectangle',
  onPredictionClick,
  onPredictionAreaSelect,
  onPredictionBrushSelect,
  fullscreenControls,
  selectedFile,
  selectedModel,
  onExport,
  isExporting = false,
  language,
}: ImageViewerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const predictionRef = useRef<HTMLImageElement>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const [selectionStart, setSelectionStart] = useState<SelectionPoint | null>(null);
  const [selectionEnd, setSelectionEnd] = useState<SelectionPoint | null>(null);
  const [brushPoints, setBrushPoints] = useState<SelectionPoint[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const t = (key: Parameters<typeof translate>[1]) => translate(language, key);

  useEffect(() => {
    setIsLoading(true);
    setError(null);
  }, [imageUrl, maskUrl, predictionUrl]);

  useEffect(() => {
    const updateFullscreenState = () => setIsFullscreen(document.fullscreenElement === viewerRef.current);
    document.addEventListener('fullscreenchange', updateFullscreenState);
    return () => document.removeEventListener('fullscreenchange', updateFullscreenState);
  }, []);

  const handleImageLoad = () => {
    setIsLoading(false);
  };

  const handleImageError = () => {
    setError(t('failedToLoadImage'));
    setIsLoading(false);
  };

  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await viewerRef.current?.requestFullscreen();
      }
    } catch (fullscreenError) {
      console.error('Fullscreen mode is unavailable:', fullscreenError);
    }
  };

  const getPredictionPoint = (event: MouseEvent<HTMLDivElement>): SelectionPoint | null => {
    const image = predictionRef.current;
    if (!image || !image.naturalWidth || !image.naturalHeight) return null;

    const bounds = event.currentTarget.getBoundingClientRect();
    const imageRatio = image.naturalWidth / image.naturalHeight;
    const containerRatio = bounds.width / bounds.height;
    const displayedWidth = containerRatio > imageRatio ? bounds.height * imageRatio : bounds.width;
    const displayedHeight = containerRatio > imageRatio ? bounds.height : bounds.width / imageRatio;
    const left = (bounds.width - displayedWidth) / 2;
    const top = (bounds.height - displayedHeight) / 2;
    const x = event.clientX - bounds.left - left;
    const y = event.clientY - bounds.top - top;

    if (x < 0 || y < 0 || x > displayedWidth || y > displayedHeight) return null;
    return {
      x: x / displayedWidth,
      y: y / displayedHeight,
      displayX: event.clientX - bounds.left,
      displayY: event.clientY - bounds.top,
    };
  };

  const handleSelectionStart = (event: MouseEvent<HTMLDivElement>) => {
    const point = getPredictionPoint(event);
    if (!point) return;
    if (correctionTool === 'point') {
      onPredictionClick?.(point.x, point.y);
      return;
    }
    if (correctionTool === 'brush') {
      setBrushPoints([point]);
      return;
    }
    setSelectionStart(point);
    setSelectionEnd(point);
  };

  const handleSelectionMove = (event: MouseEvent<HTMLDivElement>) => {
    if (correctionTool === 'brush' && brushPoints.length) {
      const point = getPredictionPoint(event);
      const lastPoint = brushPoints[brushPoints.length - 1];
      if (point && Math.hypot(point.x - lastPoint.x, point.y - lastPoint.y) >= 0.003) {
        setBrushPoints([...brushPoints, point]);
      }
      return;
    }
    if (!selectionStart) return;
    const point = getPredictionPoint(event);
    if (point) setSelectionEnd(point);
  };

  const handleSelectionEnd = (event: MouseEvent<HTMLDivElement>) => {
    const point = getPredictionPoint(event);
    if (correctionTool === 'brush') {
      if (point && brushPoints.length && onPredictionBrushSelect) {
        const lastPoint = brushPoints[brushPoints.length - 1];
        const points = Math.hypot(point.x - lastPoint.x, point.y - lastPoint.y) >= 0.001
          ? [...brushPoints, point]
          : brushPoints;
        onPredictionBrushSelect(points.map(({ x, y }) => ({ x, y })));
      }
      setBrushPoints([]);
      return;
    }
    if (!selectionStart || !point || !onPredictionAreaSelect) {
      setSelectionStart(null);
      setSelectionEnd(null);
      return;
    }
    setSelectionStart(null);
    setSelectionEnd(null);
    onPredictionAreaSelect(selectionStart.x, selectionStart.y, point.x, point.y);
  };

  const selectionLeft = selectionStart && selectionEnd
    ? Math.min(selectionStart.displayX, selectionEnd.displayX) : 0;
  const selectionTop = selectionStart && selectionEnd
    ? Math.min(selectionStart.displayY, selectionEnd.displayY) : 0;
  const selectionWidth = selectionStart && selectionEnd
    ? Math.abs(selectionStart.displayX - selectionEnd.displayX) : 0;
  const selectionHeight = selectionStart && selectionEnd
    ? Math.abs(selectionStart.displayY - selectionEnd.displayY) : 0;

  if (!imageUrl) {
    return (
      <div className="relative flex h-full w-full items-center justify-center overflow-hidden bg-gray-900">
        <p className="text-gray-400">{t('noImageLoaded')}</p>
      </div>
    );
  }

  return (
    <div ref={viewerRef} className="viewer-fullscreen relative h-full w-full overflow-hidden bg-gray-950">
      {/* Base satellite image */}
      <img
        src={imageUrl}
        alt={t('satelliteImage')}
        className="absolute inset-0 w-full h-full object-contain"
        onLoad={handleImageLoad}
        onError={handleImageError}
      />

      {/* Ground truth mask overlay */}
      {maskUrl && showMask && (
        <img
          src={maskUrl}
          alt={t('groundTruthMask')}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
      )}

      {/* Prediction mask overlay */}
      {predictionUrl && showPrediction && (
        <img
          ref={predictionRef}
          src={predictionUrl}
          alt={t('predictionMask')}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
      )}

      {predictionUrl && showPrediction && correctionMode && (
        <div
          role="presentation"
          aria-label={t('applyCorrection')}
          className="absolute inset-0 z-10 cursor-crosshair select-none"
          onMouseDown={handleSelectionStart}
          onMouseMove={handleSelectionMove}
          onMouseUp={handleSelectionEnd}
          onContextMenu={(event) => event.preventDefault()}
        >
          {selectionStart && selectionEnd && (
            <div
              className="pointer-events-none absolute border-2 border-amber-400 bg-amber-400/20"
              style={{ left: selectionLeft, top: selectionTop, width: selectionWidth, height: selectionHeight }}
            />
          )}
          {correctionTool === 'brush' && brushPoints.map((point, index) => (
            <div
              key={`${point.x}-${point.y}-${index}`}
              className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-amber-400 bg-amber-400/30"
              style={{ left: point.displayX, top: point.displayY }}
            />
          ))}
        </div>
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

      {isFullscreen && fullscreenControls && (
        <div className="absolute bottom-4 left-1/2 z-30 w-[calc(100%-2rem)] max-w-4xl -translate-x-1/2 rounded-xl border border-white/15 bg-slate-950/85 p-3 text-white shadow-2xl backdrop-blur-md">
          {fullscreenControls}
        </div>
      )}

      {/* Layer controls */}
      <div className="absolute top-4 right-4 z-20 flex flex-col gap-2 rounded-lg bg-black/50 p-2 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-white" />
          <span className="text-white text-xs font-medium">{t('layers')}</span>
        </div>
        <button
          type="button"
          onClick={toggleFullscreen}
          className="flex items-center gap-2 rounded px-2 py-1 text-xs text-white transition hover:bg-white/15"
          aria-label={isFullscreen ? t('exitFullscreen') : t('viewFullscreen')}
        >
          {isFullscreen ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
          <span>{isFullscreen ? t('exitFullscreen') : t('fullscreen')}</span>
        </button>
        {maskUrl && onToggleMask && (
          <button
            onClick={onToggleMask}
            className={`flex items-center gap-2 px-2 py-1 rounded text-xs ${
              showMask ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
            }`}
          >
            {showMask ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            <span>{t('groundTruthMask')}</span>
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
            <span>{t('prediction')}</span>
          </button>
        )}
        {predictionUrl && selectedFile && onExport && (
          <button
            onClick={onExport}
            disabled={isExporting}
            className="flex items-center gap-2 px-2 py-1 rounded text-xs bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            title={t('exportPredictionMask')}
          >
            {isExporting ? (
              <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" />
            ) : (
              <Download className="w-3 h-3" />
            )}
            <span>{t('export')}</span>
          </button>
        )}
      </div>
    </div>
  );
}
