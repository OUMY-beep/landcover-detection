import { ChevronLeft, ChevronRight } from 'lucide-react';
import { apiUrl } from '../lib/config';

interface ImageGalleryProps {
  images: string[];
  currentIndex: number;
  onSelect: (index: number) => void;
  onPrevious: () => void;
  onNext: () => void;
}

export function ImageGallery({
  images,
  currentIndex,
  onSelect,
  onPrevious,
  onNext,
}: ImageGalleryProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-gray-800">Image Gallery</h3>
        <span className="text-sm text-gray-500">
          {currentIndex < 0 ? 'Custom upload' : `${currentIndex + 1} / ${images.length}`}
        </span>
      </div>

      {/* Navigation buttons */}
      <div className="flex items-center justify-center gap-2 mb-4">
        <button
          onClick={onPrevious}
          disabled={currentIndex === 0}
          className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>
        <button
          onClick={onNext}
          disabled={currentIndex === images.length - 1}
          className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>

      {/* Thumbnail grid */}
      <div className="grid grid-cols-4 gap-2 max-h-48 overflow-y-auto">
        {images.map((image, index) => (
          <button
            key={image}
            onClick={() => onSelect(index)}
            className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all ${
              index === currentIndex
                ? 'border-blue-500 ring-2 ring-blue-200'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <img
              src={apiUrl(`/images/${image}`)}
              alt={image}
              className="w-full h-full object-cover"
            />
            {index === currentIndex && (
              <div className="absolute inset-0 bg-blue-500/20" />
            )}
          </button>
        ))}
      </div>

      {/* Current image name */}
      <div className="mt-3 text-center">
        <p className="text-sm text-gray-600 truncate">
          {currentIndex < 0 ? 'Viewing uploaded image' : images[currentIndex]}
        </p>
      </div>
    </div>
  );
}
