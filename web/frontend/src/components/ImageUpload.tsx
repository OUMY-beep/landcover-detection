import { useRef, useState } from 'react';
import { Upload, X } from 'lucide-react';

interface ImageUploadProps {
  onUpload: (file: File) => void;
  onClear: () => void;
  uploadedFilename: string | null;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/tiff', 'image/webp'];
const ACCEPTED_EXTENSIONS = '.png,.jpg,.jpeg,.tif,.tiff,.webp';

export function ImageUpload({ onUpload, onClear, uploadedFilename, disabled }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateAndUpload = (file: File | undefined) => {
    if (!file) return;

    setError(null);

    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    const validExt = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'webp'].includes(ext);
    const validType = ACCEPTED_TYPES.includes(file.type) || validExt;

    if (!validType) {
      setError('Please upload a PNG, JPEG, TIFF, or WebP image.');
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setError('File must be smaller than 20 MB.');
      return;
    }

    onUpload(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    validateAndUpload(e.target.files?.[0]);
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    validateAndUpload(e.dataTransfer.files[0]);
  };

  if (uploadedFilename) {
    return (
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center gap-2 mb-3">
          <Upload className="w-5 h-5 text-gray-600" />
          <h3 className="font-semibold text-gray-800">Your Image</h3>
        </div>
        <div className="flex items-center justify-between gap-2 bg-blue-50 rounded-lg p-3">
          <p className="text-sm text-blue-900 truncate" title={uploadedFilename}>
            {uploadedFilename}
          </p>
          <button
            onClick={onClear}
            disabled={disabled}
            className="shrink-0 p-1 rounded hover:bg-blue-100 text-blue-700 disabled:opacity-50"
            title="Back to gallery"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Segmentation applied. Select a gallery image or upload another file.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center gap-2 mb-3">
        <Upload className="w-5 h-5 text-gray-600" />
        <h3 className="font-semibold text-gray-800">Upload Image</h3>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
        <p className="text-sm font-medium text-gray-700">
          Drop an image here or click to browse
        </p>
        <p className="text-xs text-gray-500 mt-1">PNG, JPEG, TIFF, WebP — max 20 MB</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          onChange={handleFileChange}
          className="hidden"
          disabled={disabled}
        />
      </div>

      {error && <p className="text-xs text-red-600 mt-2">{error}</p>}
    </div>
  );
}
