import { useRef, useState } from 'react';
import { FolderOpen, Upload, FileImage } from 'lucide-react';

interface FileSelectorProps {
  selectedFile: File | null;
  onSelectFile: (file: File) => void;
  onSelectFolder?: (files: File[]) => void;
}

export function FileSelector({ selectedFile, onSelectFile, onSelectFolder }: FileSelectorProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[] | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFiles(null);
      onSelectFile(file);
    }
  };

  const handleFolderSelect = async () => {
    try {
      // Try to use the File System Access API for folder selection
      if ('showDirectoryPicker' in window) {
        const dirHandle = await (window as any).showDirectoryPicker();
        const files: File[] = [];
        
        for await (const entry of dirHandle.values()) {
          if (entry.kind === 'file') {
            const file = await entry.getFile();
            if (file.type.startsWith('image/')) {
              files.push(file);
            }
          }
        }
        
        if (files.length > 0) {
          setSelectedFiles(files);
          onSelectFolder?.(files);
          // Select the first file by default
          onSelectFile(files[0]);
        } else {
          alert('No image files found in the selected folder.');
        }
      } else {
        // Fallback to multiple file input
        folderInputRef.current?.click();
      }
    } catch (err) {
      // User cancelled or error
      console.log('Folder selection cancelled');
    }
  };

  const handleFolderInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const fileArray = Array.from(files);
      setSelectedFiles(fileArray);
      onSelectFolder?.(fileArray);
      onSelectFile(fileArray[0]);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center gap-2 mb-4">
        <FolderOpen className="w-5 h-5 text-gray-600" />
        <h3 className="font-semibold text-gray-800">Select Image</h3>
      </div>

      <div className="space-y-3">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors"
        >
          <Upload className="w-5 h-5 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">Select Image File</span>
        </button>

        <button
          onClick={handleFolderSelect}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg hover:border-green-400 hover:bg-green-50 transition-colors"
        >
          <FolderOpen className="w-5 h-5 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">Select Folder</span>
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        <input
          ref={folderInputRef}
          type="file"
          accept="image/*"
          multiple
          webkitdirectory=""
          onChange={handleFolderInput}
          className="hidden"
        />

        {selectedFiles && selectedFiles.length > 1 && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <p className="text-sm font-medium text-green-900">
              {selectedFiles.length} images loaded
            </p>
            <p className="text-xs text-green-700 mt-1">
              Click on individual files in the list below to switch between them
            </p>
          </div>
        )}

        {selectedFiles && selectedFiles.length > 1 && (
          <div className="max-h-48 overflow-y-auto border border-gray-200 rounded-lg">
            {selectedFiles.map((file, index) => (
              <button
                key={index}
                onClick={() => onSelectFile(file)}
                className={`w-full flex items-center gap-2 px-3 py-2 text-left text-sm transition-colors border-b border-gray-100 last:border-b-0 ${
                  selectedFile === file
                    ? 'bg-blue-50 text-blue-700 font-medium'
                    : 'hover:bg-gray-50 text-gray-700'
                }`}
              >
                <FileImage className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">{file.name}</span>
              </button>
            ))}
          </div>
        )}

        {selectedFile && !selectedFiles && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-sm font-medium text-blue-900 truncate" title={selectedFile.name}>
              {selectedFile.name}
            </p>
            <p className="text-xs text-blue-700 mt-1">
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        )}

        <p className="text-xs text-gray-500 text-center">
          Supported: PNG, JPEG, TIFF, WebP
        </p>
      </div>
    </div>
  );
}
