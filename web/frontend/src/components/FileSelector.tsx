import { useRef, useState } from 'react';
import { FolderOpen, Upload, FileImage } from 'lucide-react';
import { translate, type Language } from '../lib/i18n';

interface FileSelectorProps {
  selectedFile: File | null;
  onSelectFile: (file: File) => void;
  onSelectFolder?: (files: File[]) => void;
  language: Language;
}

export function FileSelector({ selectedFile, onSelectFile, onSelectFolder, language }: FileSelectorProps) {
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
          alert(translate(language, 'noImagesFound'));
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
    <div className="border border-slate-300 bg-white p-2 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-1.5 flex items-center gap-2">
        <FolderOpen className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        <h3 className="font-semibold text-slate-800 dark:text-slate-100">{translate(language, 'imageSource')}</h3>
      </div>

      <div className="space-y-1">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="w-full flex items-center justify-center gap-2 rounded border-2 border-dashed border-slate-300 px-2 py-1.5 transition-colors hover:border-blue-400 hover:bg-blue-50 dark:border-slate-700 dark:hover:border-blue-500 dark:hover:bg-blue-950/40"
        >
          <Upload className="w-5 h-5 text-slate-500 dark:text-slate-400" />
          <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{translate(language, 'selectImageFile')}</span>
        </button>

        <button
          onClick={handleFolderSelect}
          className="w-full flex items-center justify-center gap-2 rounded border-2 border-dashed border-slate-300 px-2 py-1.5 transition-colors hover:border-green-400 hover:bg-green-50 dark:border-slate-700 dark:hover:border-green-500 dark:hover:bg-green-950/40"
        >
          <FolderOpen className="w-5 h-5 text-slate-500 dark:text-slate-400" />
          <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{translate(language, 'selectFolder')}</span>
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
          <div className="rounded-lg border border-green-200 bg-green-50 p-3 dark:border-green-900 dark:bg-green-950/50">
            <p className="text-sm font-medium text-green-900 dark:text-green-200">
              {translate(language, 'imagesLoaded', { count: selectedFiles.length })}
            </p>
            <p className="text-xs text-green-700 mt-1 dark:text-green-300">
              {translate(language, 'selectImageFromFolder')}
            </p>
          </div>
        )}

        {selectedFiles && selectedFiles.length > 1 && (
          <div className="max-h-36 overflow-y-auto rounded-lg border border-slate-200 dark:border-slate-700">
            {selectedFiles.map((file, index) => (
              <button
                key={index}
                onClick={() => onSelectFile(file)}
                className={`w-full flex items-center gap-2 px-3 py-2 text-left text-sm transition-colors border-b border-slate-100 last:border-b-0 dark:border-slate-800 ${
                  selectedFile === file
                    ? 'bg-blue-50 text-blue-700 font-medium'
                    : 'text-slate-700 hover:bg-slate-50 dark:text-slate-200 dark:hover:bg-slate-800'
                }`}
              >
                <FileImage className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">{file.name}</span>
              </button>
            ))}
          </div>
        )}

        {selectedFile && !selectedFiles && (
          <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 dark:border-blue-900 dark:bg-blue-950/50">
            <p className="text-sm font-medium text-blue-900 truncate dark:text-blue-200" title={selectedFile.name}>
              {selectedFile.name}
            </p>
            <p className="text-xs text-blue-700 mt-1 dark:text-blue-300">
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        )}

        <p className="text-xs text-slate-500 text-center dark:text-slate-400">
          {translate(language, 'supportedFormats')}
        </p>
      </div>
    </div>
  );
}
