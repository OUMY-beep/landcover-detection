import { useState, useEffect, useCallback, useRef } from 'react';
import { Map } from 'lucide-react';
import { ImageViewer } from './components/ImageViewer';
import { ClassLegend } from './components/ClassLegend';
import { FileSelector } from './components/FileSelector';
import { ModelSelector } from './components/ModelSelector';
import { ControlPanel } from './components/ControlPanel';
import {
  fetchClasses,
  uploadPreview,
  uploadAndPredict,
} from './lib/api';
import type { ClassInfo } from './types';

type ModelType = 'unet' | 'segformer' | 'ensemble' | 'compare';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType>('unet');
  const [postprocess, setPostprocess] = useState(false);
  const [useTTA, setUseTTA] = useState(false);
  const [useSegEarthVerify, setUseSegEarthVerify] = useState(false);
  const [useBackgroundVerify, setUseBackgroundVerify] = useState(false);
  const [useAdvanced, setUseAdvanced] = useState(false);
  const [classes, setClasses] = useState<ClassInfo[]>([]);

  const [imageUrl, setImageUrl] = useState<string>('');
  const [predictionUrl, setPredictionUrl] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const [showPrediction, setShowPrediction] = useState(true);
  const [loading, setLoading] = useState(false);

  const blobUrlsRef = useRef<string[]>([]);

  const revokeBlobUrls = useCallback(() => {
    blobUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    blobUrlsRef.current = [];
  }, []);

  const trackBlobUrl = useCallback((url: string) => {
    blobUrlsRef.current.push(url);
    return url;
  }, []);

  useEffect(() => {
    loadClasses();
    return () => revokeBlobUrls();
  }, [revokeBlobUrls]);

  useEffect(() => {
    if (selectedFile) {
      processFile(selectedFile);
    }
  }, [selectedFile, selectedModel, postprocess, useTTA, useSegEarthVerify, useBackgroundVerify, useAdvanced]);

  const loadClasses = async () => {
    try {
      const data = await fetchClasses();
      setClasses(data.classes);
    } catch (error) {
      console.error('Failed to load classes:', error);
    }
  };

  const processFile = async (file: File) => {
    setLoading(true);
    setError(null);
    revokeBlobUrls();

    try {
      const previewBlob = await uploadPreview(file);
      setImageUrl(trackBlobUrl(URL.createObjectURL(previewBlob)));

      if (selectedModel === 'compare') {
        const [unetBlob, segformerBlob] = await Promise.all([
          uploadAndPredict(file, 'unet', postprocess, useTTA, useAdvanced, useSegEarthVerify, useBackgroundVerify),
          uploadAndPredict(file, 'segformer', postprocess, useTTA, useAdvanced, useSegEarthVerify, useBackgroundVerify),
        ]);
        setPredictionUrl(trackBlobUrl(URL.createObjectURL(unetBlob)));
        trackBlobUrl(URL.createObjectURL(segformerBlob));
      } else {
        const predBlob = await uploadAndPredict(
          file,
          selectedModel,
          postprocess,
          useTTA,
          useAdvanced,
          useSegEarthVerify,
          useBackgroundVerify,
        );
        setPredictionUrl(trackBlobUrl(URL.createObjectURL(predBlob)));
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to process image';
      setError(message);
      console.error('Failed to process file:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectFile = (file: File) => {
    setSelectedFile(file);
  };

  const handleSelectFolder = (files: File[]) => {
    // Files are already handled by the FileSelector component
    console.log(`Loaded ${files.length} images from folder`);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-3">
          <Map className="w-8 h-8 text-blue-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Land Cover Detection</h1>
            <p className="text-sm text-gray-500">Satellite image segmentation visualization</p>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          <div className="space-y-4">
            <FileSelector
              selectedFile={selectedFile}
              onSelectFile={handleSelectFile}
              onSelectFolder={handleSelectFolder}
            />
            <ModelSelector selectedModel={selectedModel} onSelect={setSelectedModel} />
            <ControlPanel
              postprocess={postprocess}
              onTogglePostprocess={() => setPostprocess(!postprocess)}
              useTTA={useTTA}
              onToggleTTA={() => setUseTTA(!useTTA)}
              useSegEarthVerify={useSegEarthVerify}
              onToggleSegEarthVerify={() => setUseSegEarthVerify(!useSegEarthVerify)}
              useBackgroundVerify={useBackgroundVerify}
              onToggleBackgroundVerify={() => setUseBackgroundVerify(!useBackgroundVerify)}
              useAdvanced={useAdvanced}
              onToggleAdvanced={() => setUseAdvanced(!useAdvanced)}
            />
          </div>

          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden">
                {loading ? (
                  <div className="w-full h-full flex items-center justify-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                  </div>
                ) : error ? (
                  <div className="w-full h-full flex items-center justify-center p-4">
                    <p className="text-red-400 text-sm text-center">{error}</p>
                  </div>
                ) : (
                  <ImageViewer
                    imageUrl={imageUrl}
                    maskUrl={undefined}
                    predictionUrl={predictionUrl}
                    showMask={false}
                    showPrediction={showPrediction}
                    onToggleMask={() => {}}
                    onTogglePrediction={() => setShowPrediction(!showPrediction)}
                  />
                )}
              </div>
            </div>

            {selectedFile && (
              <div className="mt-4 bg-white rounded-lg shadow p-4">
                <p className="text-sm text-gray-600">
                  <span className="font-medium">File:</span> {selectedFile.name}
                </p>
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Size:</span> {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>

          <div>
            <ClassLegend classes={classes} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
