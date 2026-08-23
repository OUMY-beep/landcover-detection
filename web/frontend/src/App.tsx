import { useState, useEffect, useCallback, useRef } from 'react';
import { Languages, Map, Moon, PanelLeft, Sun } from 'lucide-react';
import { ImageViewer } from './components/ImageViewer';
import { ClassLegend } from './components/ClassLegend';
import { CorrectionPanel } from './components/CorrectionPanel';
import { FileSelector } from './components/FileSelector';
import { ModelSelector } from './components/ModelSelector';
import {
  fetchClasses,
  uploadPreview,
  uploadAndPredict,
  saveUploadAreaCorrection,
  saveUploadBrushCorrection,
  saveUploadCorrection,
  choosePredictionExportDestination,
  exportPrediction,
  getPredictionExportName,
  savePredictionExport,
} from './lib/api';
import type { ClassInfo } from './types';
import { translate, translateClassName, type Language, type TranslationKey } from './lib/i18n';

type ModelType = 'segformer' | 'segearth' | 'hybrid';
type CorrectionTool = 'point' | 'brush' | 'rectangle';

const MODEL_LABELS: Record<ModelType, TranslationKey> = {
  segformer: 'modelSegformer',
  segearth: 'generalImagery',
  hybrid: 'hybridQuality',
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // SegFormer is the lightest stable default. Hybrid quality remains available
  // when a second open-vocabulary model is useful.
  const [selectedModel, setSelectedModel] = useState<ModelType>('segformer');
  const [classes, setClasses] = useState<ClassInfo[]>([]);

  const [imageUrl, setImageUrl] = useState<string>('');
  const [predictionUrl, setPredictionUrl] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const [showPrediction, setShowPrediction] = useState(true);
  const [loading, setLoading] = useState(false);
  const [correctionMode, setCorrectionMode] = useState(false);
  const [correctionClassId, setCorrectionClassId] = useState(6);
  const [correctionTool, setCorrectionTool] = useState<CorrectionTool>('rectangle');
  const [savingCorrection, setSavingCorrection] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [language, setLanguage] = useState<Language>(() => {
    const savedLanguage = localStorage.getItem('landcover-language');
    if (savedLanguage === 'en' || savedLanguage === 'fr') return savedLanguage;
    return navigator.language.toLowerCase().startsWith('fr') ? 'fr' : 'en';
  });
  const [darkMode, setDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('landcover-theme');
    return savedTheme ? savedTheme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const blobUrlsRef = useRef<string[]>([]);
  const t = (key: TranslationKey, values?: Record<string, string | number>) => translate(language, key, values);

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
    document.documentElement.classList.toggle('dark', darkMode);
    localStorage.setItem('landcover-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  useEffect(() => {
    document.documentElement.lang = language;
    localStorage.setItem('landcover-language', language);
  }, [language]);

  useEffect(() => {
    if (selectedFile) {
      processFile(selectedFile);
    }
  }, [selectedFile, selectedModel]);

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

      const predBlob = await uploadAndPredict(file, selectedModel);
      setPredictionUrl(trackBlobUrl(URL.createObjectURL(predBlob)));
    } catch (error) {
      const message = error instanceof Error ? error.message : t('failedToProcess');
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

  const saveCorrection = async (save: () => Promise<unknown>) => {
    if (!selectedFile || savingCorrection) return;

    setSavingCorrection(true);
    setError(null);
    try {
      await save();
      await processFile(selectedFile);
    } catch (error) {
      const message = error instanceof Error ? error.message : t('failedToSaveCorrection');
      setError(message);
      console.error('Failed to save correction:', error);
    } finally {
      setSavingCorrection(false);
    }
  };

  const handlePredictionClick = (x: number, y: number) => {
    if (!selectedFile) return;
    return saveCorrection(() => saveUploadCorrection(selectedFile, x, y, correctionClassId));
  };

  const handlePredictionBrushSelect = (points: Array<{ x: number; y: number }>) => {
    if (!selectedFile) return;
    return saveCorrection(() => saveUploadBrushCorrection(selectedFile, points, correctionClassId));
  };

  const handlePredictionAreaSelect = (x1: number, y1: number, x2: number, y2: number) => {
    if (!selectedFile) return;
    return saveCorrection(() => saveUploadAreaCorrection(selectedFile, x1, y1, x2, y2, correctionClassId));
  };

  const handleExport = async () => {
    if (!selectedFile || isExporting) return;

    setIsExporting(true);
    setError(null);
    try {
      // The picker must run immediately from this click. Waiting for the
      // prediction request first can cause Chromium to block it as a popup.
      const suggestedName = getPredictionExportName(selectedFile.name);
      const destination = await choosePredictionExportDestination(suggestedName);
      if (destination === null) return;

      const exportedImage = await exportPrediction(selectedFile, selectedModel);
      await savePredictionExport(exportedImage, suggestedName, destination);
    } catch (error) {
      const message = error instanceof Error ? error.message : t('failedToExport');
      setError(message);
      console.error('Failed to export prediction:', error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#e9edf1] text-slate-900 transition-colors dark:bg-[#18212b] dark:text-slate-100">
      <header className="h-14 border-b border-[#1b2632] bg-[#2f3b48] text-white shadow-sm">
        <div className="mx-auto flex h-full max-w-[1440px] items-center justify-between gap-3 px-4 lg:px-6">
          <div className="flex items-center gap-3">
            <div className="rounded-md bg-[#2772b7] p-1.5 text-white">
              <Map className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-wide text-white">{t('appTitle')}</h1>
              <p className="text-[11px] text-slate-300">{t('workspace')}</p>
            </div>
          </div>
          <div className="hidden items-center gap-2 text-xs text-slate-300 md:flex">
            <PanelLeft className="h-4 w-4" />
            <span>{t('mapCanvas')}</span>
          </div>
          <button
            type="button"
            onClick={() => setLanguage(language === 'en' ? 'fr' : 'en')}
            className="inline-flex items-center gap-1.5 rounded border border-slate-500 bg-[#3c4b5c] px-2.5 py-1.5 text-xs font-medium text-white transition hover:bg-[#4a5b6e]"
            aria-label={t('switchLanguage')}
            title={t('switchLanguage')}
          >
            <Languages className="h-4 w-4" />
            <span>{language === 'en' ? 'FR' : 'EN'}</span>
          </button>
          <button
            type="button"
            onClick={() => setDarkMode(!darkMode)}
            className="inline-flex items-center gap-2 rounded border border-slate-500 bg-[#3c4b5c] px-2.5 py-1.5 text-xs font-medium text-white transition hover:bg-[#4a5b6e]"
            aria-label={darkMode ? t('useLightMode') : t('useDarkMode')}
          >
            {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            <span className="hidden sm:inline">{darkMode ? t('lightMode') : t('darkMode')}</span>
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] p-3 lg:px-4 lg:py-0">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[240px_minmax(0,1fr)]">
          <aside className="self-start space-y-3 lg:h-[calc(100vh-56px)] lg:overflow-y-auto lg:pr-1">
            <div className="space-y-3">
            <FileSelector
              selectedFile={selectedFile}
              onSelectFile={handleSelectFile}
              onSelectFolder={handleSelectFolder}
              language={language}
            />
            </div>

            <div className="space-y-3">
            <div className="hidden space-y-2 rounded-md border border-slate-300 bg-white p-3 shadow-none dark:border-slate-700 dark:bg-slate-900">
              <div>
                <h3 className="font-semibold text-slate-800 dark:text-slate-100">Manual correction</h3>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Choose a class and mark the incorrect area on the map.
                </p>
              </div>
              <select
                value={correctionClassId}
                onChange={(event) => setCorrectionClassId(Number(event.target.value))}
                disabled={!classes.length || savingCorrection}
                className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-700 disabled:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:disabled:bg-slate-800"
                aria-label="Correct class"
              >
                {classes.map((cls) => (
                  <option key={cls.id} value={cls.id}>{cls.name}</option>
                ))}
              </select>
              <div className="grid grid-cols-3 gap-1" aria-label="Correction tool">
                {(['point', 'brush', 'rectangle'] as const).map((tool) => (
                  <button
                    key={tool}
                    type="button"
                    disabled={savingCorrection}
                    onClick={() => setCorrectionTool(tool)}
                    className={`rounded px-1 py-1.5 text-xs font-medium capitalize disabled:opacity-50 ${
                      correctionTool === tool ? 'bg-slate-700 text-white dark:bg-blue-600' : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    {tool}
                  </button>
                ))}
              </div>
              <button
                type="button"
                disabled={!selectedFile || !predictionUrl || savingCorrection}
                onClick={() => setCorrectionMode(!correctionMode)}
                className={`w-full rounded px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
                  correctionMode ? 'bg-amber-500 text-white' : 'bg-blue-600 text-white'
                }`}
              >
                {savingCorrection ? 'Saving correction…' : correctionMode ? 'Finish correcting' : `Use ${correctionTool}`}
              </button>
              {correctionMode && !savingCorrection && (
                <p className="text-xs text-amber-700 dark:text-amber-300">
                  {correctionTool === 'point' && 'Click a small location to correct.'}
                  {correctionTool === 'brush' && 'Drag to paint an irregular correction area.'}
                  {correctionTool === 'rectangle' && 'Drag to select a rectangular correction area.'}
                </p>
              )}
            </div>
            <ModelSelector selectedModel={selectedModel} onSelect={setSelectedModel} language={language} />
            <CorrectionPanel
              classes={classes}
              classId={correctionClassId}
              tool={correctionTool}
              correctionMode={correctionMode}
              saving={savingCorrection}
              canCorrect={Boolean(selectedFile && predictionUrl)}
              language={language}
              onClassChange={setCorrectionClassId}
              onToolChange={setCorrectionTool}
              onToggle={() => setCorrectionMode(!correctionMode)}
            />
            <ClassLegend classes={classes} compact collapsible language={language} />
            </div>
          </aside>

          <section className="min-w-0">
            <div className="border border-slate-300 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-900">
              <div className="flex items-center justify-between gap-3 border-b border-slate-300 bg-slate-100 px-3 py-2 dark:border-slate-700 dark:bg-slate-800">
                <div>
                  <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100">{t('mapCanvas')}</h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">{t('segmentationPreview')}</p>
                </div>
                <span className="shrink-0 rounded bg-[#dcecf9] px-2 py-1 text-xs font-semibold text-[#1f5f97] dark:bg-blue-950/60 dark:text-blue-200">
                  {t(MODEL_LABELS[selectedModel])}
                </span>
              </div>
              <div className="h-64 overflow-hidden bg-gray-900 sm:h-72 lg:h-80">
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
                    correctionMode={correctionMode && !savingCorrection}
                    correctionTool={correctionTool}
                    onPredictionClick={handlePredictionClick}
                    onPredictionAreaSelect={handlePredictionAreaSelect}
                    onPredictionBrushSelect={handlePredictionBrushSelect}
                    selectedFile={selectedFile}
                    selectedModel={selectedModel}
                    onExport={handleExport}
                    isExporting={isExporting}
                    language={language}
                    fullscreenControls={(
                      <div className="flex flex-wrap items-end justify-center gap-2">
                        <label className="flex min-w-32 flex-1 flex-col gap-1 text-xs font-medium text-slate-300 sm:flex-none">
                          {t('model')}
                          <select
                            value={selectedModel}
                            onChange={(event) => setSelectedModel(event.target.value as ModelType)}
                            disabled={loading || savingCorrection}
                            className="rounded-lg border border-slate-600 bg-slate-800 px-2 py-2 text-sm text-white outline-none focus:border-blue-400 disabled:opacity-60"
                          >
                            <option value="segformer">SegFormer</option>
                            <option value="segearth">{t('generalImagery')}</option>
                            <option value="hybrid">{t('hybridQuality')}</option>
                          </select>
                        </label>
                        <label className="flex min-w-28 flex-1 flex-col gap-1 text-xs font-medium text-slate-300 sm:flex-none">
                          {t('correctClass')}
                          <select
                            value={correctionClassId}
                            onChange={(event) => setCorrectionClassId(Number(event.target.value))}
                            disabled={!classes.length || savingCorrection}
                            className="rounded-lg border border-slate-600 bg-slate-800 px-2 py-2 text-sm text-white outline-none focus:border-blue-400 disabled:opacity-60"
                          >
                            {classes.map((cls) => <option key={cls.id} value={cls.id}>{translateClassName(language, cls.name)}</option>)}
                          </select>
                        </label>
                        <div className="flex flex-col gap-1 text-xs font-medium text-slate-300">
                          {t('tool')}
                          <div className="flex rounded-lg border border-slate-600 bg-slate-800 p-1">
                            {(['point', 'brush', 'rectangle'] as const).map((tool) => (
                              <button
                                key={tool}
                                type="button"
                                disabled={savingCorrection}
                                onClick={() => setCorrectionTool(tool)}
                                className={`rounded px-2 py-1.5 text-xs font-medium capitalize transition ${
                                  correctionTool === tool ? 'bg-blue-600 text-white' : 'text-slate-300 hover:bg-slate-700'
                                }`}
                              >
                                {t(tool)}
                              </button>
                            ))}
                          </div>
                        </div>
                        <button
                          type="button"
                          disabled={savingCorrection}
                          onClick={() => setCorrectionMode(!correctionMode)}
                          className={`rounded-lg px-3 py-2 text-sm font-semibold transition disabled:opacity-60 ${
                            correctionMode ? 'bg-amber-500 text-slate-950 hover:bg-amber-400' : 'bg-blue-600 text-white hover:bg-blue-500'
                          }`}
                        >
                          {correctionMode ? t('finishCorrecting') : t('correctPrediction')}
                        </button>
                      </div>
                    )}
                  />
                )}
              </div>
              <div className="flex items-center justify-between border-t border-slate-300 bg-slate-100 px-3 py-1 text-[11px] text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400">
                <span>{t('segmentationOverlay')}</span>
                <span>{t('fullscreenHint')}</span>
              </div>
            </div>

          </section>

        </div>
      </main>
    </div>
  );
}

export default App;
