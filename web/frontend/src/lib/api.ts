import { API_BASE } from './config';

async function readErrorMessage(response: Response, fallback: string): Promise<string> {
  const contentType = response.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    const data = await response.json().catch(() => ({}));
    if (data && typeof data.error === 'string' && data.error.trim()) {
      return data.error;
    }
  }

  const bodyText = await response.text().catch(() => '');
  if (bodyText.trim()) {
    return `${fallback} (HTTP ${response.status} ${response.statusText}): ${bodyText.slice(0, 200)}`;
  }

  return `${fallback} (HTTP ${response.status} ${response.statusText})`;
}

export async function fetchImages(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/images`);
  const data = await response.json();
  return data.images;
}

export async function fetchImage(filename: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/images/${filename}`);
  if (!response.ok) throw new Error('Failed to fetch image');
  return response.blob();
}

export async function fetchMask(filename: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/masks/${filename}`);
  if (!response.ok) throw new Error('Failed to fetch mask');
  return response.blob();
}

export async function fetchPrediction(model: string, filename: string, postprocess = false): Promise<Blob> {
  const response = await fetch(`${API_BASE}/predict/${model}/${filename}?postprocess=${postprocess}`);
  if (!response.ok) throw new Error('Failed to fetch prediction');
  return response.blob();
}

export async function fetchImageMetrics(model: string, filename: string, postprocess = false) {
  const response = await fetch(`${API_BASE}/metrics/image/${model}/${filename}?postprocess=${postprocess}`);
  if (!response.ok) throw new Error('Failed to fetch metrics');
  return response.json();
}

export async function fetchClasses() {
  const response = await fetch(`${API_BASE}/classes`);
  if (!response.ok) throw new Error('Failed to fetch classes');
  return response.json();
}

export async function fetchGlobalMetrics() {
  const response = await fetch(`${API_BASE}/metrics/global`);
  if (!response.ok) throw new Error('Failed to fetch global metrics');
  return response.json();
}

export async function fetchConfusionMatrix(model: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/reports/confusion/${model}`);
  if (!response.ok) throw new Error('Failed to fetch confusion matrix');
  return response.blob();
}

export async function uploadPreview(file: File): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/upload/preview`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to upload image'));
  }
  return response.blob();
}

export async function uploadAndPredict(
  file: File,
  model: string,
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', model);

  const response = await fetch(`${API_BASE}/predict/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to run segmentation'));
  }
  return response.blob();
}

export async function saveUploadCorrection(
  file: File,
  x: number,
  y: number,
  classId: number,
  radius = 8,
) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('x', String(x));
  formData.append('y', String(y));
  formData.append('class_id', String(classId));
  formData.append('radius', String(radius));

  const response = await fetch(`${API_BASE}/corrections/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to save correction'));
  }
  return response.json();
}

export async function saveUploadAreaCorrection(
  file: File,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  classId: number,
) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('x1', String(x1));
  formData.append('y1', String(y1));
  formData.append('x2', String(x2));
  formData.append('y2', String(y2));
  formData.append('class_id', String(classId));

  const response = await fetch(`${API_BASE}/corrections/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to save correction area'));
  }
  return response.json();
}

export async function saveUploadBrushCorrection(
  file: File,
  points: Array<{ x: number; y: number }>,
  classId: number,
  radius = 12,
) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('points', JSON.stringify(points));
  formData.append('class_id', String(classId));
  formData.append('radius', String(radius));

  const response = await fetch(`${API_BASE}/corrections/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to save brush correction'));
  }
  return response.json();
}

export type PredictionExportFileHandle = {
  createWritable: () => Promise<{
    write: (data: Blob) => Promise<void>;
    close: () => Promise<void>;
  }>;
};

/** Build an editable, meaningful filename for the browser Save As dialog. */
export function getPredictionExportName(sourceFilename: string): string {
  const stem = sourceFilename.replace(/\.[^/.]+$/, '') || 'landcover';
  return `${stem}-segmentation.png`;
}

/**
 * Open the browser's native Save As dialog while the Export click is still
 * active.  `null` means that the user cancelled; `undefined` means that the
 * current browser does not expose the File System Access API.
 */
export async function choosePredictionExportDestination(
  suggestedName: string,
): Promise<PredictionExportFileHandle | null | undefined> {
  const saveFilePicker = (window as any).showSaveFilePicker as undefined | ((options: {
    suggestedName: string;
    types: Array<{
      description: string;
      accept: Record<string, string[]>;
    }>;
  }) => Promise<PredictionExportFileHandle>);

  if (typeof saveFilePicker !== 'function') return undefined;

  try {
    return await saveFilePicker({
      suggestedName,
      types: [{
        description: 'PNG image',
        accept: { 'image/png': ['.png'] },
      }],
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      return null;
    }

    // Keep export available in browsers that implement the API only partially.
    console.warn('Native Save As dialog is unavailable; using browser download instead.', error);
    return undefined;
  }
}

export async function exportPrediction(
  file: File,
  model: string,
  postprocess = true,
  use_tta = true,
  use_advanced = true,
  use_crf = false,
  confidence_threshold = 0.6,
  use_multi_scale = true,
  temperature = 0.8,
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', model);
  formData.append('postprocess', String(postprocess));
  formData.append('tta', String(use_tta));
  formData.append('advanced', String(use_advanced));
  formData.append('crf', String(use_crf));
  formData.append('confidence_threshold', String(confidence_threshold));
  formData.append('multi_scale', String(use_multi_scale));
  formData.append('temperature', String(temperature));

  const response = await fetch(`${API_BASE}/export/prediction`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to export prediction'));
  }

  return response.blob();
}

/** Save an exported PNG to the selected location, or use a download fallback. */
export async function savePredictionExport(
  blob: Blob,
  suggestedName: string,
  destination?: PredictionExportFileHandle,
): Promise<void> {
  if (destination) {
    const writable = await destination.createWritable();
    await writable.write(blob);
    await writable.close();
    return;
  }

  // Firefox and older browsers do not provide a programmable Save As dialog.
  // Their download settings determine whether their own save-location prompt
  // is shown.
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}
