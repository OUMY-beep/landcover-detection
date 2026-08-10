const API_BASE = 'http://localhost:5000/api';

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

export async function fetchCompare(filename: string, postprocess = false) {
  const response = await fetch(`${API_BASE}/compare/${filename}?postprocess=${postprocess}`);
  if (!response.ok) throw new Error('Failed to fetch comparison');
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
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || 'Failed to upload image');
  }
  return response.blob();
}

export async function uploadAndPredict(
  file: File,
  model: string,
  postprocess = false,
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', model);
  formData.append('postprocess', String(postprocess));

  const response = await fetch(`${API_BASE}/predict/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || 'Failed to run segmentation');
  }
  return response.blob();
}
