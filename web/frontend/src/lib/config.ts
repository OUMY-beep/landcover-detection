const normalizedBase = (import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:5000/api').trim();

export const API_BASE = normalizedBase.replace(/\/$/, '');

export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${normalizedPath}`;
}
