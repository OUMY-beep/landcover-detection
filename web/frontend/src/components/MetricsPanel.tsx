import { BarChart3 } from 'lucide-react';
import type { ImageMetrics } from '../types';

interface MetricsPanelProps {
  metrics: ImageMetrics | null;
  modelName?: string;
  noGroundTruth?: boolean;
}

export function MetricsPanel({ metrics, modelName, noGroundTruth }: MetricsPanelProps) {
  if (noGroundTruth) {
    return (
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-gray-600" />
          <h3 className="font-semibold text-gray-800">
            Metrics {modelName && <span className="text-blue-600">({modelName})</span>}
          </h3>
        </div>
        <p className="text-gray-500 text-sm">
          Metrics are not available for uploaded images (no ground-truth mask).
        </p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-gray-600" />
          <h3 className="font-semibold text-gray-800">Metrics</h3>
        </div>
        <p className="text-gray-500 text-sm">No metrics available</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-gray-600" />
        <h3 className="font-semibold text-gray-800">
          Metrics {modelName && <span className="text-blue-600">({modelName})</span>}
        </h3>
      </div>

      <div className="space-y-4">
        {/* Overall metrics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded-lg p-3">
            <p className="text-xs text-blue-600 font-medium mb-1">Pixel Accuracy</p>
            <p className="text-2xl font-bold text-blue-900">
              {(metrics.pixel_accuracy * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-green-50 rounded-lg p-3">
            <p className="text-xs text-green-600 font-medium mb-1">Mean IoU</p>
            <p className="text-2xl font-bold text-green-900">
              {(metrics.mean_iou * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Per-class IoU */}
        <div>
          <p className="text-sm font-medium text-gray-700 mb-2">IoU per Class</p>
          <div className="space-y-2">
            {metrics.iou_per_class.map((item) => (
              <div key={item.class_id} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{item.class_name}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full"
                      style={{ width: `${(item.iou || 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-600 w-12 text-right">
                    {item.iou !== null ? `${(item.iou * 100).toFixed(1)}%` : 'N/A'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
