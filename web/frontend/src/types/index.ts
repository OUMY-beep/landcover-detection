export interface ImageMetrics {
  pixel_accuracy: number;
  mean_iou: number;
  iou_per_class: Array<{
    class_id: number;
    class_name: string;
    iou: number | null;
  }>;
}

export interface ModelResult {
  model: string;
  filename: string;
  pixel_accuracy: number;
  mean_iou: number;
  iou_per_class: Array<{
    class_id: number;
    class_name: string;
    iou: number | null;
  }>;
}

export interface CompareResponse {
  filename: string;
  results: {
    unet: ModelResult;
    segformer: ModelResult;
  };
}

export interface ClassInfo {
  id: number;
  name: string;
}

export interface ImagesResponse {
  images: string[];
  total: number;
}
