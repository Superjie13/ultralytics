from ultralytics.models.yolo_manitou import detect, detect_multiCam, detect_multiCam_reid, segment

from .model import YOLOManitou, YOLOManitou_MultiCam

__all__ = ["detect", "detect_multiCam", "segment", "YOLOManitou", "YOLOManitou_MultiCam"]  # allow simpler import
