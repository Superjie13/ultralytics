# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import cv2
from ultralytics.engine.results import Results
from ultralytics.models import yolo_manitou
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.models.yolo_manitou.utils import invert_manitou_resize_crop_xyxy, process_mask_native, process_mask

class ManitouSegmentationPredictor_MultiCam(yolo_manitou.detect_multiCam.ManitouPredictor_MultiCam):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"