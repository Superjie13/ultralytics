# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import cv2
import torch
from ultralytics.engine.results import Results
from ultralytics.models import yolo_manitou
from ultralytics.utils import DEFAULT_CFG, ops, LOGGER
from ultralytics.models.yolo_manitou.utils import invert_manitou_resize_crop_xyxy, process_mask_native, process_mask
from ultralytics.utils.ops import crop_mask
from ultralytics.data.manitou_loaders import LoadManitouImagesAndRadar

class ManitouSegmentationPredictor_MultiCam(yolo_manitou.detect_multiCam_reid.ManitouPredictor_MultiCam_ReID):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, _batch):
        """
        Apply non-max suppression and process segmentation detections for each image in the input batch.

        Args:
            preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
            img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
            orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

        Returns:
            (list): List of Results objects containing the segmentation predictions for each image in the batch.
                   Each Results object includes both bounding boxes and segmentation masks.

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo11n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        # Extract protos - tuple if PyTorch model or array if exported
        save_feats = getattr(self, "save_feats", True)
        orig_imgs = [
            _batch[i]["orig_images"][f"cam{j}"] for i in range(len(_batch)) for j in range(1, 5)
        ]  # get original images from batch
        paths = [
            self.batch[0][i][f"cam{j}"] for i in range(len(self.batch[0])) for j in range(1, 5)
        ]  # get paths from batch
        img = self.preprocess(_batch)

        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        preds = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            #return_idxs=save_feats,
        )

        feat_preds = []
        for b in range(len(preds)):
            if preds[b].shape[0] == 0:
                feat_preds.append(torch.empty((0, self._feats[b].shape[0]), device=self._feats.device))  # empty case
                continue
            masks = torch.einsum('in,nhw->ihw', preds[b][:, 6:].to(protos[b].dtype), protos[b]).sigmoid()
            masks = crop_mask(masks, preds[b][:, :4] / self.args.mask_ratio).unsqueeze(1)
            emb = (masks * self._feats[b]).sum(dim=(2, 3)) / (masks.sum(dim=(2, 3)) + 1e-6)
            feat_preds.append(emb)
        self._feats = feat_preds

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = [
                self._feats[j] for j in range(len(self._feats))
            ]           

        results = self.construct_results(preds, img, orig_imgs, protos)
        
        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f

        results = {"cameras": results, "radars": _batch[0]["radar"]}  # add radar data to results

        return results

    def construct_results(self, preds, img, orig_imgs, protos):
        """
        Construct a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.
            protos (List[torch.Tensor]): List of prototype masks.

        Returns:
            (List[Results]): List of result objects containing the original images, image paths, class names,
                bounding boxes, and masks.
        """
        res_list = []
        cams = ["cam1", "cam2", "cam3", "cam4"]
        for cam, pred, orig_img, proto in zip(cams, preds, orig_imgs, protos):
            img_path = self.batch[0][0][cam]
            res_list.append(self.construct_result(pred, img, orig_img, img_path, proto))
        return res_list

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """
        Construct a single result object from the prediction.

        Args:
            pred (np.ndarray): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        assert tuple(orig_img.shape[:2]) == tuple(self.pre_crop_cfg["original_size"]), (
            f"Original image size {orig_img.shape[:2]} does not match pre-crop cfg {self.pre_crop_cfg['original_size']}"
        )

        if not len(pred):  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = invert_manitou_resize_crop_xyxy(pred[:, :4], self.pre_crop_cfg)
            masks = process_mask_native(proto, pred[:, 6:], pred[:, :4], self.pre_crop_cfg)  # HWC
        else:
            masks = process_mask(
                proto, pred[:, 6:], pred[:, :4], img.shape[2:], self.pre_crop_cfg, upsample=True
            )  # HWC
            pred[:, :4] = invert_manitou_resize_crop_xyxy(pred[:, :4], self.pre_crop_cfg)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

    def setup_source(self, data_cfg):
        """
        Set up source and inference mode.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor):
                Source for inference.
        """
        self.calib_params = data_cfg["calib_params"]

        # Check image size
        if isinstance(self.args.imgsz, int):
            self.args.imgsz = (self.args.imgsz, self.args.imgsz)

        # h = self.args.imgsz[0] // self.model.stride * self.model.stride
        # w = math.ceil(self.args.imgsz[1] / self.model.stride) * self.model.stride
        # scale = w / self.args.imgsz[1]
        # new_h = int(self.args.imgsz[0] * scale)
        # y1 = new_h - h
        # x1 = 0
        # y2 = new_h
        # x2 = w
        # tlbr = (y1, x1, y2, x2)
        # self.pre_crop_cfg = {
        #     "is_crop": False,
        #     "scale": 1,
        #     "crop_tlbr": tlbr,
        #     "original_size": (self.args.imgsz[0], self.args.imgsz[1]),
        # }
        # if (h, w) != (self.args.imgsz[0], self.args.imgsz[1]):
        #     self.pre_crop_cfg["is_crop"] = True
        #     self.pre_crop_cfg["scale"] = w / self.args.imgsz[1]
        #     self.imgsz = (h, w)
        # else:
        #     self.imgsz = (self.args.imgsz[0], self.args.imgsz[1])
        
        # Resize and crop image size    
        crop_size = (512, 1024)
        assert crop_size[0] % 32 == 0 and crop_size[1] % 32 == 0, "Image size must be divisible by 32."
        tlbr = (220, 0, 220 + crop_size[0], 0 + crop_size[1])
        # image resolution preprocessing
        self.pre_crop_cfg = {
            "is_crop": True,  # whether to crop images
            "scale": crop_size[1] / self.args.imgsz[1],  # scale factor for width
            "crop_tlbr": tlbr,  # top-left-bottom-right coordinates for cropping
            "original_size": (self.args.imgsz[0], self.args.imgsz[1]),
            "crop_size": crop_size # ADD Crop size for mask processing
        }
        LOGGER.warning(
            f"Image size {self.args.imgsz} will be resized and cropped to {crop_size} for prediction."
        )
        # Update image size for the model
        self.imgsz = crop_size

        if self.use_radar:
            if self.pre_crop_cfg["is_crop"]:  # if use the ManitouResizeCrop_MultiImg, we need to update the camera intrinsics
                LOGGER.info("Updating camera intrinsics for Manitou dataset with pre-crop configuration.")
                # update camera intrinsics
                # h, w = self.pre_crop_cfg["original_size"]
                # crop_h, crop_w = self.pre_crop_cfg["target_size"]
                # new_h, new_w = int(h *self.pre_crop_cfg["scale"]), int(w * self.pre_crop_cfg["scale"])
                # y_off = new_h - crop_h
                # for cam_idx in range(1, 5):
                #     mat_K = self.calib_params[f"camera{cam_idx}_K"]
                #     cvt_mat = np.array([
                #                     [self.pre_crop_cfg["scale"], 0,                           0],
                #                     [0,                          self.pre_crop_cfg["scale"],  -y_off],
                #                     [0,                          0,                           1]
                #                 ], dtype=mat_K.dtype)
                #     self.calib_params[f"new_camera{cam_idx}_K"] = cvt_mat @ mat_K
                from ultralytics.data.augmentV1 import ManitouResizeCrop
                pre_crop = ManitouResizeCrop(
                    self.pre_crop_cfg["scale"],
                    self.pre_crop_cfg["crop_tlbr"],
                    1.0 if self.pre_crop_cfg["is_crop"] else 0.0,)
                for cam_idx in range(1, 5):
                    self.calib_params[f"new_camera{cam_idx}_K"] = pre_crop.update_camera_intrinsics(self.calib_params[f"camera{cam_idx}_K"])

        radar_accumulation = data_cfg.pop("radar_accumulation", 3)
        
        self.dataset = LoadManitouImagesAndRadar(data_cfg=data_cfg,
                                                 radar_accumulation=radar_accumulation,
                                                 batch=1,
                                                 pre_transform=self.get_pre_transform(),
                                                 use_radar=self.use_radar)                        