import torch
import torch.nn.functional as F

from ultralytics.utils import ops


def invert_manitou_resize_crop_xyxy(bboxes, pre_crop_cfg):
    """
    Post-process bounding box predictions for Manitou detection. Details in `ManitouResizeCrop` data augmentation.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is the number of detections, xyxy format.
        pre_crop_cfg (dict): Configuration dictionary containing pre-crop settings.
            - is_crop (bool): Whether cropping was applied.
            - scale (float): Scaling factor used during cropping.
            - crop_tlbr (tuple): Top-left-bottom-right coordinates for cropping.
            - original_size (tuple): Original size before resizing and cropping.
    """
    if pre_crop_cfg["is_crop"]:
        scale = pre_crop_cfg["scale"]
        y1, x1, y2, x2 = pre_crop_cfg["crop_tlbr"]

        # Clone to avoid in-place modification
        boxes = bboxes.clone()
        # Shift from crop-local to scaled-image coords
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1
        # Rescale back to original image coords
        boxes /= scale

    return boxes

def invert_manitou_resize_crop_masks(masks, pre_crop_cfg):
    """
    Invert the resize and crop operation applied to masks during preprocessing.

    Args:
        masks (torch.Tensor): (N, H, W).
    """
    if pre_crop_cfg["is_crop"]:
        scale = pre_crop_cfg["scale"]
        crop_size = pre_crop_cfg["crop_size"]
        original_size = pre_crop_cfg["original_size"]
        y1, x1, y2, x2 = pre_crop_cfg["crop_tlbr"]

        scaled_h = int(original_size[0] * scale)
        scaled_w = int(original_size[1] * scale)

        masks_scaled = torch.zeros((masks.shape[0], scaled_h, scaled_w), dtype=masks.dtype, device=masks.device)
        masks_scaled[:, y1:y2, x1:x2] = masks
        masks = F.interpolate(masks_scaled.unsqueeze(0), size=original_size, mode="nearest")[0]
    
    return masks

def process_mask(protos, masks_in, bboxes, shape, pre_crop_cfg, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        pre_crop_cfg (dict): Configuration dictionary containing pre-crop settings.
            - is_crop (bool): Whether cropping was applied.
            - scale (float): Scaling factor used during cropping.
            - target_size (tuple): Target size after resizing and cropping.
            - original_size (tuple): Original size before resizing and cropping.
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = ops.crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="nearest", align_corners=False)[0]  # CHW
    masks = invert_manitou_resize_crop_masks(masks, pre_crop_cfg)
    return masks.gt_(0.0)


def process_mask_native(protos, masks_in, bboxes, pre_crop_cfg):
    """
    Apply masks to bounding boxes using the output of the mask head with native upsampling.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms.
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms.  (being inverted from manitou_resize_crop_xyxy)
        pre_crop_cfg (dict): Configuration dictionary containing pre-crop settings.
            - is_crop (bool): Whether cropping was applied.
            - scale (float): Scaling factor used during cropping.
            - target_size (tuple): Target size after resizing and cropping.
            - original_size (tuple): Original size before resizing and cropping.

    Returns:
        (torch.Tensor): The returned masks with dimensions [h, w, n].
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = invert_manitou_resize_crop_masks(masks, pre_crop_cfg)
    masks = ops.crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)

