from ultralytics import YOLOManitou_MultiCam

project = "runs/manitou_segreid_remap"
data_cfg = "/root/workspace/ultralytics/ultralytics/cfg/datasets/manitou.yaml" 
weights = "/root/workspace/ultralytics/tools/runs/manitou_segreid_remap/train2/weights/best.pt"

device = [3]
batch_size = 2
imgsz = (1552, 1936)  # (height, width)

# Get pre_crop_cfg
pre_crop_cfg = {
    "is_crop": False,  
    "scale": 1.0,
    "crop_tlbr": (0, 0, 0, 0),
    "crop_size": [imgsz[0], imgsz[1]],
    "original_size": [imgsz[0], imgsz[1]],
}

crop_size = (512, 1024)
assert crop_size[0] % 32 == 0 and crop_size[1] % 32 == 0, "Image size must be divisible by 32 for training and validation."
tlbr = (220, 0, 220 + crop_size[0], 0 + crop_size[1])
pre_crop_cfg["is_crop"] = True
pre_crop_cfg["scale"] = crop_size[1] / imgsz[1]
pre_crop_cfg["crop_tlbr"] = tlbr
pre_crop_cfg["crop_size"] = crop_size

model = YOLOManitou_MultiCam(weights, task="segment_reid")
metrics = model.val(data=data_cfg, imgsz=imgsz, batch=batch_size, device=device, pre_crop_cfg=pre_crop_cfg)


