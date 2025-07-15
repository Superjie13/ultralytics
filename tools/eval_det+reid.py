from ultralytics import YOLOManitou_MultiCam

project = "runs/manitou_reid_remap"
data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou.yaml"
# weights = "/root/workspace/ultralytics/tools/runs/manitou_remap/train/weights/best.pt"
weights = "/home/shu/Documents/PROTECH/ultralytics/runs/manitou_reid_remap/train2/weights/last.pt"

device = [1]
batch_size = 8
imgsz = (1552, 1936)  # (height, width)
model = YOLOManitou_MultiCam(weights)
metrics = model.val(data=data_cfg, imgsz=imgsz, batch=batch_size, device=device)
