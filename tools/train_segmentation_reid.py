from ultralytics import YOLOManitou_MultiCam
from ultralytics.models import yolo_manitou

project = 'runs/manitou_remap'
data_cfg = "/root/workspace/ultralytics/ultralytics/cfg/datasets/manitou.yaml" 

epochs = 1
batch_size_per_gpu = 1
device = [1]  # list of GPU devices
batch_size = batch_size_per_gpu * len(device)  # total batch size
#imgsz = (1552, 1936)  # (height, width)
imgsz = (1000, 1200)  # (height, width)
weights = "yolo11s-seg.pt"
model = YOLOManitou_MultiCam('yolo11s-seg.yaml').load(weights)  # load a model
results = model.train(data=data_cfg, epochs=epochs, imgsz=imgsz, trainer=yolo_manitou.segment_multiCam.ManitouSegmentationTrainer_MultiCam, batch=batch_size, device=device, project=project)  # train the model
