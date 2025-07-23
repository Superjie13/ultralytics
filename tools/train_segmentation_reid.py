from ultralytics import YOLOManitou_MultiCam
from ultralytics.models import yolo_manitou

project = 'runs/manitou_segreid_remap'
data_cfg = "/root/workspace/ultralytics/ultralytics/cfg/datasets/manitou.yaml" 

epochs = 120
batch_size_per_gpu = 4
device = [0,1]  # list of GPU devices
batch_size = batch_size_per_gpu * len(device)  # total batch size
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

weights = "yolo11s-seg.pt"
#model_cfg = "/root/workspace/ultralytics/ultralytics/cfg/models/11/yolo11-seg.yaml"
model = YOLOManitou_MultiCam(weights, task='segment_reid')  # load a model

#model = YOLOManitou_MultiCam('yolo11s-seg.yaml').load(weights)  # load a model
#results = model.train(data=data_cfg, epochs=epochs, imgsz=imgsz, trainer=yolo_manitou.segment_multiCam.ManitouSegmentationTrainer_MultiCam, batch=batch_size, device=device, project=project)  # train the model
results = model.train(
    data=data_cfg,
    epochs=epochs,
    imgsz=imgsz,
    #trainer=yolo_manitou.segment_multiCam.ManitouSegmentationTrainer_MultiCam,
    batch=batch_size,
    device=device,
    project=project,
    pre_crop_cfg=pre_crop_cfg,
)  # train the model