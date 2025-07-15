from ultralytics import YOLOManitou_MultiCam

project = "runs/manitou_reid_remap"
data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou.yaml"
epochs = 120
batch_size_per_gpu = 1
device = [1,2,3]  # list of GPU devices
batch_size = batch_size_per_gpu * len(device)  # total batch size
imgsz = (1552, 1936)  # (height, width)
# weights = "/datasets/dataset/best.pt"
weights = "/home/shu/Documents/PROTECH/ultralytics/runs/manitou_remap/train3/weights/best.pt"
model = YOLOManitou_MultiCam(weights, task='reid')  # load a model
results = model.train(
    data=data_cfg,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    device=device,
    project=project,
)  # train the model
