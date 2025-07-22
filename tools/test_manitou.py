from ultralytics import YOLOManitou

# Test the prediction
imgsz = (1552, 1936)  # (height, width)
# path = '/home/shu/Documents/PROTECH/ultralytics/datasets/manitou_mini/data/rosbag2_2025_02_17-14_04_34/camera1'
path = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/frames/rosbag2_2025_01_22-11_40_06/camera1/"
checkpoint = "/home/shu/Documents/PROTECH/ultralytics/runs/manitou_remap/train3/weights/best.pt"
model = YOLOManitou(checkpoint)

# Get pre_crop_cfg
pre_crop_cfg = {
    "is_crop": False,  
    "scale": 1.0,
    "crop_tlbr": (0, 0, 0, 0),
    "crop_size": [imgsz[0], imgsz[1]],
    "original_size": [imgsz[0], imgsz[1]],
}

# h = imgsz[0] // 32 * 32
# w = math.ceil(imgsz[1] / 32) * 32
# scale = w / imgsz[1]
# new_h = int(imgsz[0] * scale)
# y1 = new_h - h
# x1 = 0
# y2 = new_h
# x2 = w
# tlbr = (y1, x1, y2, x2)

# if imgsz != (h, w):
#     pre_crop_cfg["is_crop"] = True
#     pre_crop_cfg["scale"] = w / imgsz[1]
#     pre_crop_cfg["crop_tlbr"] = tlbr
#     pre_crop_cfg["crop_size"] = [h, w]

crop_size = (512, 1024)
assert crop_size[0] % 32 == 0 and crop_size[1] % 32 == 0, "Image size must be divisible by 32 for training and validation."
tlbr = (220, 0, 220 + crop_size[0], 0 + crop_size[1])
pre_crop_cfg["is_crop"] = True
pre_crop_cfg["scale"] = crop_size[1] / imgsz[1]
pre_crop_cfg["crop_tlbr"] = tlbr
pre_crop_cfg["crop_size"] = crop_size

results = model.track(source=path, imgsz=imgsz, conf=0.25, max_det=100, pre_crop_cfg=pre_crop_cfg, save_frames=True, tracker="bytetrack.yaml")
for result in results:
    # boxes = result.boxes.xyxy.cpu().numpy()  # get the bounding boxes
    # confs = result.boxes.conf.cpu().numpy()  # get the confidence scores
    cls = result.boxes.cls.cpu().numpy()  # get the class labels
    track_ids = result.boxes.id.cpu().numpy()  if result.boxes.id is not None else []

    print(f" Class Labels: {cls}, Track IDs: {track_ids}")  # print the results
    result.save(font_size=0.8, line_width=2)  # save the results

from pathlib import Path

# path = Path(path).glob("*.jpg")
# path = sorted(path)  # sort the path to get the correct order
# for p in path:
#     p = str(p)
#     result = model.predict(
#         source=p, 
#         imgsz=imgsz, 
#         conf=0.25, 
#         max_det=100, 
#         pre_crop_cfg=pre_crop_cfg,
#         save_frames=True,
#         device="cuda:1" 
#     )[0]
#     result.save(font_size=0.8, line_width=2)  # save the results
