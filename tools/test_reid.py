from pathlib import Path
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLOManitou_MultiCam
from ultralytics.data.manitou_api import get_manitou_calibrations


def draw_results_on_image(img, boxes, confs, global_indices):
    for i, (box, conf, idx) in enumerate(zip(boxes, confs, global_indices)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID:{idx} Conf:{conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img


data_root = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou"
weights = "/home/shu/Documents/PROTECH/ultralytics/runs/manitou_reid_remap/train/weights/last.pt"
device = [1]
imgsz = (1552, 1936)  # (height, width)

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

model = YOLOManitou_MultiCam(weights)
calib_params = get_manitou_calibrations("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/calibration/")

rosbag_name = "rosbag2_2025_02_17-14_29_31"
frame = "000076"


cam1_paths = os.path.join(data_root, f"key_frames/{rosbag_name}/camera1/{frame}.jpg")
cam2_paths = os.path.join(data_root, f"key_frames/{rosbag_name}/camera2/{frame}.jpg")
cam3_paths = os.path.join(data_root, f"key_frames/{rosbag_name}/camera3/{frame}.jpg")
cam4_paths = os.path.join(data_root, f"key_frames/{rosbag_name}/camera4/{frame}.jpg")

data_cfg = {
    "camera1": [cam1_paths],
    "camera2": [cam2_paths],
    "camera3": [cam3_paths],
    "camera4": [cam4_paths],
    "radar1": None,
    "radar2": None,
    "radar3": None,
    "radar4": None,
    "calib_params": calib_params,
    "filter_cfg": {},
    "radar_accumulation": 1,
}

results = model.predict(data_cfg=data_cfg, imgsz=imgsz, conf=0.25, max_det=100, pre_crop_cfg=pre_crop_cfg, save=False)

processed_images = []
global_index = 1
features_list = []
camera_idx = []
all_boxes = []
all_confs = []
all_img_indices = []

for cam_idx, res in enumerate(results[0][0]):
    print(cam_idx)
    img = res.orig_img.copy()
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
    features = res.feats.cpu().numpy() if res.feats is not None else []
    n = len(boxes)
    indices = list(range(global_index, global_index + n))
    global_index += n

    camera_idx.extend([cam_idx] * n)
    all_boxes.extend(boxes)
    all_confs.extend(confs)
    all_img_indices.extend([cam_idx] * n)

    processed_images.append(img)
    features_list.append(features)

features_tensor = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32)

features_norm = F.normalize(features_tensor, p=2, dim=1)
similarity_matrix = torch.mm(features_norm, features_norm.t())

camera_idx_tensor = torch.tensor(camera_idx)
same_cam_mask = camera_idx_tensor.unsqueeze(0) == camera_idx_tensor.unsqueeze(1)
similarity_matrix[same_cam_mask] = 0

print("Cosine similarity matrix :")
for i in range(similarity_matrix.shape[0]):
    line = [f"{similarity_matrix[i, j].item():.3f}" for j in range(similarity_matrix.shape[1])]
    print(" ".join(line))

# === Ajout : matrice de réidentification binaire ===
thresh = 0.7
reid_matrix = torch.zeros_like(similarity_matrix, dtype=torch.int)

for i in range(similarity_matrix.shape[0]):
    max_sim, max_j = torch.max(similarity_matrix[i], dim=0)
    if max_sim.item() > thresh:
        reid_matrix[i, max_j] = 1

print("\nRe-identification binary matrix (thresh=0.7):")
for i in range(reid_matrix.shape[0]):
    line = [str(reid_matrix[i, j].item()) for j in range(reid_matrix.shape[1])]
    print(" ".join(line))

top_matches = torch.argmax(similarity_matrix, dim=1).tolist()

annotated_images = [img.copy() for img in processed_images]
for i, (box, conf, img_idx, match_idx) in enumerate(zip(all_boxes, all_confs, all_img_indices, top_matches)):
    x1, y1, x2, y2 = map(int, box)
    # label = f"ID: {i+1} -> ReID: {match_idx+1}"
    label = f"ID: {i + 1}"
    cv2.rectangle(annotated_images[img_idx], (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated_images[img_idx], label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

target_size = (1920, 1536)
resized_images = [cv2.resize(img, target_size) for img in annotated_images]

top_row = np.hstack([resized_images[0], resized_images[2]])
bottom_row = np.hstack([resized_images[1], resized_images[3]])
combined_image = np.vstack([top_row, bottom_row])

cv2.imwrite("grid_output.jpg", combined_image)
print("Image enregistrée : grid_output.jpg")
