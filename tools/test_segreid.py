from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.colors as mcolors
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

data_root = "/datasets/dataset/manitou"
weights = "/root/workspace/ultralytics/tools/runs/manitou_segreid_remap/train9/weights/best.pt"
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

crop_size = (512, 1024)
assert crop_size[0] % 32 == 0 and crop_size[1] % 32 == 0, "Image size must be divisible by 32 for training and validation."
tlbr = (220, 0, 220 + crop_size[0], 0 + crop_size[1])
pre_crop_cfg["is_crop"] = True
pre_crop_cfg["scale"] = crop_size[1] / imgsz[1]
pre_crop_cfg["crop_tlbr"] = tlbr
pre_crop_cfg["crop_size"] = crop_size

model = YOLOManitou_MultiCam(weights, task="segment_reid")

calib_params = get_manitou_calibrations("/datasets/dataset/manitou/calibration/")

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
all_masks = []

for cam_idx, res in enumerate(results[0][0]):
    print(f"Camera {cam_idx}")

    img = res.orig_img.copy()
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
    features = res.feats.cpu().numpy() if res.feats is not None else []
    masks = res.masks.data.cpu().numpy() if res.masks is not None else []

    n = len(boxes)
    indices = list(range(global_index, global_index + n))
    global_index += n

    camera_idx.extend([cam_idx] * n)
    all_boxes.extend(boxes)
    all_confs.extend(confs)
    all_img_indices.extend([cam_idx] * n)
    all_masks.extend(masks)

    processed_images.append(img)
    features_list.append(features)

# Réidentification
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

top_matches = torch.argmax(similarity_matrix, dim=1).tolist()

# Affichage avec masques et ReID
annotated_images = [img.copy() for img in processed_images]

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

for i, (box, conf, img_idx, match_idx, mask) in enumerate(zip(all_boxes, all_confs, all_img_indices, top_matches, all_masks)):
    x1, y1, x2, y2 = map(int, box)
    label = f"ID: {i+1} -> ReID: {match_idx+1}"

    cv2.rectangle(annotated_images[img_idx], (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated_images[img_idx], label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # masque
    if mask.shape != (imgsz[0], imgsz[1]):
        mask = cv2.resize(mask, (imgsz[1], imgsz[0]))

    mask = (mask > 0.5).astype(np.uint8) 

    color = (0, 255, 0)
    alpha = 0.3

    colored_mask = np.zeros_like(annotated_images[img_idx], dtype=np.uint8)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]

    mask_indices = mask.astype(bool)
    annotated_img = annotated_images[img_idx]
    annotated_img[mask_indices] = (
        alpha * colored_mask[mask_indices] + (1 - alpha) * annotated_img[mask_indices]
    ).astype(np.uint8)

# === Sauvegarde finale ===
target_size = (1920, 1536)
resized_images = [cv2.resize(img, target_size) for img in annotated_images]
top_row = np.hstack([resized_images[0], resized_images[2]])
bottom_row = np.hstack([resized_images[1], resized_images[3]])
combined_image = np.vstack([top_row, bottom_row])

cv2.imwrite("grid_output_segment_reid.jpg", combined_image)
print("Image enregistrée : grid_output_segment_reid.jpg")