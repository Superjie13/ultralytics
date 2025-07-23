import os

from ultralytics import YOLOManitou_MultiCam
from ultralytics.data.manitou_api import ManitouAPI, get_manitou_calibrations

# Dataset
data_root = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou"
radar_dir = "radars"
cam_dir = "key_frames"
calib_params = get_manitou_calibrations("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/calibration/")

annotations_path = os.path.join(data_root, "annotations_multi_view", "manitou_coco_val_remap.json")
manitou = ManitouAPI(annotations_path)
manitou.info()


def get_bag_ids_from_name(name):
    valid_bag_ids = manitou.get_bag_ids()
    for bag_id in valid_bag_ids:
        if manitou.bags[bag_id]["name"] == name:
            return bag_id
    return -1


# get bag ids
selected_bag_id = get_bag_ids_from_name('rosbag2_2025_02_17-14_29_31')
video_ids = manitou.get_vid_ids(bagIds=selected_bag_id)

cam1_ids = []
cam2_ids = []
cam3_ids = []
cam4_ids = []

for video_id in video_ids:
    imgs = manitou.get_img_ids_from_vid(video_id)
    if "camera1" in manitou.vids[video_id]["name"]:
        cam1_ids = imgs
    elif "camera2" in manitou.vids[video_id]["name"]:
        cam2_ids = imgs
    elif "camera3" in manitou.vids[video_id]["name"]:
        cam3_ids = imgs
    elif "camera4" in manitou.vids[video_id]["name"]:
        cam4_ids = imgs

cam1_paths = []
cam2_paths = []
cam3_paths = []
cam4_paths = []
radar1_paths = []
radar2_paths = []
radar3_paths = []
radar4_paths = []
for c1, c2, c3, c4 in zip(cam1_ids, cam2_ids, cam3_ids, cam4_ids):
    assert (
        manitou.imgs[c1]["frame_name"]
        == manitou.imgs[c2]["frame_name"]
        == manitou.imgs[c3]["frame_name"]
        == manitou.imgs[c4]["frame_name"]
    ), "Frame names do not match across cameras"
    cam1_name = manitou.imgs[c1]["file_name"]
    cam2_name = manitou.imgs[c2]["file_name"]
    cam3_name = manitou.imgs[c3]["file_name"]
    cam4_name = manitou.imgs[c4]["file_name"]
    cam1_paths.append(os.path.join(data_root, cam_dir, cam1_name))
    cam2_paths.append(os.path.join(data_root, cam_dir, cam2_name))
    cam3_paths.append(os.path.join(data_root, cam_dir, cam3_name))
    cam4_paths.append(os.path.join(data_root, cam_dir, cam4_name))
    radar1_name = manitou.radars[c1]["file_name"]
    radar2_name = manitou.radars[c2]["file_name"]
    radar3_name = manitou.radars[c3]["file_name"]
    radar4_name = manitou.radars[c4]["file_name"]
    radar1_paths.append(os.path.join(data_root, radar_dir, radar1_name))
    radar2_paths.append(os.path.join(data_root, radar_dir, radar2_name))
    radar3_paths.append(os.path.join(data_root, radar_dir, radar3_name))
    radar4_paths.append(os.path.join(data_root, radar_dir, radar4_name))

data_cfg = {
    "camera1": cam1_paths,
    "camera2": cam2_paths,
    "camera3": cam3_paths,
    "camera4": cam4_paths,
    "radar1": radar1_paths,
    "radar2": radar2_paths,
    "radar3": radar3_paths,
    "radar4": radar4_paths,
    "calib_params": calib_params,
    "filter_cfg": {},
    "radar_accumulation": 3,  # number of frames to accumulate for radar
}

# Test the prediction
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

checkpoint = "/home/shu/Documents/PROTECH/ultralytics/runs/manitou_remap/train3/weights/best.pt"
model = YOLOManitou_MultiCam(checkpoint)

results = model.predict(
    data_cfg=data_cfg, 
    imgsz=imgsz, 
    conf=0.25, 
    max_det=100, 
    pre_crop_cfg=pre_crop_cfg,
    use_radar=True, 
    draw_radar=True, 
    save=True)

# # Test the tracking
# results = model.track(data_cfg=data_cfg,
#                       imgsz=imgsz,
#                       conf=0.25,
#                       max_det=100,
#                       tracker="mvtrack.yaml",
#                       persist=True)
# # Save the results
# for (result, radar) in results:
#     result.save(font_size=0.8, line_width=2)  # save the results
