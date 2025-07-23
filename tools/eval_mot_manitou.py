import torch
from ultralytics import YOLOManitou
from ultralytics.utils.eval_mot_manitou_base import EvalManitouMOT

if __name__ == "__main__":
    data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou.yaml"
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
    # import math
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

    checkpoint = '/home/shu/Documents/PROTECH/ultralytics/runs/manitou_remap/train3/weights/best.pt'
    verbose = False
    num_extra_frames = 3
    
    keep_bag_names = [
        # 'rosbag2_2025_01_22-11_28_05',
        'rosbag2_2025_01_22-11_40_06'
        ]  # Specify the bag names to keep, or leave empty to use all bags
    
    model = YOLOManitou(checkpoint)

    evaluator = EvalManitouMOT(data_cfg, 
                               keep_bag_names=keep_bag_names,
                               tracker_cfg="bytetrack.yaml", 
                               model=model, 
                               imgsz=imgsz, 
                               conf_thr=0.25, 
                               pre_crop_cfg=pre_crop_cfg,
                               max_det=100, 
                               device=torch.device('cuda:1'),  # must use torch.device class to avoid always loading model to cuda:0
                               verbose=verbose, 
                               num_extra_frames=num_extra_frames)
    evaluator.run()
