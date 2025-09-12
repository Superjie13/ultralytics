from typing import Optional, Dict, List
import cv2
import numpy as np
from pathlib import Path
import torch


def build_positional_encoding(pre_crop_cfg=None):
    try:
        cam1_pos_enc_p3 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam1_pos_enc_p3.pth", map_location="cpu")
        cam2_pos_enc_p3 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam2_pos_enc_p3.pth", map_location="cpu")
        cam3_pos_enc_p3 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam3_pos_enc_p3.pth", map_location="cpu")
        cam4_pos_enc_p3 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam4_pos_enc_p3.pth", map_location="cpu")

        cam1_pos_enc_p4 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam1_pos_enc_p4.pth", map_location="cpu")
        cam2_pos_enc_p4 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam2_pos_enc_p4.pth", map_location="cpu")
        cam3_pos_enc_p4 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam3_pos_enc_p4.pth", map_location="cpu")
        cam4_pos_enc_p4 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam4_pos_enc_p4.pth", map_location="cpu")

        cam1_pos_enc_p5 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam1_pos_enc_p5.pth", map_location="cpu")
        cam2_pos_enc_p5 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam2_pos_enc_p5.pth", map_location="cpu")
        cam3_pos_enc_p5 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam3_pos_enc_p5.pth", map_location="cpu")
        cam4_pos_enc_p5 = torch.load("/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/cam4_pos_enc_p5.pth", map_location="cpu")

        pos_enc = {
            "cam1_pos_enc_p3": cam1_pos_enc_p3,
            "cam2_pos_enc_p3": cam2_pos_enc_p3,
            "cam3_pos_enc_p3": cam3_pos_enc_p3,
            "cam4_pos_enc_p3": cam4_pos_enc_p3,
            "cam1_pos_enc_p4": cam1_pos_enc_p4,
            "cam2_pos_enc_p4": cam2_pos_enc_p4,
            "cam3_pos_enc_p4": cam3_pos_enc_p4,
            "cam4_pos_enc_p4": cam4_pos_enc_p4,
            "cam1_pos_enc_p5": cam1_pos_enc_p5,
            "cam2_pos_enc_p5": cam2_pos_enc_p5,
            "cam3_pos_enc_p5": cam3_pos_enc_p5,
            "cam4_pos_enc_p5": cam4_pos_enc_p5
        }

        print("Positional encodings loaded from .pth files")

    except FileNotFoundError:
        assert pre_crop_cfg is not None, "pre_crop_cfg must be provided if .pth files are not found."
        scale = pre_crop_cfg["scale"]
        tlbr = pre_crop_cfg["crop_tlbr"]
        # positional encoding
        cam1_pos_enc_path = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/camera1_lut_pos_enc.npy"
        cam2_pos_enc_path = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/camera2_lut_pos_enc.npy"
        cam3_pos_enc_path = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/camera3_lut_pos_enc.npy"
        cam4_pos_enc_path = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/lut/camera4_lut_pos_enc.npy"

        # Load positional encodings
        cam1_pos_enc = np.load(cam1_pos_enc_path)
        cam2_pos_enc = np.load(cam2_pos_enc_path)
        cam3_pos_enc = np.load(cam3_pos_enc_path)
        cam4_pos_enc = np.load(cam4_pos_enc_path)

        # resize and crop based on pre_crop_cfg
        # 1) resize based on pre_crop_cfg["scale"]
        cam1_pos_enc_resized = cv2.resize(cam1_pos_enc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cam2_pos_enc_resized = cv2.resize(cam2_pos_enc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cam3_pos_enc_resized = cv2.resize(cam3_pos_enc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cam4_pos_enc_resized = cv2.resize(cam4_pos_enc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2) crop based on tlbr
        cam1_pos_enc = cam1_pos_enc_resized[tlbr[0]:tlbr[2], tlbr[1]:tlbr[3]]
        cam2_pos_enc = cam2_pos_enc_resized[tlbr[0]:tlbr[2], tlbr[1]:tlbr[3]]
        cam3_pos_enc = cam3_pos_enc_resized[tlbr[0]:tlbr[2], tlbr[1]:tlbr[3]]
        cam4_pos_enc = cam4_pos_enc_resized[tlbr[0]:tlbr[2], tlbr[1]:tlbr[3]]
        
        # stride to [1/8, 1/16, 1/32]
        cam1_pos_enc_p3 = cv2.resize(cam1_pos_enc, (cam1_pos_enc.shape[1] // 8, cam1_pos_enc.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
        cam2_pos_enc_p3 = cv2.resize(cam2_pos_enc, (cam2_pos_enc.shape[1] // 8, cam2_pos_enc.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
        cam3_pos_enc_p3 = cv2.resize(cam3_pos_enc, (cam3_pos_enc.shape[1] // 8, cam3_pos_enc.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
        cam4_pos_enc_p3 = cv2.resize(cam4_pos_enc, (cam4_pos_enc.shape[1] // 8, cam4_pos_enc.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
        
        cam1_pos_enc_p4 = cv2.resize(cam1_pos_enc, (cam1_pos_enc.shape[1] // 16, cam1_pos_enc.shape[0] // 16), interpolation=cv2.INTER_CUBIC)
        cam2_pos_enc_p4 = cv2.resize(cam2_pos_enc, (cam2_pos_enc.shape[1] // 16, cam2_pos_enc.shape[0] // 16), interpolation=cv2.INTER_CUBIC)
        cam3_pos_enc_p4 = cv2.resize(cam3_pos_enc, (cam3_pos_enc.shape[1] // 16, cam3_pos_enc.shape[0] // 16), interpolation=cv2.INTER_CUBIC)
        cam4_pos_enc_p4 = cv2.resize(cam4_pos_enc, (cam4_pos_enc.shape[1] // 16, cam4_pos_enc.shape[0] // 16), interpolation=cv2.INTER_CUBIC)
        
        cam1_pos_enc_p5 = cv2.resize(cam1_pos_enc, (cam1_pos_enc.shape[1] // 32, cam1_pos_enc.shape[0] // 32), interpolation=cv2.INTER_CUBIC)
        cam2_pos_enc_p5 = cv2.resize(cam2_pos_enc, (cam2_pos_enc.shape[1] // 32, cam2_pos_enc.shape[0] // 32), interpolation=cv2.INTER_CUBIC)
        cam3_pos_enc_p5 = cv2.resize(cam3_pos_enc, (cam3_pos_enc.shape[1] // 32, cam3_pos_enc.shape[0] // 32), interpolation=cv2.INTER_CUBIC)
        cam4_pos_enc_p5 = cv2.resize(cam4_pos_enc, (cam4_pos_enc.shape[1] // 32, cam4_pos_enc.shape[0] // 32), interpolation=cv2.INTER_CUBIC)

        pos_enc = {
            "cam1_pos_enc": cam1_pos_enc,
            "cam2_pos_enc": cam2_pos_enc,
            "cam3_pos_enc": cam3_pos_enc,
            "cam4_pos_enc": cam4_pos_enc,
            "cam1_pos_enc_p3": cam1_pos_enc_p3,
            "cam2_pos_enc_p3": cam2_pos_enc_p3,
            "cam3_pos_enc_p3": cam3_pos_enc_p3,
            "cam4_pos_enc_p3": cam4_pos_enc_p3,
            "cam1_pos_enc_p4": cam1_pos_enc_p4,
            "cam2_pos_enc_p4": cam2_pos_enc_p4,
            "cam3_pos_enc_p4": cam3_pos_enc_p4,
            "cam4_pos_enc_p4": cam4_pos_enc_p4,
            "cam1_pos_enc_p5": cam1_pos_enc_p5,
            "cam2_pos_enc_p5": cam2_pos_enc_p5,
            "cam3_pos_enc_p5": cam3_pos_enc_p5,
            "cam4_pos_enc_p5": cam4_pos_enc_p5,
        }

        print("Positional encodings loaded from .npy files")

    return pos_enc
    

def prepare_pos_enc(pos_enc: Dict, cam_idx: List):
        """
        Prepare position encodings for all cameras listed in cam_idx.
        Args:
            pos_enc (dict): including the pyramid position encodings at P3(1/8), P4(1/16), P5(1/32).
            cam_idx (list): list of camera indices to prepare position encodings for.
        Returns:
            list: prepared position encodings of all cameras at different levels.
        """
        p3_list = [pos_enc[f"cam{i}_pos_enc_p3"] for i in cam_idx]
        p4_list = [pos_enc[f"cam{i}_pos_enc_p4"] for i in cam_idx]
        p5_list = [pos_enc[f"cam{i}_pos_enc_p5"] for i in cam_idx]

        # stack to batch
        p3_pos_enc = torch.concat(p3_list, dim=0)  # [B, C, H, W]
        p4_pos_enc = torch.concat(p4_list, dim=0)  # [B, C, H, W]
        p5_pos_enc = torch.concat(p5_list, dim=0)  # [B, C, H, W]

        return [p3_pos_enc, p4_pos_enc, p5_pos_enc]