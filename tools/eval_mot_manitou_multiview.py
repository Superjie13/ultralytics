from ultralytics import YOLOManitou_MultiCam
from ultralytics.utils.eval_mot_manitou_base import EvalManitouMOTMV


if __name__ == "__main__":
    project = 'runs/manitou_remap_multiview'
    data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou_mini.yaml"
    imgsz = (1552, 1936)  # (height, width)
    checkpoint = '/home/shu/Documents/PROTECH/ultralytics/runs/manitou_remap/train3/weights/best.pt'
    verbose = False
    num_extra_frames = 3

    keep_bag_names = [
        # 'rosbag2_2025_01_22-10_53_25'
        ]
    
    model = YOLOManitou_MultiCam(checkpoint)  # load a model

    evaluator = EvalManitouMOTMV(data_cfg, 
                                keep_bag_names=keep_bag_names,
                                tracker_cfg="mvtrack.yaml", 
                                model=model, 
                                imgsz=imgsz, 
                                conf_thr=0.25, 
                                max_det=100, 
                                device='cuda:0',
                                verbose=verbose, 
                                num_extra_frames=num_extra_frames)
    evaluator.run()
    
