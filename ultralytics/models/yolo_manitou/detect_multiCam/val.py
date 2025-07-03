import torch
import numpy as np

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.data import build_manitou_dataset, ManitouAPI, build_dataloader, get_manitou_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou, compute_reid_map, compute_ap
from ultralytics.utils.ops import Profile
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.models.yolo_manitou.detect import ManitouValidator
from ultralytics.nn.autobackend import AutoBackend


class ManitouValidator_MultiCam(ManitouValidator):
    """
    A class extending the DetectionValidator class for validation based on a Manitou detection model.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm, optional): Progress bar for displaying progress.
            args (SimpleNamespace, optional): Configuration for validation.
            _callbacks (Callbacks, optional): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        self.nt_per_class = None
        self.nt_per_image = None
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        
        self.eval_tracking = False

        self.metricReid = {
            'map50-95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'VP': 0.0,
            'FP': 0.0,
            'FN': 0.0,
            'VN': 0.0,
            'thresh': 0.7
        }
        self.metricReid_counts = 0

    def preprocess(self, batch):
        key_frames = batch["key_frames"]
        ref_frames = batch["ref_frames"]
        
        key_frames["img"] = key_frames["img"].to(self.device, non_blocking=True)
        key_frames["img"] = (key_frames["img"].half() if self.args.half else key_frames["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            key_frames[k] = key_frames[k].to(self.device)
        
        if ref_frames is not None:
            ref_frames["img"] = ref_frames["img"].to(self.device, non_blocking=True)
            ref_frames["img"] = (ref_frames["img"].half() if self.args.half else ref_frames["img"].float()) / 255
            for k in ["batch_idx", "cls", "bboxes"]:
                ref_frames[k] = ref_frames[k].to(self.device)

        return batch

    def init_metrics(self, model):
        self._init_det_metrics(model)

    def _init_det_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        self.class_map = list(range(1, len(model.names) + 1))
        if self.names is not None:
            assert self.names == model.names, f"Model names: {model.names} do not match dataloader names: {self.names}"
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            stats (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            self.stride = model.stride
            pt, jit, engine = model.pt, model.jit, model.engine
            if engine:
                self.args.batch = model.batch_size
            elif not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch}")
            
            self.data = get_manitou_dataset(self.args.data)
            self.names = self.data["names"]

            model.names = self.names

            dataset = build_manitou_dataset(
                                            cfg=self.args,
                                            ann_path=self.data["val"],
                                            batch=self.args.batch,
                                            data=self.data,
                                            mode="val",
                                            stride=self.stride,
                                            multi_cam=True,
                                        )
            
            self.dataloader = self.dataloader or build_dataloader(dataset, self.args.batch, self.args.workers, shuffle=False, rank=-1)
            
            imgsz = dataset.imgsz
            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz[0], imgsz[1]))  # warmup
            
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )

        self.metricReid = {k: 0 if k != 'thresh' else 0.7 for k in self.metricReid}
        self.metricReid_counts = 0

        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        
        model.is_train = False
        if isinstance(model, AutoBackend):
            model = model.model
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

                # Vérification : skip si key frames n'ont aucune GT box
                key_bboxes = batch["key_frames"]['bboxes']

                if isinstance(key_bboxes, list):
                    total_key = sum([b.size(0) for b in key_bboxes])
                else:
                    total_key = key_bboxes.size(0)

                if total_key == 0:
                    print(f"Skip batch {batch_i}: no GT boxes in key frames")
                    continue

            # Inference 
            with dt[1]:
                preds, features = model(batch['key_frames']['img'], embed=model.featmap_idxs)
            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds, features)[1]
                    
            # Postprocess
            with dt[3]:
                preds, feat_preds = self.postprocess_forReid(preds, features)
                
            self.update_metrics_wReid(preds, feat_preds, batch["key_frames"])
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch["key_frames"], batch_i)
                self.plot_predictions(batch["key_frames"], preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        
        for key in ['map50-95', 'precision', 'recall']:
            if self.metricReid_counts > 0:
                self.metricReid[key] = self.metricReid[key]/self.metricReid_counts
            else:
                self.metricReid[key] = 0.0
        
        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        print("Reidentification mAP, Precision and Recall : ", self.metricReid)
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            #results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val"), 
            "reid_map": self.metricReid['map50-95'],
            "reid_precision": self.metricReid['precision'], "reid_recall": self.metricReid['recall'], "reid_VP": self.metricReid['VP'],
            "reid_FP": self.metricReid['FP'],
            "reid_FN": self.metricReid['FN'],
            "reid_VN": self.metricReid['VN'],}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            return stats
            
    def update_metrics_wReid(self, preds, features, batch):
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing ground truth.
        """        
        feat_list = []
        ids_list = []
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox, ids = pbatch.pop("cls"), pbatch.pop("bbox"), pbatch.pop("ins_ids")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )
            
            # Skip ReID computation if no predictions or no ground truths
            if predn.shape[0] == 0 or bbox.shape[0] == 0:
                continue

            ious = box_iou(predn[:, :4], bbox)
            if ious.numel() == 0 or ious.size(1) == 0:
                continue  # Avoid reduction on empty tensors

            #ious = box_iou(predn[:, :4], bbox)
            max_ious, gt_indices = ious.max(dim=1)
            keep = max_ious > 0.65
            if keep.sum() == 0:
                continue  # No valid associations

            kept_features = features[si][keep]
            matched_gt_indices = gt_indices[keep]
            matched_gt_ids = ids[matched_gt_indices]

            feat_list.append(kept_features)
            ids_list.append(matched_gt_ids)

        nb_cam = max(batch["cam_idx"])
        B = len(ids_list) // nb_cam

        feat_batches = []
        ids_batches = []
        for b in range(B):
            #feat_b = torch.cat(feat_list[b * nb_cam:(b + 1) * nb_cam], dim=0)
            #ids_b = torch.cat(ids_list[b * nb_cam:(b + 1) * nb_cam], dim=0)
            chunk_feats = feat_list[b * nb_cam:(b + 1) * nb_cam]
            chunk_ids = ids_list[b * nb_cam:(b + 1) * nb_cam]
            if len(chunk_feats) == 0 or len(chunk_ids) == 0:
                continue
            try:
                feat_b = torch.cat(chunk_feats, dim=0)
                ids_b = torch.cat(chunk_ids, dim=0)
            except RuntimeError:
                continue  # Skip if one of the entries is empty

            if ids_b.numel() == 0:
                continue
            
            precision_list = []
            recall_list = []
            for thresh in [round(x, 2) for x in np.arange(0.5, 0.95 + 1e-9, 0.05)]:
                reid_metrics = compute_reid_map(feat_b, ids_b, thresh=thresh)         
                precision_list.append(reid_metrics["precision"])
                recall_list.append(reid_metrics["recall"])
            
            map50_95 = compute_ap(recall_list, precision_list)[0]
            self.metricReid['map50-95'] += map50_95
            reid_metrics = compute_reid_map(feat_b, ids_b, thresh=0.7)         
            self.metricReid_counts += 1
            for key in ['precision', 'recall', 'VP', 'FP', 'FN', 'VN']:
                self.metricReid[key] += reid_metrics[key]


    def postprocess_forReid(self, preds, features):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[torch.Tensor]): Processed predictions after NMS.
        """
        return ops.non_max_suppression_forReid(
            preds,
            features,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )            

    def _prepare_batch(self, si, batch):
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (dict): Batch data containing images and annotations.

        Returns:
            (dict): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        ids = torch.cat(batch["ins_ids"], dim=0).squeeze(1).long().to(self.device)[idx]
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "ins_ids": ids , "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}