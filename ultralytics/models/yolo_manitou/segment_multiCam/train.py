from copy import copy
import torch
from torch import distributed as dist
import torch.nn as nn
import math
import time

from ultralytics.models import yolo_manitou
from ultralytics.nn.tasks import SegmentationModel_MultiView
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    callbacks,
    colorstr,
)
from ultralytics.utils.checks import check_amp
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    de_parallel,
    unset_deterministic,
)

class ManitouSegmentationTrainer_MultiCam(yolo_manitou.detect_multiCam_reid.ManitouTrainer_MultiCam_ReID):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        #overrides["task"] = "segment_reid"
        super().__init__(cfg, overrides, _callbacks)

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        # target_w = math.ceil(self.args.imgsz[1] / gs) * gs
        # target_h = math.ceil
        # self.args.imgsz = (self.args.imgsz[0] // gs * gs, self.args.imgsz[1] // gs * gs)  # grid size (multiple of gs)
        # LOGGER.info(f"Image will be cropped to {self.args.imgsz} for training. i.e. crop (0: self.args.imgsz[0], 0: self.args.imgsz[1])")
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "reid_loss"
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return a SegmentationModel with specified configuration and weights.

        Args:
            cfg (dict | str | None): Model configuration. Can be a dictionary, a path to a YAML file, or None.
            weights (str | Path | None): Path to pretrained weights file.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (SegmentationModel): Initialized segmentation model with loaded weights if specified.

        Examples:
            >>> trainer = SegmentationTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-seg.yaml")
            >>> model = trainer.get_model(weights="yolo11n-seg.pt", verbose=False)
        """
        model = SegmentationModel_MultiView(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "reid_loss"
        return yolo_manitou.segment_multiCam.ManitouSegmentationValidator_MultiCam(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni, prefix):
        """
        Plot training sample images with labels, bounding boxes, and masks.

        This method creates a visualization of training batch images with their corresponding labels, bounding boxes,
        and segmentation masks, saving the result to a file for inspection and debugging.

        Args:
            batch (dict): Dictionary containing batch data with the following keys:
                'img': Images tensor
                'batch_idx': Batch indices for each box
                'cls': Class labels tensor (squeezed to remove last dimension)
                'bboxes': Bounding box coordinates tensor
                'masks': Segmentation masks tensor
                'im_file': List of image file paths
            ni (int): Current training iteration number, used for naming the output file.

        Examples:
            >>> trainer = SegmentationTrainer()
            >>> batch = {
            ...     "img": torch.rand(16, 3, 640, 640),
            ...     "batch_idx": torch.zeros(16),
            ...     "cls": torch.randint(0, 80, (16, 1)),
            ...     "bboxes": torch.rand(16, 4),
            ...     "masks": torch.rand(16, 640, 640),
            ...     "im_file": ["image1.jpg", "image2.jpg"],
            ... }
            >>> trainer.plot_training_samples(batch, ni=5)
        """
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"{prefix}_train_batch_{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png