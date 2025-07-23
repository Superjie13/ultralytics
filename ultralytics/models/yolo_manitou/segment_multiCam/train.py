from copy import copy

from ultralytics.models import yolo_manitou
from ultralytics.nn.tasks import SegmentationModel_MultiView
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class ManitouSegmentationTrainer_MultiCam(yolo_manitou.detect_multiCam.ManitouTrainer_MultiCam):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

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
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
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