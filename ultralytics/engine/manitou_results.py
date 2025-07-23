from ultralytics.engine.results import Results, Boxes, Masks
from ultralytics.utils import  ops

class ManitouResults(Results):
    """Manitou Results class for handling radar and camera results."""

    def update(self, boxes=None, masks=None, probs=None, **kwargs):
        """
        Updates the Results object with new detection data.

        This method allows updating the boxes, masks, probabilities, and oriented bounding boxes (OBB) of the
        Results object. It ensures that boxes are clipped to the original image shape.

        Args:
            boxes (torch.Tensor | None): A tensor of shape (N, 6) containing bounding box coordinates and
                confidence scores. The format is (x1, y1, x2, y2, conf, class).
            masks (torch.Tensor | None): A tensor of shape (N, H, W) containing segmentation masks.
            probs (torch.Tensor | None): A tensor of shape (num_classes,) containing class probabilities.

        Examples:
            >>> results = model("image.jpg")
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def new(self):
        """
        Creates a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return ManitouResults(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)
