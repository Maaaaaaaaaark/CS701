import torch
from torchvision import tv_tensors

from sarbench.data import quarter_turn


def test_quarter_turn_rotates_pixels_boxes_and_canvas_together():
    image = tv_tensors.Image(torch.arange(24).reshape(1, 4, 6))
    boxes = tv_tensors.BoundingBoxes([[1.0, 1.0, 5.0, 3.0]], format='XYXY', canvas_size=(4, 6))

    rotated_image, rotated_boxes = quarter_turn(image, boxes, 1)

    assert rotated_image.shape == (1, 6, 4)
    assert rotated_boxes.canvas_size == (6, 4)
    torch.testing.assert_close(rotated_boxes.as_subclass(torch.Tensor), torch.tensor([[1.0, 1.0, 3.0, 5.0]]))
    torch.testing.assert_close(rotated_image.as_subclass(torch.Tensor), torch.rot90(image, 1, (-2, -1)))


def test_four_quarter_turns_are_identity():
    image = tv_tensors.Image(torch.arange(24).reshape(1, 4, 6))
    boxes = tv_tensors.BoundingBoxes([[1.0, 1.0, 5.0, 3.0]], format='XYXY', canvas_size=(4, 6))

    rotated_image, rotated_boxes = quarter_turn(image, boxes, 4)

    torch.testing.assert_close(rotated_image, image)
    torch.testing.assert_close(rotated_boxes, boxes)
    assert rotated_boxes.canvas_size == boxes.canvas_size
