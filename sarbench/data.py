"""The course data in TA Zichen's folder layout (see README): every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133


class SARMultiTask(Dataset):
    """The images of classification/{split}.csv with their boxes from detection/instances_{split}.json.
    An item is (image [1, size, size], label 0..8, target), target = {'boxes': XYXY on the resized
    image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}."""

    def __init__(self, root, split, train=False, size=512, limit=None, val_fraction=0.1, split_seed=0):
        self.root = Path(root)
        self.official_layout = (self.root / 'train' / 'labels.csv').exists()
        if self.official_layout:
            if split not in ('train', 'val'):
                raise ValueError('The released course data has labels only for train; use train or val')
            with open(self.root / 'train' / 'labels.csv') as f:
                rows = list(csv.DictReader(f))
            # One fixed, class-stratified holdout. Save split_seed with every run for reproducibility.
            per_class = defaultdict(list)
            for row in rows:
                per_class[int(row['label'])].append(row)
            val_ids = set()
            for label, class_rows in per_class.items():
                shuffled = class_rows.copy()
                random.Random(split_seed + label).shuffle(shuffled)
                n_val = max(1, round(len(shuffled) * val_fraction))
                val_ids.update(int(row['image_id']) for row in shuffled[:n_val])
            want_val = split == 'val'
            self.rows = [row for row in rows if (int(row['image_id']) in val_ids) == want_val]
            self.rows = self.rows[:limit]
            self.gt_json = self.root / 'train' / 'instances.json'
            self.image_root = self.root / 'train'
            file_key = 'file_name'
        else:
            with open(self.root / 'classification' / f'{split}.csv') as f:
                self.rows = list(csv.DictReader(f))[:limit]
            self.gt_json = self.root / 'detection' / f'instances_{split}.json'
            self.image_root = self.root
            file_key = 'image'
        coco = json.loads(self.gt_json.read_text())
        self.file_key = file_key
        self.coco_images = {img['file_name']: img for img in coco['images']}
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((size, size)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,  # overhead imagery has no canonical "up"
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([MEAN], [STD]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        file_name = row[self.file_key]
        info = self.coco_images[file_name]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image = Image.open(self.image_root / file_name).convert('L')  # PIL also handles the release's JPEG/PNG
        image, boxes = self.transform(image, boxes)
        target = {'boxes': boxes, 'labels': torch.tensor([a['category_id'] for a in anns]),
                  'image_id': info['id'], 'orig_size': orig_size}
        return image, int(row['label']), target

    @property
    def image_ids(self):
        """COCO image ids in this subset, useful for recording the exact internal split."""
        return [int(row.get('image_id', self.coco_images[row[self.file_key]]['id'])) for row in self.rows]


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
