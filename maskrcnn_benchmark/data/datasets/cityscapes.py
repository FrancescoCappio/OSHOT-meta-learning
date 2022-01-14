import os

import torch
import torch.utils.data
from PIL import Image
import sys

import json


from maskrcnn_benchmark.structures.bounding_box import BoxList


class CityscapesDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.foggy = "foggy" in split

        if split == 'foggy_train':
            split = 'train'
            self._root_img_path = os.path.join(self.root, 'leftImg8bit_trainval_foggyDBF', 'leftImg8bit_foggyDBF', split)
            self._imgpath = os.path.join(self._root_img_path, '%s', '%s_leftImg8bit_foggy_beta_0.02.png')
        elif split == 'foggy_val':
            split = 'val'
            self._root_img_path = os.path.join(self.root, 'leftImg8bit_trainval_foggyDBF', 'leftImg8bit_foggyDBF', split)
            self._imgpath = os.path.join(self._root_img_path, '%s', '%s_leftImg8bit_foggy_beta_0.02.png')
        else:
            self._root_img_path = os.path.join(self.root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split)
            self._imgpath = os.path.join(self._root_img_path, '%s', '%s_leftImg8bit.png')

        self._annopath = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', split, '%s', '%s_gtFine_polygons.json')

        ids = []
        for city in os.listdir(os.path.join(self._root_img_path)):
            city_ids = os.listdir(os.path.join(self._root_img_path, city))
            if self.foggy:
                city_ids = list(filter(lambda x: x.split('_')[-1] == '0.02.png', city_ids))
                end_prefix = -4
            else:
                end_prefix = -1
            city_ids = ['_'.join(city_id.split('_')[:end_prefix]) for city_id in city_ids]
            ids.extend(city_ids)

        self._cls = CityscapesDataset.CLASSES

        self.ids = [img_id for img_id in ids if self._boxes_count(img_id)]

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.class_to_ind = dict(zip(self._cls, range(len(self._cls))))
        self.categories = dict(zip(range(len(self._cls)), self._cls))

    def set_keep_difficult(self, value):
        self.keep_difficult = value

    def set_transforms(self, transforms):
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % (img_id.split('_')[0], img_id)).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_img_name(self, index):
        return self.ids[index]

    def __len__(self):
        return len(self.ids)

    def _boxes_count(self, img_id):
        anno = json.loads(open(self._annopath % (img_id.split('_')[0], img_id)).read())
        boxes = 0
        for obj in anno['objects']:
            boxes += int(obj['label'] in self._cls)
        return boxes

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = json.loads(open(self._annopath % (img_id.split('_')[0], img_id)).read())
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _bbox_from_polygon(self, polygon):
        xmin, ymin, xmax, ymax = 2048, 1024, 0, 0
        for x,y in polygon:
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        return xmin, ymin, xmax, ymax

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []

        for obj in target["objects"]:
            if obj['label'] in self._cls:
                difficult = 0
                name = obj['label']
                polygon = obj['polygon']
                bndbox = self._bbox_from_polygon(polygon)

                boxes.append(bndbox)
                gt_classes.append(self.class_to_ind[name])
                difficult_boxes.append(difficult)

        im_info = target['imgHeight'], target['imgWidth']

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = json.loads(open(self._annopath % (img_id.split('_')[0], img_id)).read())
        return {"height": anno['imgHeight'], "width": anno['imgWidth']}

    def map_class_id_to_class_name(self, class_id):
        return CityscapesDataset.CLASSES[class_id]
