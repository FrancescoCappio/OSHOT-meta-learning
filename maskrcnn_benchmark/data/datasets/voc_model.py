import os

import torch
import torch.utils.data
import numpy as np
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


class PascalVOCModelDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, data_dir, split, desired_classes_subset=None, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        cls = PascalVOCModelDataset.CLASSES

        if desired_classes_subset is None:
            self.keep_all_classes = True
        else:
            self.keep_all_classes = False
            self.subset_class_names = ['__background__ ']
            self.subset_class_names.extend(desired_classes_subset)
            self.subset_class_dict = {class_name: i for i, class_name in enumerate(self.subset_class_names)}

        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

        self.complete_ids_list = []

        if isinstance(self.image_set, tuple):
            for spl in self.image_set:
                img_set_split = os.path.join(self.root, "ImageSets", "Main", "%s.txt" % spl)
                self.complete_ids_list.extend(PascalVOCModelDataset._read_image_ids(img_set_split))
        else:
            image_sets_file = os.path.join(self.root, "ImageSets", "Main", "%s.txt" % self.image_set)
            self.complete_ids_list = PascalVOCModelDataset._read_image_ids(image_sets_file)

        # now we make a list with only the ids of the images containing objects of the subset
        # of the classes we are interested in
        if not self.keep_all_classes:
            self.ids = self._filter_ids()
        else:
            self.ids = self.complete_ids_list

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_img_name(self, index):
        return self.ids[index]

    def _filter_ids(self):
        """
        Iterating on the list of ids self.complete_ids_list we build up
        a new list containing only the ids of the images having at least one
        element of one of the subset of the classes we are interested in
        """
        self.inverted_subset_dict = {}
        for name in self.subset_class_names:
            self.inverted_subset_dict[self.class_to_ind[name]] = name

        filtered_ids = []

        for image_id in self.complete_ids_list:
            boxes, labels, is_difficult = self._get_annotation(image_id)

            found = False
            for idx, lbl in enumerate(labels):
                # ignore difficult instances
                if lbl in self.inverted_subset_dict and is_difficult[idx] == 0:
                    found = True
                    break

            if found:
                filtered_ids.append(image_id)

        return filtered_ids

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.root, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_ind[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):

            # filter instances
            class_name = obj.find("name").text.lower().strip()

            if not self.keep_all_classes:
                if class_name not in self.subset_class_names:
                    continue

            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            if not self.keep_all_classes:
                gt_classes.append(self.subset_class_dict[class_name])
            else:
                gt_classes.append(self.class_to_ind[class_name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        if self.keep_all_classes:
            return PascalVOCModelDataset.CLASSES[class_id]
        else:
            return self.subset_class_names[class_id]

    def set_keep_difficult(self,difficult):
        self.keep_difficult = difficult

    def set_transforms(self, transforms):
        self.transforms = transforms
