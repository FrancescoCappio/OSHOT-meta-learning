from torch.utils.data import Dataset


class DetectionSubset(Dataset):
    """
        Subset of a dataset at specified indices, exposes an extra
    method for querying the sizes of the image and another one to obtain the target

        Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
        """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        if hasattr(dataset, "map_class_id_to_class_name"):
            self.map_class_id_to_class_name = dataset.map_class_id_to_class_name
        if hasattr(dataset, "categories"):
            self.categories = dataset.categories

    def __getitem__(self, idx):
        image, target, index = self.dataset[self.indices[idx]]
        return image, target, idx

    def __len__(self):
        return len(self.indices)

    def get_img_info(self, idx):
        return self.dataset.get_img_info(self.indices[idx])

    def get_groundtruth(self, idx):
        return self.dataset.get_groundtruth(self.indices[idx])

    def set_keep_difficult(self,difficult):
        self.dataset.set_keep_difficult(difficult)

    def set_transforms(self, transforms):
        self.dataset.set_transforms(transforms)
