# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import warnings

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    """
        Dataset to concatenate multiple datasets.
        Purpose: useful to assemble different existing datasets, possibly
        large-scale datasets as the concatenation operation is done in an
        on-the-fly manner.

        Arguments:
            datasets (sequence): List of datasets to be concatenated
        """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        if hasattr(datasets[0], "categories"):
            self.categories = datasets[0].categories
        if hasattr(datasets[0], "map_class_id_to_class_name"):
            self.map_class_id_to_class_name = datasets[0].map_class_id_to_class_name

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        image, target, index = self.datasets[dataset_idx][sample_idx]
        return image, target, idx

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)

    def get_groundtruth(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_groundtruth(sample_idx)

    def set_keep_difficult(self,difficult):
        for ds in self.datasets:
            ds.set_keep_difficult(difficult)

    def set_transforms(self, transforms):
        for ds in self.datasets:
            ds.set_transforms(transforms)
