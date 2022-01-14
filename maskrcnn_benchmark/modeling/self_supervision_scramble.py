import torch
import os
import numpy as np
from ..data.transforms.transforms import DeNormalize, Normalize

from torchvision.transforms import functional as F

from torchvision import transforms

class ToTensor(object):
    def __call__(self, image):
        img = F.to_tensor(image)
        return img

class To255(object):
    def __call__(self, image):
        return image*255

class ToRGB(object):
    def __call__(self, image):
        return image[[2, 1, 0]]

class ToPILImage(object):
    def __init__(self):
        self.to_pil_transform = transforms.ToPILImage()
    def __call__(self, image):
        # should pass from bgr to rgb
        image = image[[2, 1, 0]]
        # should pass from 0-255 range to 0-1 range
        image = image/255
        return self.to_pil_transform(image)


class SelfSup_Scrambler():
    def __init__(self, cfg, device):
        super(SelfSup_Scrambler, self).__init__()
        self.grid_size = 3
        self.jig_classes = cfg.MODEL.SELF_SUPERVISOR.CLASSES
        self.permutations = self.__retrieve_permutations(self.jig_classes, cfg.MODEL.SELF_SUPERVISOR.PATH)
        self.reverse = self.__compute_reverse_permutations(self.permutations)
        self.orders = None
        self.val_orders = None
        self.device = torch.device(device)

        self.tile_transformer = transforms.Compose([DeNormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD, reshape=True),
                                                    ToPILImage(),
                                                    transforms.RandomGrayscale(0.1),
                                                    ToTensor(),
                                                    Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)]
                                                   )

    def __retrieve_permutations(self, classes, permutations_path):
        all_perm = np.load(os.path.join(permutations_path, 'permutations_%d.npy' % (classes)))
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def __compute_reverse_permutations(self, permutations):
        reverse = np.zeros(permutations.shape, dtype=int)
        for i in range(permutations.shape[0]):
            for j in range(permutations.shape[1]):
                reverse[i, permutations[i, j]] = j
        return reverse

    def get_tile(self, img, n):
        h = float(img.size()[1]) / self.grid_size
        w = float(img.size()[2]) / self.grid_size

        y = int(n / self.grid_size)
        x = n % self.grid_size

        tile = torch.zeros([img.size()[0], int(h), int(w)], dtype=torch.float).to(self.device)

        top = int(y * h)
        left = int(x * w)
        tile[:, :, :] = img[0:tile.size()[0], top:int(top + h), left:int(left + w)]

        return tile

    def _compute_val_orders(self):
        self.val_orders = torch.Tensor(range(len(self.permutations) + 1)).type(torch.long)

    def _compute_orders(self, size, bias=1.0):
        orders = torch.randint(0, len(self.permutations) + 1, (size,))
        probabilities = torch.rand(size) > bias
        self.orders = orders * probabilities.long()
        self.labels = self.orders

    @staticmethod
    def rotate_single(x, rotation=-1):
        t = torch.zeros(x.size(), dtype=torch.float32)

        if rotation == -1:
            rotation = torch.randint(4, ())

        if rotation == 1:
            t = torch.transpose(x, 1, 2)  # 90 degree
        elif rotation == 2:
            t = torch.flip(x, [1, 2])  # 180 degree
        elif rotation == 3:
            t = torch.flip(torch.transpose(x, 1, 2), [1, 2])  # 270
        else:
            t[:, :, :] = x

        return t, rotation

    def shuffle_single(self, x):
        t = torch.zeros(x.size(), dtype=torch.float32)
        n_grids = self.grid_size ** 2

        step_x = x.size(2) // self.grid_size
        step_y = x.size(1) // self.grid_size

        order = torch.randint(len(self.permutations) + 1, ())

        if order:
            for ii in range(n_grids):
                t_start_x = (ii % self.grid_size) * step_x
                t_start_y = (ii // self.grid_size) * step_y

                s_start_x = (self.permutations[order - 1][ii] % self.grid_size) * step_x
                s_start_y = (self.permutations[order - 1][ii] // self.grid_size) * step_y

                tile = x[:, s_start_y:(s_start_y + step_y), s_start_x:(s_start_x + step_x)]
                tile = self.tile_transformer(tile)

                t[:, t_start_y:(t_start_y + step_y), t_start_x:(t_start_x + step_x)] = tile
        else:
            for ii in range(n_grids):
                t_start_x = (ii % self.grid_size) * step_x
                t_start_y = (ii // self.grid_size) * step_y

                tile = x[:, t_start_y:(t_start_y + step_y), t_start_x:(t_start_x + step_x)]
                tile = self.tile_transformer(tile)

                t[:, t_start_y:(t_start_y + step_y), t_start_x:(t_start_x + step_x)] = tile
        return t, order

    def slice_scramble(self, x, device=None):
        device = self.device if device is None else device
        t_images = torch.zeros(x.size(), dtype=torch.float32, device=device)

        n_grids = self.grid_size ** 2

        step_x = x.size(3) // self.grid_size
        step_y = x.size(2) // self.grid_size

        for i in range(len(x)):

            order = self.orders[i]
            if order:
                for ii in range(n_grids):
                    t_start_x = (ii % self.grid_size) * step_x
                    t_start_y = (ii // self.grid_size) * step_y

                    s_start_x = (self.permutations[order - 1][ii] % self.grid_size) * step_x
                    s_start_y = (self.permutations[order - 1][ii] // self.grid_size) * step_y

                    tile = x[i, :, s_start_y:(s_start_y + step_y), s_start_x:(s_start_x + step_x)]
                    tile = self.tile_transformer(tile)

                    t_images[i, :, t_start_y:(t_start_y + step_y), t_start_x:(t_start_x + step_x)] = tile
            else:
                for ii in range(n_grids):
                    t_start_x = (ii % self.grid_size) * step_x
                    t_start_y = (ii // self.grid_size) * step_y

                    tile = x[i, :, t_start_y:(t_start_y + step_y), t_start_x:(t_start_x + step_x)]
                    tile = self.tile_transformer(tile)

                    t_images[i, :, t_start_y:(t_start_y + step_y), t_start_x:(t_start_x + step_x)] = tile

        return t_images  # , orders

    def slice_descramble(self, image, order):
        descrambled_image = torch.zeros(image.size(), dtype=torch.float32)

        n_grids = self.grid_size ** 2

        step_x = image.size(2) // self.grid_size
        step_y = image.size(1) // self.grid_size

        if order:
            for ii in range(n_grids):
                t_start_x = (ii % self.grid_size) * step_x
                t_start_y = (ii // self.grid_size) * step_y

                s_start_x = (self.reverse[order - 1][ii] % self.grid_size) * step_x
                s_start_y = (self.reverse[order - 1][ii] // self.grid_size) * step_y

                descrambled_image[:, t_start_y:(t_start_y + step_y), t_start_x:(t_start_x + step_x)] = image[:,
                                                                                                       s_start_y:(
                                                                                                                   s_start_y + step_y),
                                                                                                       s_start_x:(
                                                                                                                   s_start_x + step_x)]
        else:
            descrambled_image = image
        return descrambled_image
