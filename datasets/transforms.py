import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

import imgaug.augmenters as iaa
import imgaug.augmentables.segmaps as ias
import torch


class Transformer:
    def __init__(self, opt, subset='train'):
        self.opt = opt
        self.subset = subset

    def get_torch_transformer(self, normalize=False, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [transforms.ToTensor()]
        if normalize:
            # transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
            #                                         (0.5, 0.5, 0.5))]
            # Grey scale case
            transform_list += [transforms.Normalize((0.5, ),
                                                    (0.5, ))]
        return transforms.Compose(transform_list)


    def get_imgaug_transforms(self):
        transform_list = []
        seq = iaa.Sequential([], random_order=False)
        # Data augmentation should be applied on training data only
        if self.subset == 'train':
            if self.opt.flip_prob is not None:
                # print("flip")
                transform_list.append(iaa.Fliplr(opt.flip_prob))

            if self.opt.crop is not None:
                # print("crop")
                transform_list.append(iaa.Crop(percent=(0, self.opt.crop)))

            if self.opt.gauss_noise:
                # print("gauss noise")
                transform_list.append(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0))

            if self.opt.aug_scale is not None:
                # print("scale")
                min_scale, max_scale = 1 - self.opt.aug_scale, 1 + self.opt.aug_scale
                transform_list.append(iaa.Affine(scale={"x": (min_scale, max_scale), "y": (min_scale, max_scale)}))

            if self.opt.aug_angle is not None:
                # print("rotate")
                transform_list.append(iaa.Affine(rotate=(-self.opt.aug_angle, self.opt.aug_angle)))

            if transform_list:  # transform list is not empty
                seq.append(iaa.SomeOf((0, None), transform_list))
        # Cuz resize has to be deterministic, not applied with prob inside SomeOf
        if self.opt.resize:
            # print("resize")
            seq.append(iaa.Resize({"height": self.opt.load_size, "width": self.opt.load_size}))
        # print("===== transformer length", len(seq))
        # print(seq)
        return seq


    def transform_slice_mask_imgaug(self, transformer, image, label):
        image, label = np.asarray(image), np.asarray(label).astype(np.int16)
        segmap = ias.SegmentationMapsOnImage(label, shape=image.shape)
        img_aug, seg_aug = transformer(image=image, segmentation_maps=segmap)

        # img_aug is returned as numpy.ndarray, seg_aug is returned as seg object from iaa
        # with shape (W, H, 1), so have to convert it to arr and squeeze it
        img_aug, seg_aug = img_aug, np.squeeze(seg_aug.arr)

        return img_aug, seg_aug

    def __call__(self, img, label ):
        imgaug_transformer = self.get_imgaug_transforms()
        label_torch_transformer = self.get_torch_transformer(normalize=False)
        img_torch_transformer = self.get_torch_transformer(normalize=True)
        if imgaug_transformer is not None:
            img, label = self.transform_slice_mask_imgaug(imgaug_transformer, img, label)

        label_tensor = label_torch_transformer(np.ascontiguousarray(label)).type(torch.FloatTensor)
        image_tensor = img_torch_transformer(np.ascontiguousarray(img)).type(torch.FloatTensor)

        # input = torch.tensor(input).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
        # if self.opt.resize:
        #     size = [self.opt.load_size,self.opt.load_size]
        #     input = F.interpolate(input, size=size) # method nearest
        #
        # input = input.squeeze(1)
        # print(torch.unique(input))
        # print(input.size(), type(input))
        # x, y = self.transform_slice_mask_imgaug(imgaug_transformer, x, y)
        # input = torch_transformer(input)
        # y = torch_transformer(y)
        # print("after transformation")
        # print(x.shape, y.shape)
        return image_tensor, label_tensor


def process_item(x):
    x = torch.tensor(x).squeeze()
    x = x.view(1, x.shape[0], x.shape[1]).permute([0, 1, 2]).float()
    return x