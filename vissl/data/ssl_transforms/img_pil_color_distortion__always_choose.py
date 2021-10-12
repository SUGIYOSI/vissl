# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict
import random
from collections.abc import Sequence

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform

# MARK: RandomChoiceを拡張
# https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomChoice
class RandomTransforms:
    """Base class for a list of transformations with randomness

    Args:
        transforms (sequence): list of transformations
    """

    def __init__(self, transforms, weights=None):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms
        self.weights = weights

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomChoice_With_Weights(RandomTransforms):
    def __call__(self, img):
        t = random.choices(self.transforms, weights=self.weights)[0]
        return t(img)

@register_transform("ImgPilColorDistortion_AlwaysChoose")
class ImgPilColorDistortion_AlwaysChoose(ClassyTransform):
    """
    Apply Random color distortions to the input image.
    There are multiple different ways of applying these distortions.
    This implementation follows SimCLR - https://arxiv.org/abs/2002.05709
    It randomly distorts the hue, saturation, brightness of an image and can
    randomly convert the image to grayscale.
    """

    def __init__(self, strength):
        """
        Args:
            strength (float): A number used to quantify the strength of the
                              color distortion.
        """
        self.strength = strength
        self.color_jitter = pth_transforms.ColorJitter(
            0.8 * self.strength,
            0.8 * self.strength,
            0.8 * self.strength,
            0.2 * self.strength,
        )
        self.gray = pth_transforms.RandomGrayscale(p=1.0)
        self.transforms = RandomChoice_With_Weights(
            transforms=[self.color_jitter, self.gray],
            weights=[0.8, 0.2])

    def __call__(self, image):
        return self.transforms(image)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilColorDistortion_AlwaysChoose":
        """
        Instantiates ImgPilColorDistortion_AlwaysChoose from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilColorDistortion_AlwaysChoose instance.
        """
        strength = config.get("strength", 1.0)
        logging.info(f"ImgPilColorDistortion_AlwaysChoose | Using strength: {strength}")
        return cls(strength=strength)
