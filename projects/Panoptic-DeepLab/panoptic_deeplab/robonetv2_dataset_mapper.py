# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Callable, List, Union
import torch
from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import h5py
import random
import pdb
from PIL import Image
import io, os
import cv2
import pandas as pd
from pathlib import Path
from detectron2.data.datasets.tdw_playroom import _collate_playroom_metadata
import matplotlib.pyplot as plt
from .target_generator import PanopticDeepLabTargetGenerator
# from .utils import _object_id_hash, delta_image, optical_flow, visualize_views
# from .custom_transform import RandomResizedCrop
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import imageio
from torchvision.io import read_image
import shutil

__all__ = ["RoboNetDatasetMapper"]


class RoboNetV2DatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        training: bool,
        frame_idx: List,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
        """
        # fmt: off
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.training = training

        ## determine randomization
        self.seed = 0
        self.rng = np.random.RandomState(seed=self.seed)

        self.start_frame = 9
        self.num_camera_views = 1

        self.frame_idx = frame_idx

    @classmethod
    def from_config(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomFlip())

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "frame_idx": cfg.INPUT.FRAME_IDX,

        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        try:

            file_name = dataset_dict['file_name']
            movie = {}

            if not 'annotations' in file_name and not 'bridge_gt' in file_name:
                # Choose trajectory
                trajs = sorted(os.listdir(file_name))
                movie['traj'] = random.choice(trajs) if self.training else trajs[0]
                if not self.training:
                    if 'put_sweet_potato_in_pot_which_is_in_sink_distractors' in file_name or 'put_banana_on_place' in file_name:
                        movie['traj'] = 'traj16'

                # Choose camera view
                file_name = os.path.join(file_name, movie['traj'])
                cameras = [i for i in sorted(os.listdir(file_name)) if 'images' in i and 'depth' not in i]
                movie['camera'] = random.choice(cameras) if self.training else cameras[0]
                if not self.training:
                    if 'put_sweet_potato_in_pot_which_is_in_sink_distractors' in file_name or 'put_banana_on_place' in file_name:
                        movie['camera'] = 'images1'
                    if 'upenn' in file_name:
                        movie['camera'] = 'images2'

                # Choose frame indices
                file_name = os.path.join(file_name, movie['camera'])


                if not self.training:
                    frame_idx = self.start_frame
                else:
                    frame_idx = dataset_dict.get('frame', None)
                    if frame_idx is None:
                        assert isinstance(self.frame_idx, list), self.frame_idx
                        if len(self.frame_idx) > 0:
                            if len(self.frame_idx) == 1:
                                frame_idx = self.frame_idx[0]
                            elif len(self.frame_idx) == 2:
                                frame_idx = self.rng.randint(self.frame_idx[0], self.frame_idx[1] + 1)

                        else:
                            num_frames = len(sorted(os.listdir(file_name)))
                            frame_idx = self.rng.randint(self.start_frame, num_frames - 1)

            else:
                segment_color = read_image(file_name)
                movie['segment_id_map'] = self.process_segmentation_color(segment_color)[1]

                image_folder = '/data2/honglinc/robonetv2/toykitchen_fixed_cam/berkeley/'

                if 'test' in file_name:
                    file_name = file_name.replace('./bridge_annotations/test/', image_folder).replace('+','/').replace('_all_objects.png', '')
                elif 'val' in file_name:
                    file_name = file_name.replace('./bridge_annotations/val/', image_folder).replace('+', '/').replace('_all_objects.png', '')
                elif 'bridge_gt' in file_name:
                    file_name = file_name.replace('./bridge_gt/', image_folder).replace('+', '/').replace(
                        '_all_objects.png', '')
                file_name, frame_idx = file_name.split('im_')
                frame_idx = int(frame_idx)

            movie['frame_id'] = torch.tensor(frame_idx).view(1)


            # src = os.path.join(file_name, 'im_%d.jpg' % movie['frame_id'])
            # dst = src.split('/data2/honglinc/robonetv2/toykitchen_fixed_cam/berkeley/')[-1]
            # dst = dst.replace('/', '+')
            # dst = os.path.join('./bridge_test', dst)
            # print('Copy images', dst)
            # shutil.copyfile(src, dst)

            movie['image'] = read_image(os.path.join(file_name, 'im_%d.jpg' % movie['frame_id']))
            movie['image_1'] = read_image(os.path.join(file_name, 'im_%d.jpg' % (movie['frame_id'] + 1)))
            movie['frames'] = torch.cat([movie['image'].unsqueeze(0), movie['image_1'].unsqueeze(0)], 0)
            movie['width'], movie['height'] = movie['frames'].shape[-2], movie['frames'].shape[-1]

            dataset_dict.update(movie)

            # pdb.set_trace()
            # visualize_views(dataset_dict, plot_keys=['image', 'image_1'])
            return dataset_dict
        except Exception as e:
            print('Encoutering the following error when loading', file_name)
            print(e)
            return None

    def process_segmentation_color(self, seg_color):
        # convert segmentation color to integer segment id
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        return raw_segment_map, segment_map

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out