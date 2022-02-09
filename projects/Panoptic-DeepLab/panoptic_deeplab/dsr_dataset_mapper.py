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
# from detectron2.data.datasets.tdw_playroom import _collate_playroom_metadata
import matplotlib.pyplot as plt
from .target_generator import PanopticDeepLabTargetGenerator
# from .utils import _object_id_hash, delta_image, optical_flow, visualize_views
# from .custom_transform import RandomResizedCrop
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import imageio

__all__ = ["DSRDatasetMapper"]


class DSRDatasetMapper:
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
            panoptic_target_generator: Callable,
            training: bool,
            meta: List,
            view_generator: Callable,
            data_augmentation: Callable,
            delta_image_sup: bool,
            dual_delta_images: bool,
            # frame_idx: List,
            # delta_image_threshold: float,
            # fix_single: bool
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
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.tot_seq_len = 10
        self.seq_len = 10
        self.volume_size = [128, 128, 48]
        self.direction_num = 8
        self.voxel_size = 0.004

        self.returns = ['action', 'color_heightmap', 'color_image', 'tsdf', 'mask_3d', 'scene_flow_3d']
        self.data_per_seq = self.tot_seq_len // self.seq_len
        self.center_crop_fn = transforms.CenterCrop(size=240)
        # self.resize_fn = transforms.Resize(size=512)

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

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN

        meta = None
        thing_ids = [0, 1]
        ignore_label = 255
        assert ignore_label not in thing_ids
        panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=ignore_label,
            thing_ids=thing_ids,
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        view_generator = None
        data_augmentation = None
        delta_image_sup = False
        dual_delta_images = False

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
            "meta": meta,
            "view_generator": view_generator,
            "data_augmentation": data_augmentation,
            "delta_image_sup": delta_image_sup,
            # "delta_image_threshold": cfg.MODEL.GRAPH_INFERENCE_HEAD.DELTA_IMAGE_THRESHOLD,
            "dual_delta_images": dual_delta_images,
            # "frame_idx": cfg.MODEL.GRAPH_INFERENCE_HEAD.FRAME_IDX,  # (for debugging only)
            # "fix_single": cfg.MODEL.PANOPTIC_DEEPLAB.FIX_SINGLE
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        data_path = dataset_dict['file_name']
        step_id = int(data_path.split('_')[-1].split('.hdf5')[0])
        next_step_id = step_id + 1
        # image
        f = h5py.File(data_path, 'r')
        dataset_dict['image'] = torch.tensor(f['color_image_small']).permute(2, 0, 1)
        dataset_dict['image'] = self.center_crop_resize(dataset_dict['image'])

        # image_1
        f = h5py.File(data_path.split('%d.hdf5' % step_id)[0] + '%d.hdf5' % next_step_id, 'r')
        dataset_dict['image_1'] = torch.tensor(f['color_image_small']).permute(2, 0, 1)
        dataset_dict['image_1'] = self.center_crop_resize(dataset_dict['image_1'])
        dataset_dict['frames'] = torch.cat([dataset_dict['image'][None], dataset_dict['image_1'][None]], 0)

        return dataset_dict

    def center_crop_resize(self, image):
        out = self.center_crop_fn(image)
        # out = self.resize_fn(out)
        return out
