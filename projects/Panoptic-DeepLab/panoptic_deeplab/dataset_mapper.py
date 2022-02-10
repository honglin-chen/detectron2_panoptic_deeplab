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

from detectron2.data.datasets.tdw_playroom import _collate_playroom_metadata
import matplotlib.pyplot as plt
from .target_generator import PanopticDeepLabTargetGenerator
from .utils import _object_id_hash, delta_image, optical_flow, visualize_views, sample_image_inds_from_probs
# from .custom_transform import RandomResizedCrop
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
# from detectron2.projects.spatial_temporal.models.dual_stream import CropFromDistribution
__all__ = ["PanopticDeeplabDatasetMapper"]


class PanopticDeeplabDatasetMapper:
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
            # delta_image_sup: bool,
            # dual_delta_images: bool,
            frame_idx: int,
            # delta_image_threshold: float,
            # fix_single: bool,
            # thing_sup: bool,
            # raft_flow: bool,
            # motion_crop_func: Callable,
            # motion_crop_train: bool,
            # motion_crop_test: bool,
            # delta_time: int,
            # apply_motion_filter: bool,
            # single_image: bool
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

        self.panoptic_target_generator = panoptic_target_generator
        self.training = training
        try:
            self.meta = meta.metadict
        except:
            self.meta = None
        self.delta_image_sup = False
        self.view_generator = view_generator
        self.data_augmentation = data_augmentation
        # self.delta_image_sup = delta_image_sup
        self.set_dummy_objects = False
        self.frame_idx = frame_idx
        # self.delta_image_threshold = delta_image_threshold
        # self.dual_delta_images = dual_delta_images
        self.fix_single = False #fix_single
        self.thing_sup = False #thing_sup
        # self.raft_flow = raft_flow
        # self.motion_crop_func = motion_crop_func
        self.motion_crop_train = False
        self.motion_crop_test = False
        self.delta_time = 1

        # self.motion_thresh = 0.03
        # self.motion_area_thresh = 0.05
        # self.apply_motion_filter = apply_motion_filter
        self.single_image = False #single_image


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

        if 'playroom' in dataset_names[0]:

            meta = MetadataCatalog.get(dataset_names[0].split('/')[0])
            thing_ids = [0,1]
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
            # if cfg.MODEL.GRAPH_INFERENCE_HEAD.MULTI_VIEW:
            #     view_generator = ViewGenerator(size=cfg.INPUT.CROP.SIZE,
            #                                    min_crop=cfg.MODEL.GRAPH_INFERENCE_HEAD.MIN_CROP,
            #                                    color_transform=cfg.MODEL.GRAPH_INFERENCE_HEAD.COLOR_TRANSFORM)
            # else:
            #     view_generator = None

            # if cfg.MODEL.GRAPH_INFERENCE_HEAD.DATA_AUGMENTATION:
            #     data_augmentation = DataAugmentation(size=cfg.INPUT.CROP.SIZE,
            #                                          min_crop=cfg.MODEL.GRAPH_INFERENCE_HEAD.MIN_CROP)
            # else:
            #     data_augmentation = None
        else:
            meta = MetadataCatalog.get(dataset_names[0])
            panoptic_target_generator = PanopticDeepLabTargetGenerator(
                ignore_label=meta.ignore_label,
                thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
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
        raft_flow = False

        motion_crop_func = None

        # if cfg.INPUT.MOTION_CROP_TRAIN or cfg.INPUT.MOTION_CROP_TEST:
        #     size = [cfg.INPUT.MOTION_CROP_SIZE] * 2
        #     motion_crop_func = CropFromDistribution(size)

        # for obj in cfg.MODEL.GRAPH_INFERENCE_HEAD.OBJ_SUP:
        #     if "delta_image" in obj:
        #         delta_image_sup = True
        #     if "dual_delta_images" in obj:
        #         dual_delta_images = True
        #     if "raft_flow" in obj:
        #         raft_flow = True

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
            "meta": meta,
            "view_generator": None,
            "data_augmentation": None,
            # "delta_image_sup": delta_image_sup,
            # "delta_image_threshold": cfg.MODEL.GRAPH_INFERENCE_HEAD.DELTA_IMAGE_THRESHOLD,
            # "dual_delta_images": dual_delta_images,
            "frame_idx": cfg.INPUT.FRAME_IDX,  # (for debugging only)
            # "fix_single": cfg.MODEL.PANOPTIC_DEEPLAB.FIX_SINGLE,
            # "thing_sup": cfg.MODEL.PANOPTIC_DEEPLAB.THING_SUP,
            # "raft_flow": raft_flow,
            # "motion_crop_func": motion_crop_func,
            # "motion_crop_train": False, #cfg.INPUT.MOTION_CROP_TRAIN,
            # "motion_crop_test": False, #cfg.INPUT.MOTION_CROP_TEST,
            # "apply_motion_filter": False, #cfg.INPUT.APPLY_MOTION_FILTER,
            # "delta_time": 1, #cfg.INPUT.DELTA_TIME,
            # "single_image": False, #cfg.MODEL.ARGS.single_image
        }
        return ret

    def read_frame(self, path, frame_idx):
        image_path = os.path.join(path, format(frame_idx, '05d') + '.png')
        image = utils.read_image(image_path, format=self.image_format)
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return image

    def random_trial(self, root, folder_path):
        rand_trial = random.choice(range(0, 32))
        rand_split = random.choice(range(0, 1250))
        split_trial = os.path.join(f'model_split_{rand_trial}', format(rand_split, '04d'))
        print('New split trial: ', split_trial)
        dataset_dict = {
            'file_name': os.path.join(folder_path, 'images', split_trial),
            'root': root
        }
        return dataset_dict, split_trial

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        if not self.training and self.single_image:
            print('warning: using single image')
            return torch.load('./notebook/inputs.pt')[0]

        # different loading protocol for TDW playroom datase
        if 'playroom' in dataset_dict['file_name'] or 'tdw_datasets' in dataset_dict['file_name']:
            # parameter (todo: move to init)
            sequence_length = 1
            delta_time = self.delta_time # if self.training else 1
            trial_start = 5
            sources = ['images', 'objects', 'flow']
            key_map = {'objects': '_id', 'images': 'image', 'flow': 'flow'}

            assert sequence_length == 1, "not implemented for sequence_length > 1 as shown in the following FIXME"

            # file meta data
            root = dataset_dict['root']
            splits = dataset_dict["file_name"].split('/images/')
            folder_path, split_trial = splits[0], splits[1]
            num_frames = len(os.listdir(os.path.join(dataset_dict['file_name'])))

            valid = False
            invalid_count = 0
            while not valid:
                if invalid_count > 2:
                    dataset_dict, split_trial = self.random_trial(root, folder_path)
                    num_frames = len(os.listdir(os.path.join(dataset_dict['file_name'])))
                    print('New dataset', dataset_dict['file_name'])
                    invalid_count = 0
                invalid_count += 1
                # sample indices, ignore the first `trial_start` frames of the video
                if num_frames - sequence_length * delta_time - 1 <= trial_start:  # invalid video input
                    start_idx = 0
                else:
                    if self.training:
                        assert isinstance(self.frame_idx, list), self.frame_idx
                        if len(self.frame_idx) > 0:
                            if len(self.frame_idx) == 1:
                                start_idx = self.frame_idx[0]
                            elif len(self.frame_idx) == 2:
                                start_idx = random.choice(range(self.frame_idx[0], self.frame_idx[1] + 1))
                            # print('Frame idx choice: ', start_idx)
                        else:
                            print('Warning: training with all frames')
                            start_idx = random.choice(range(trial_start, num_frames - sequence_length * delta_time - 1))
                    else:
                        start_idx = trial_start  # int(num_frames // 2)

                sequence_idx = [start_idx + i * delta_time for i in range(sequence_length)]
                dataset_dict["frame_id"] = sequence_idx

                for source in sources:
                    if source == 'flow' and not os.path.isdir(dataset_dict['file_name'].replace('/images/', '/flow/')):
                        continue

                    data_path = os.path.join(folder_path, source, split_trial)
                    for n in sequence_idx:
                        dataset_dict[key_map[source]] = self.read_frame(data_path, frame_idx=n)

                        if source == 'images':
                            dataset_dict['image_1'] = self.read_frame(data_path, frame_idx=n+delta_time)
                            # image_pair = torch.cat([dataset_dict['image'][None], dataset_dict['image_1'][None]])
                            # delta_images, delta_images_floodfill = delta_image(image_pair.float() / 255.,                                                  thresh=self.delta_image_threshold)
                            # dataset_dict['delta_image'] = delta_images
                            # dataset_dict['delta_image_floodfill'] = delta_images_floodfill

                # convert segmentation color to integer object id
                segment_ids = _object_id_hash(dataset_dict['_id'], val=256, dtype=torch.long)
                segment_ids = segment_ids.squeeze(0)
                dataset_dict['segment_id_map'] = segment_ids.clone()
                _, segment_ids = torch.unique(segment_ids, return_inverse=True)
                segment_ids -= segment_ids.min()
                dataset_dict['objects'] = segment_ids
                dataset_dict['width'] = dataset_dict['height'] = 512

                frames = torch.cat([dataset_dict['image'].unsqueeze(0),
                                    dataset_dict['image_1'].unsqueeze(0)], 0)
                dataset_dict['frames'] = frames

                if (self.training and self.motion_crop_train) or (not self.training and self.motion_crop_test):
                    assert self.motion_crop_func is not None
                    data = {
                        'frames': frames.unsqueeze(0),
                        'segment_id_map': dataset_dict['segment_id_map'][None,None,None].expand(-1, 2, -1, -1, -1),
                        'delta_image': dataset_dict['delta_image'][None, None].expand(-1, 2, -1, -1, -1),
                        'delta_image_floodfill': dataset_dict['delta_image_floodfill'][None, None].expand(-1, 2, -1, -1, -1)
                    }
                    if 'flow' in dataset_dict.keys():
                        data['flow'] = dataset_dict['flow'][None, None].expand(-1, 2, -1, -1, -1)
                    data = self.motion_crop_func(data, data['delta_image_floodfill'], training=self.training)

                    data['frames'] = data['frames'][0]
                    data['segment_id_map'] = data['segment_id_map'][0, 0, 0]
                    data['delta_image'] = data['delta_image'][0, 0]
                    data['delta_image_floodfill'] = data['delta_image_floodfill'][0, 0]
                    if 'flow' in dataset_dict.keys():
                        data['flow'] = data['flow'][0, 0]
                    _, segment_ids = torch.unique(data['segment_id_map'], return_inverse=True)
                    segment_ids -= segment_ids.min()
                    data['objects'] = segment_ids
                    dataset_dict.update(data)
                else:
                    dataset_dict['frames'] = frames

                valid = True
                # valid = self.motion_filter(dataset_dict['delta_image']) if self.apply_motion_filter else True
                # print('valid: ', valid, data_path, start_idx)
                # if not valid:
                #     print('Warning: image is not valid with motion filter')
                #     visualize_views(dataset_dict, plot_keys=['image', 'objects', 'gt_moving', 'delta_image', 'frames', 'flow'])

            dataset_dict["file_name"] = os.path.join(root, 'playroom_large_v3_images', split_trial + '.hdf5')
            if 'model_split' in dataset_dict['file_name'] or 'material_split' in dataset_dict['file_name'] and \
                    'val' not in dataset_dict['file_name']:

                if 'material_split' in dataset_dict['file_name']:
                    key = dataset_dict['file_name'].split(dataset_dict['root'] + '/playroom_large_v1/')[-1]
                    per_segment_id = self.meta['primitives_large_v1/' + key]
                else:
                    per_segment_id = self.meta[dataset_dict['file_name'].split(dataset_dict['root'] + '/')[-1]]

                if isinstance(per_segment_id, dict):
                    obj_types = ['probe', 'target', 'distractor', 'occluder', 'zone', 'background', 'moving']
                    per_segment_id = [int(per_segment_id[k]) for k in obj_types if k in per_segment_id.keys()]
                else:
                    per_segment_id = [int(i) for i in per_segment_id]

                if len(per_segment_id) == 6:
                    per_segment_id_tensor = torch.as_tensor(np.ascontiguousarray(per_segment_id)).long()
                elif len(per_segment_id) == 7:  # the last item is the moving id
                    per_segment_id_tensor = torch.as_tensor(np.ascontiguousarray(per_segment_id[0:6])).long()
                    moving_id_tensor = torch.as_tensor(np.ascontiguousarray(per_segment_id[-1])).long()
                    dataset_dict['moving_id'] = moving_id_tensor.unsqueeze(0)
                    dataset_dict['gt_moving'] = dataset_dict['segment_id_map'] == dataset_dict['moving_id']
                    dataset_dict['gt_moving'] = dataset_dict['gt_moving'].unsqueeze(0)
                    assert per_segment_id_tensor[int(split_trial.split('/')[-1]) % 4] == moving_id_tensor, \
                        (split_trial, per_segment_id_tensor[int(split_trial.split('/')[-1]) % 4], moving_id_tensor)

                dataset_dict['per_segment_id'] = per_segment_id_tensor.unsqueeze(0)

            '''
            if self.view_generator is not None and self.training:
                views = self.view_generator(dataset_dict)
                dataset_dict.update(views)

            if self.data_augmentation is not None and self.training:
                dataset_dict.update(self.data_augmentation(dataset_dict))

            # panoptic-deeplab target generator
            if self.fix_single:
                trial_id = int(split_trial.split('/')[-1])
                obj_id = trial_id % 4
                segments_info = [
                    {'id': per_segment_id[i] if i == obj_id else -1,
                     'category_id': 1 if i == obj_id else -1, 'iscrowd': 0} for i in range(len(per_segment_id))]
                targets = self.panoptic_target_generator(dataset_dict['segment_id_map'], segments_info)
                targets['offset_weights'] = targets['center_weights']

                # targets['offset_weights'] = torch.ones_like(targets['offset_weights'])
            elif self.thing_sup:
                segments_info = [
                    {'id': per_segment_id[i] if i < 4 else -1,
                     'category_id': 1 if i < 4 else 0, 'iscrowd': 0} for i in range(6)]
                targets = self.panoptic_target_generator(dataset_dict['segment_id_map'], segments_info)
            else:
                segments_info = [{'id': i, 'category_id': 1 if i > 0 else 0, 'iscrowd': 0} for i in per_segment_id]
                targets = self.panoptic_target_generator(dataset_dict['segment_id_map'], segments_info)
            dataset_dict.update(targets)

            # if True: #self.training:
            # print('offset_weights', torch.unique(dataset_dict['offset_weights']))
            # visualize_views(dataset_dict, plot_keys=['image', 'objects', 'center', 'center_weights', 'offset', 'offset_weights'])
            # visualize_views(dataset_dict, plot_keys=['image', 'objects', 'gt_moving', 'delta_image', 'delta_image_floodfill', 'flow', 'flow_thresh'])
            '''

        else:
            raise ValueError
            # Load image.
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            utils.check_image_size(dataset_dict, image)
            # Panoptic label is encoded in RGB image.
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")

            # Reuses semantic transform for panoptic labels.
            aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
            _ = self.augmentations(aug_input)
            image, pan_seg_gt = aug_input.image, aug_input.sem_seg

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict['objects'] = torch.as_tensor(rgb2id(pan_seg_gt))

            # Generates training targets for Panoptic-DeepLab.
            targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), dataset_dict["segments_info"])
            dataset_dict.update(targets)

        return dataset_dict

    def motion_filter(self, delta_images):
        moving = (delta_images > self.motion_thresh).float()
        moving_fraction = moving.mean()
        motion_filter = moving_fraction > self.motion_area_thresh
        if not motion_filter:
            print('moving fraction: ', moving_fraction)
        return motion_filter


class DataAugmentation:
    def __init__(self, size, min_crop):
        self.size = size

        self.interpolation = Image.NEAREST

        self.color_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
        ])

        self.geometric_transform = RandomResizedCrop(
            size=self.size,
            scale=(min_crop, 1.),
            interpolation=self.interpolation
        )

    def __call__(self, dataset_dict,
                 transform_keys=['objects', 'segment_id_map', 'flow', 'delta_image', 'delta_image_floodfill']):
        output = dict()
        image = dataset_dict['image'].clone()
        _, H, W = image.shape

        # apply color and geometric transformations
        output['image'] = self.color_transform(image)
        output['image'], params = self.geometric_transform(output['image'])

        # apply the same geometric transform to transform_keys
        i, j, h, w = params
        output['params'] = torch.as_tensor([i / H, j / W, h / H, w / W]).unsqueeze(0)

        for key in transform_keys:
            if key in dataset_dict.keys():
                if len(dataset_dict[key].shape) == 2:
                    output[key] = F.resized_crop(dataset_dict[key][None], i, j, h, w, self.size, self.interpolation)
                    output[key] = output[key].squeeze(0)
                elif len(dataset_dict[key].shape) == 3:
                    output[key] = F.resized_crop(dataset_dict[key], i, j, h, w, self.size, self.interpolation)
                else:
                    raise ValueError

        return output


class ViewGenerator:
    def __init__(self, size, min_crop, color_transform):
        self.size = size

        self.interpolation = Image.NEAREST

        if color_transform:
            self.color_transform = transforms.Compose([
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
            ])
        else:
            self.color_transform = torch.nn.Identity()

        self.geometric_transform = RandomResizedCrop(
            size=self.size,
            scale=(min_crop, 1.),
            interpolation=self.interpolation
        )

    def __call__(self, dataset_dict,
                 transform_keys=['objects', 'segment_id_map', 'flow', 'delta_image', 'delta_image_floodfill']):
        assert 'image' in dataset_dict.keys(), "cannot find image in dataset_dict"
        views = dict()
        image = dataset_dict['image'].clone()
        _, H, W = image.shape

        # apply two different color transformation on the same image
        # print('warning: color transform disabled')

        views['image'] = self.color_transform(image)
        views['image_1'] = self.color_transform(image)

        # geometric transform is only applied to the second transformed image
        views['image_1'], params = self.geometric_transform(views['image_1'])

        # apply the same geometric transform to transform_keys
        i, j, h, w = params
        # print('dataset_mapper resize crop params: ', i, j, h, w, self.size)
        views['params'] = torch.as_tensor([i / H, j / W, h / H, w / W]).unsqueeze(0)

        for key in transform_keys:
            if key in dataset_dict.keys():
                if len(dataset_dict[key].shape) == 2:
                    views[key+'_1'] = F.resized_crop(dataset_dict[key][None], i, j, h, w, self.size, self.interpolation)
                    views[key+'_1'] = views[key+'_1'].squeeze(0)
                elif len(dataset_dict[key].shape) == 3:
                    views[key+'_1'] = F.resized_crop(dataset_dict[key], i, j, h, w, self.size, self.interpolation)
                else:
                    raise ValueError

        return views




