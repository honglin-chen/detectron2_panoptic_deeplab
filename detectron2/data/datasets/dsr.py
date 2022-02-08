import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
import glob
import pdb
import pandas as pd
from pathlib import Path
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog


def load_dsr(root, folder_name, file_pattern, dataset_name):
    dataset_dicts = []

    file_list = glob.glob(os.path.join(root, folder_name, file_pattern))

    new_file_list = []
    for file in file_list:
        idx = int(file.split('/')[-1].split('_')[0])
        if dataset_name == 'dsr/train':
            if idx >= 5:
                new_file_list.append(file)
        elif dataset_name == 'dsr/eval':
            if idx > 5 and idx < 10:
                new_file_list.append(file)
        elif dataset_name == 'dsr/val':
            if idx < 5:
                new_file_list.append(file)
        else:
            raise ValueError

    print(dataset_name, len(new_file_list))
    file_list = new_file_list

    dataset_dicts += [{'file_name': filename, 'root': root, 'width': 240, 'height': 240,} for filename in file_list]


    return dataset_dicts


def register_dsr(root, dataset_name, file_pattern):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(root, str), root

    folder_name = 'dsr_dataset/real_test_data'

    training = 'train' in dataset_name
    DatasetCatalog.register(dataset_name,
                            lambda: load_dsr(root, folder_name=folder_name, file_pattern=file_pattern, dataset_name=dataset_name))

    # # 2. Optionally, add metadata about this dataset,
    # # since they might be useful in evaluation, visualization or logging
    # # meta_path = os.path.join(root, folder_name, 'meta_data.pkl')
    # meta_path = '/data1/honglinc/dsr/hdf5/meta_data.pkl'
    # if not os.path.exists(meta_path):
    #     raise FileNotFoundError("You need to generate the meta_data.pkl file for this dataset")
    #
    # metadict = pd.read_pickle(meta_path, compression='gzip')
    # print('metadict: ', metadict)
    # MetadataCatalog.get('dsr').set(metadict=metadict)


if __name__ == "__main__":
    """
    Test the TDW playroom dataset loader.

    Usage:
        python -m detectron2.data.datasets.tdw_playroom
    """
    # from detectron2.utils.logger import setup_logger
    # from detectron2.utils.visualizer import Visualizer
    # import detectron2.data.datasets  # noqa # add pre-defined metadata
    # import sys

    folder_name = '/mnt/fs5/dbear/tdw_datasets/playroom_large_v1/model_split_0'
    register_tdw_playroom_instances(folder_name)