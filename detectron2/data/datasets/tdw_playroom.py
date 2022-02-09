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

from pathlib import Path
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog


def load_tdw_playroom(root, folder_name, split, file_pattern):
    dataset_dicts = []

    for i in split:
        if folder_name in ['playroom_large_v1', 'playroom_large_v1_main']:
            file_list = glob.glob(os.path.join(root, folder_name, 'model_split_%s' % str(i), file_pattern))
        elif folder_name in ['playroom_large_v1_main_images', 'playroom_large_v3_images']:
            file_list = glob.glob(os.path.join(root, folder_name, 'images', 'model_split_%s' % str(i), file_pattern))
        elif folder_name == 'playroom_large_v1_main_images_mat':
            file_list = glob.glob(os.path.join(root, 'playroom_large_v1_main_images', 'images', 'material_split_%s' % str(i), file_pattern))
        elif 'playroom_large_v1_main_images_mat_' in folder_name:
            file_list = glob.glob(os.path.join(root, 'playroom_large_v1_main_images', 'images', '*_split_%s' % str(i), file_pattern))
        else:
            raise ValueError
        # file_list = glob.glob(os.path.join('/data5/dbear/tdw_datasets/cylinder_miss_contain_tdwroom', file_pattern))
        # file_list = glob.glob(os.path.join('/data5/dbear/tdw_datasets/playroom_simple_v7safari', file_pattern))
        # file_list = glob.glob(os.path.join('/mnt/fs4/dbear/tdw_datasets/relational_v2/contain_from_basket_18inx18inx12iin_wood_mesh_to_b04_clownfish_with_box_tapered_beech/', '*.hdf5'))
        # file_list = glob.glob(os.path.join('/data5/dbear/tdw_datasets/playroom_occlude1', '*.hdf5'))
        dataset_dicts += [{'file_name': filename, 'width': 512, 'height': 512, 'root': root} for filename in file_list]

    return dataset_dicts


def register_tdw_playroom_instances(root, folder_name, split, file_pattern='*.hdf5'):
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
    assert isinstance(folder_name, str), folder_name
    assert isinstance(file_pattern, str), file_pattern
    # 1. register a function which returns dicts
    original_folder_name = folder_name
    if '/train' in folder_name:
        folder_name = folder_name.split('/train')[0]
    elif '/val' in folder_name:
        folder_name = folder_name.split('/val')[0]
    elif '/eval' in folder_name:
        folder_name = folder_name.split('/eval')[0]
    elif '/test' in folder_name:
        folder_name = folder_name.split('/test')[0]
    elif '/single_train' in folder_name:
        folder_name = folder_name.split('/single_train')[0]
    elif '/single_val' in folder_name:
        folder_name = folder_name.split('/single_val')[0]
    elif '/sup_train' in folder_name:
        folder_name = folder_name.split('/sup_train')[0]
    elif '/sup_val' in folder_name:
        folder_name = folder_name.split('/sup_val')[0]
    elif '/vis_val' in folder_name:
        folder_name = folder_name.split('/vis_val')[0]
    else:
        raise ValueError
    DatasetCatalog.register(original_folder_name,
                            lambda: load_tdw_playroom(root, folder_name, split, file_pattern))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    meta_path = os.path.join(root, folder_name, 'meta.json')
    metadict = json.loads(Path(meta_path).open().read())

    MetadataCatalog.get(folder_name).set(metadict=metadict)


def _collate_playroom_metadata(dataset_names):
    assert isinstance(dataset_names, (list, tuple))

    for dataset_name in dataset_names:
        meta = MetadataCatalog.get(dataset_name)
        pdb.set_trace()



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