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
import json
from .. import DatasetCatalog, MetadataCatalog
import copy

def load_robonet(root, folder_name, file_pattern, dataset_name):
    dataset_dicts = []
    assert dataset_name in ['robonetv2/train', 'robonetv2/val', 'robonetv2/test', 'robonetv2/annotation', 'robonetv2/trainval', 'robonetv2/upenn']

    file_list = glob.glob(os.path.join(root, folder_name, '*', '*', '2*', 'raw', 'traj_group0'))
    file_list = sorted(file_list)  # the file list must be sorted
    assert len(file_list) == 397

    # val_id = set()
    # while len(val_id) < 25:
    #     rand_id = random.randint(0, len(file_list))
    #     if rand_id not in val_id:
    #         val_id.add(rand_id)
    # pdb.set_trace()
    # val_id = set(sorted(val_id))

    val_id = [19, 60, 66, 73, 89, 101, 130, 154, 180, 184, 194, 201, 204, 233, 280, 299, 304, 332, 343, 344, 353, 358, 367, 390, 393]

    # test_id = set()
    # while len(test_id) < 25:
    #     rand_id = random.randint(0, len(file_list))
    #     if rand_id not in val_id:
    #         test_id.add(rand_id)
    # test_id = set(sorted(test_id))
    # pdb.set_trace()

    test_id = [5, 7, 61, 84, 90, 95, 103, 119, 152, 162, 169, 202, 234, 242, 244, 253, 256, 272, 291, 306, 310, 315, 327, 375, 395]
    train_id = [i for i in range(len(file_list)) if i not in val_id and i not in test_id]
    for i in test_id:
        assert i not in val_id, i

    if dataset_name == 'robonetv2/train':
        file_list = [file_list[i] for i in train_id]
    elif dataset_name == 'robonetv2/val':
        file_list = [file_list[i] for i in val_id][0:24]
    elif dataset_name == 'robonetv2/test':
        file_list = [file_list[i] for i in test_id][0:24]
    elif dataset_name == 'robonetv2/annotation':
        file_list = glob.glob('./bridge_gt/*')

    elif dataset_name == 'robonetv2/trainval':
        file_list = [file_list[i] for i in train_id]
        temp = [
            # 'put_lid_on_pot_or_pan',
            'put_carrot_on_cutting_board',
            'put_banana_on_plate',
            # 'put_sweet_potato_in_pot_which_is_in_sink'
        ]
        done = []
        new_file_list = []
        for file_name in file_list:
            for k in temp:
                if k in file_name:
                    new_file_list.append(file_name)
                    done.append(k)
                    continue
        pdb.set_trace()
        file_list = sorted(new_file_list)# [-2:]
    elif dataset_name == 'robonetv2/upenn':
        pdb.set_trace()
        # pick_up_pot_50
        pdb.set_trace()
        file_list = sorted(glob.glob(os.path.join(root, 'robonetv2/toykitchen_fixed_cam/upenn', '*', '*', '2*', 'raw', 'traj_group0')))[90:]


    print(dataset_name, len(file_list))
    dataset_dicts += [{'file_name': filename, 'root': root} for filename in file_list]

    return dataset_dicts


def register_robonetv2(root, dataset_name, file_pattern):
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

    folder_name = 'robonetv2/toykitchen_fixed_cam/berkeley'
    DatasetCatalog.register(dataset_name,
                            lambda: load_robonet(root, folder_name=folder_name, file_pattern=file_pattern, dataset_name=dataset_name))

    # # 2. Optionally, add metadata about this dataset,
    # # since they might be useful in evaluation, visualization or logging
    # # meta_path = os.path.join(root, folder_name, 'meta_data.pkl')
    # meta_path = '/data1/honglinc/robonet/hdf5/meta_data.pkl'
    # if not os.path.exists(meta_path):
    #     raise FileNotFoundError("You need to generate the meta_data.pkl file for this dataset")
    #
    # metadict = pd.read_pickle(meta_path, compression='gzip')
    # print('metadict: ', metadict)
    # MetadataCatalog.get('robonet').set(metadict=metadict)


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