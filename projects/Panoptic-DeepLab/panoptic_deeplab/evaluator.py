from __future__ import absolute_import, division, print_function

import functools
import sys

import numpy as np
# from tensorflow.python.keras import backend as keras_backend
# from tensorflow.python.ops import math_ops
# import tensorflow.compat.v1 as tf
# import tensorflow.contrib as contrib
import torch
import sys
sys.path.append('/mnt/fs6/honglinc/VisualVectorizingNetwork')
from detectron2.evaluation import DatasetEvaluator
#import vvn.models.panoptic_segmentation.resnet_fpn as resnet_fpn
#import vvn.models.panoptic_segmentation.panoptic_segmentation_model as pm
import copy
import pdb
import math
from collections import OrderedDict
import detectron2.utils.comm as comm
import itertools

class KPCompetitionEvaluator(DatasetEvaluator):
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """
    def __init__(self, dataset_name, distributed=True):
        self.result = dict()
        self.prefix = dataset_name.split('/')[-1] + '/'
        self._distributed = distributed

    def process(self, inputs, outputs):

        if len(self.result) == 0:
            for k, v in outputs.items():
                self.result[k] = [v]
        else:
            for k, v in outputs.items():
                self.result[k].append(v)

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """

        if self._distributed:
            comm.synchronize()
            result = comm.gather(self.result, dst=0)

            if not comm.is_main_process():
                return {}

            collate_results = result[0]
            for i, r in enumerate(result):
                if i > 0:
                    for k, v in r.items():
                        collate_results[k] += v

        else:
            collate_results = self.result

        out = OrderedDict()

        # agg rest of the metrics by taking mean

        for k, v in collate_results.items():
            if 'metric' in k or 'loss' in k:
                print('Evaluation on %s with %d items' % (k, len(v)))
                v_cpu = [_v.cpu() for _v in v]
                out[self.prefix+k] = torch.tensor(np.nanmean(torch.stack(v_cpu).numpy()))
        return out

