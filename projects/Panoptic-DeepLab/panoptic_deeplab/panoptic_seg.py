# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry

from .post_processing import get_panoptic_segmentation, get_instance_segmentation
import pdb
import os
import matplotlib.pyplot as plt
import numpy as np
# from .thingness_evaluate import load_model
from .raft_evaluate import EvalRAFT, viz_flow_seg
from .utils import visualize_center_offset, compute_center_offset, measure_static_segmentation_metric, BCELoss
from mpl_toolkits.axes_grid1 import make_axes_locatable


__all__ = ["PanopticDeepLab", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch"]


INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""


@META_ARCH_REGISTRY.register()
class PanopticDeepLab(nn.Module):
    """
    Main class for panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        assert (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
            == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED

        self.raft_supervision =  cfg.MODEL.PANOPTIC_DEEPLAB.RAFT_SUPERVISION
        self.full_supervision = cfg.MODEL.PANOPTIC_DEEPLAB.FULL_SUPERVISION
        self.raft_threshold = cfg.MODEL.PANOPTIC_DEEPLAB.RAFT_THRESHOLD
        self.input_size = cfg.INPUT.CROP.SIZE
        self.predict_thing_mask = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_THING_MASK

        if self.raft_supervision:
            assert self.raft_threshold is not None
            print('Loading raft model from ./RAFT/models/raft-sintel.pth with threshold', self.raft_threshold)
            self.raft_model = EvalRAFT(ckpt_path='./RAFT/models/raft-sintel.pth')
            # self.raft_model = load_model('./RAFT/models/raft-sintel.pth', small=False, train=False, cuda=True, freeze_bn=True, gpus=[0])
            # load_path = 'thingness-tdw-selfsup-bs2-small-20frames.pth'
            # self.thingness_model = load_model(load_path, small=True, train=True, cuda=True, freeze_bn=False, gpus=[0])

        if self.predict_thing_mask:
            assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1
            self.sem_seg_head.loss = BCELoss()

        self.vis_saved_path = os.path.join('./validation_vis', cfg.OUTPUT_DIR.split('/')[-1])
        self.dataset_name = cfg.DATASETS.TEST[0]
        if not os.path.exists(self.vis_saved_path):
            os.makedirs(self.vis_saved_path)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, iter):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )
        images = ImageList.from_tensors(images, size_divisibility)

        features = self.backbone(images.tensor)

        losses = {}

        # assertion checks
        if 'playroom' in self.dataset_name or 'dsr' in self.dataset_name:
            # assert self.raft_supervision
            assert self.predict_thing_mask, "Training with PDL should have predict_thing_mask=True"
            assert "sem_seg" not in batched_inputs[0]
            assert self.size_divisibility == 0

        if self.raft_supervision:
            outputs = self.create_raft_supervision(batched_inputs)
            center_targets, offset_targets, center_weights, offset_weights, motion_segments = outputs
        elif "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
            offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
            offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor
        else:
            center_targets = None
            center_weights = None

            offset_targets = None
            offset_weights = None

        # # visualize center and offsets supervision signals for training
        # for i in range(len(batched_inputs)):
        #     visualize_center_offset(batched_inputs[i]['image'], center_targets[i], offset_targets[i], center_weights[i], offset_weights[i])

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        elif self.predict_thing_mask:
            assert self.raft_supervision or self.full_supervision, (self.raft_supervision, self.full_supervision)
            if self.raft_supervision:
                targets = motion_segments.float()
            elif self.full_supervision:
                # object_segments = [x["object_segments"].to(self.device) for x in batched_inputs]
                # object_segments = ImageList.from_tensors(object_segments, size_divisibility).tensor
                # targets = (object_segments.unsqueeze(1) > 0).float()
                targets = offset_weights.clone()
                motion_segments = targets
                assert torch.equal(targets, offset_weights), pdb.set_trace()

            weights = None
        else:
            targets = None
            weights = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, targets, weights)
        losses.update(sem_seg_losses)


        center_results, offset_results, center_losses, offset_losses = self.ins_embed_head(
            features, center_targets, center_weights, offset_targets, offset_weights
        )
        losses.update(center_losses)
        losses.update(offset_losses)

        if self.training:
            return losses
        else:
            out = self.postprocessing(sem_seg_results, center_results, offset_results, batched_inputs, images, motion_segments, iter)
            losses.update(out)
            return losses

        if self.benchmark_network_speed:
            return []

        processed_results = []
        for sem_seg_result, center_result, offset_result, input_per_image, image_size in zip(
            sem_seg_results, center_results, offset_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            c = sem_seg_postprocess(center_result, image_size, height, width)
            o = sem_seg_postprocess(offset_result, image_size, height, width)
            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.argmax(dim=0, keepdim=True),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )
            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor(
                            [sem_scores * center_scores], device=panoptic_image.device
                        )
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results

    def create_raft_supervision(self, batched_inputs):
        x1 = [x["image"].to(self.device) for x in batched_inputs]
        x2 = [x["image_1"].to(self.device) for x in batched_inputs]

        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )

        x1 = ImageList.from_tensors(x1, size_divisibility).tensor
        x2 = ImageList.from_tensors(x2, size_divisibility).tensor

        raft_flow = self.raft_model(x1, x2)
        flow_mag = (raft_flow ** 2).sum(1) ** 0.5
        motion_segments = flow_mag > self.raft_threshold
        motion_segments = motion_segments.unsqueeze(1)

        # # Visualize flow outputs
        # for i in range(raft_flow.shape[0]):
        #     viz_flow_seg(x1[i][None], raft_flow.detach()[i][None], motion_segments[i][None, None], flow_mag[i][None].detach())

        center_weights = offset_weights = motion_segments
        center_targets, offset_targets = compute_center_offset(motion_segments)

        return center_targets, offset_targets, center_weights, offset_weights, motion_segments

    def postprocessing(self, sem_seg_results, center_results, offset_results, batched_inputs, images, motion_segments, iter):
        # visualize results and compute metric
        vis = True
        if 'playroom' in self.dataset_name:
            if int(batched_inputs[0]['file_name'].split('/')[-1].split('.hdf5')[0]) > 100:
                vis = False

        _, instance_seg = self.create_instance_segments(sem_seg_results, center_results, offset_results, batched_inputs,
                                                     images, iter, vis)

        if 'playroom' in self.dataset_name:
            sup_metric, _ = self.measure_segments(motion_segments, batched_inputs, moving_only=True)
            sinobj_metric, sinobj_vis = self.measure_segments(instance_seg, batched_inputs, moving_only=True)
            allobj_metric, allobj_vis = self.measure_segments(instance_seg, batched_inputs, moving_only=False)

            file_name = int(batched_inputs[0]['file_name'].split('/')[-1].split('.hdf5')[0])
            if file_name < 100:
                sup_miou = sup_metric['metric_segments_mean_ious']
                self.visualize_segments(sinobj_vis['segments'], motion_segments, batched_inputs, iter,
                                        miou=sup_miou, prefix='sinobj')
                self.visualize_segments(allobj_vis['segments'], motion_segments, batched_inputs, iter,
                                        miou=sup_miou, prefix='allobj')

            return {
                'metric_sup_miou': sup_metric['metric_segments_mean_ious'],
                'metric_sinobj_miou': sinobj_metric['metric_segments_mean_ious'],
                'metric_allobj_miou': allobj_metric['metric_segments_mean_ious'],
            }
        else:
            return {}


    def measure_segments(self, segments, batched_inputs, moving_only):
        data_dict = {'segments': segments.squeeze(1).int()}
        return  measure_static_segmentation_metric(data_dict, batched_inputs, self.input_size,
                                                 segment_key=['segments'],
                                                 moving_only=moving_only,
                                                 eval_full_res=True)


    def create_instance_segments(self, sem_seg_results, center_results, offset_results, batched_inputs, images, iter, vis):
        assert len(batched_inputs) == 1
        for sem_seg_result, center_result, offset_result, input_per_image, image_size in zip(
                sem_seg_results, center_results, offset_results, batched_inputs, images.image_sizes
        ):
            height, width = self.input_size
            assert height == center_result.shape[-2] and width == center_result.shape[-1], (self.input_size, center_result.shape)

            center_heatmap = sem_seg_postprocess(center_result, image_size, height, width)
            offsets = sem_seg_postprocess(offset_result, image_size, height, width)

            if sem_seg_result is None:
                sem_seg = torch.zeros_like(center_heatmap)
                thing_seg = torch.ones_like(sem_seg)
                thing_ids = []
            elif self.predict_thing_mask:
                sem_seg = sem_seg_postprocess(sem_seg_results, image_size, height, width)
                thing_seg = sem_seg.sigmoid() > 0.1


                # _, thing_logits = self.thingness_model(images.tensor, images.tensor, test_mode=True)
                # thing_mask = thing_logits.sigmoid() > 0.05
                #
                # pdb.set_trace()
                # plt.subplot(1, 2, 1)
                # plt.imshow(batched_inputs[0]['image'].permute(1, 2, 0).cpu())
                # plt.subplot(1, 2, 2)
                # plt.imshow(thing_mask[0,0].cpu())
                # plt.show()
                # plt.close()

                thing_ids = []
            else:
                sem_seg = sem_seg_postprocess(sem_seg_results, image_size, height, width)
                sem_seg = sem_seg.argmax(dim=0, keepdim=True)
                thing_ids = self.meta.thing_dataset_id_to_contiguous_id.values()

                # print('using coco thing ids')
                # thing_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

                thing_seg = torch.zeros_like(sem_seg)
                for thing_class in list(thing_ids):
                    thing_seg[sem_seg == thing_class] = 1

            raw_instance, thing_instance, center = get_instance_segmentation(
                sem_seg,
                center_heatmap,
                offsets,
                thing_seg,
                thing_ids,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )

            if vis:
                fig = plt.figure(figsize=(24, 4))
                fontsize = 19

                vis_items = {
                    'Image': batched_inputs[0]['image'].permute(1, 2, 0).cpu(),
                    'Center': center_heatmap[0].cpu(),
                    'Offset-Y': offsets[0].cpu(),
                    'Offset-X': offsets[1].cpu(),
                    'Objectness': thing_seg[0].cpu(),
                    'Raw instance': raw_instance[0].cpu(),
                    'Object instance': thing_instance[0].cpu()
                }

                num_items = len(vis_items.keys())
                for i, (k, v) in enumerate(vis_items.items()):
                    plt.subplot(1, num_items, i+1)

                    plt.imshow(v)
                    plt.title(k, fontsize=fontsize)
                    plt.axis('off')

                file_name = batched_inputs[0]['file_name'].split('/')[-1].split('.hdf5')[0]
                save_path = os.path.join(self.vis_saved_path, 'step_%s_%s.png' % ('eval' if iter is None else str(iter), file_name))
                # print('Save visualization to ', save_path)
                plt.savefig(save_path)
                # plt.show()
                plt.close()

        return raw_instance, thing_instance

    def visualize_segments(self, segment_vis, labels, batched_inputs,  iter, miou, prefix=''):
        matched_cc_preds, matched_gts, cc_ious = segment_vis

        H = W = self.input_size

        fsz = 19
        num_plots = 2+len(matched_cc_preds[0])*2
        fig = plt.figure(figsize=(num_plots * 4, 5))
        gs = fig.add_gridspec(1, num_plots)
        ax1 = fig.add_subplot(gs[0])

        image = batched_inputs[0]['frames'][0]
        plt.imshow(image.permute([1, 2, 0]).cpu())
        plt.axis('off')
        ax1.set_title('Image', fontsize=fsz)

        ax = fig.add_subplot(gs[1])

        if labels is None:
            labels = torch.zeros(1, 1, H, W)
        plt.imshow(labels[0, 0].cpu())
        plt.title('Supervision \n (IoU: %.2f)' % miou, fontsize=fsz)
        plt.axis('off')

        for i, (cc_pred, gt, cc_iou) in enumerate(zip(matched_cc_preds[0], matched_gts[0], cc_ious[0])):
            ax = fig.add_subplot(gs[2 + i])
            ax.imshow(cc_pred)
            ax.set_title('Pred (IoU: %.2f)' % cc_iou, fontsize=fsz)
            plt.axis('off')

            ax = fig.add_subplot(gs[2 + len(matched_cc_preds[0]) + i])
            ax.imshow(gt)
            plt.axis('off')
            ax.set_title('GT %d' % i, fontsize=fsz)

        file_name = batched_inputs[0]['file_name'].split('/')[-1].split('.hdf5')[0]
        save_path = os.path.join(self.vis_saved_path, 'step_%s_%s_%s.png' % ('eval' if iter is None else str(iter), file_name, prefix))
        # print('Save fig to ', save_path)
        plt.savefig(save_path, bbox_inches='tight')

        # plt.show()
        plt.close()

@SEM_SEG_HEADS_REGISTRY.register()
class PanopticDeepLabSemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])
        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=ignore_value, top_k_percent_pixels=loss_top_k)
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.training:
            return y, self.losses(y, targets, weights)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


@INS_EMBED_BRANCHES_REGISTRY.register()
class PanopticDeepLabInsEmbedHead(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset = self.layers(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            center = F.interpolate(
                center, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            offset = (
                F.interpolate(
                    offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
            )
            return center, offset, {}, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return center, offset

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses
