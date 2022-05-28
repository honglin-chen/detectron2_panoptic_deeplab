import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torchvision import transforms
from kornia.contrib import connected_components

from detectron2.projects.panoptic_deeplab.segmentation_metrics import SegmentationMetrics
import pdb
def visualize_center_offset(image, center_targets, offset_targets, center_weights, offset_weights):
    print(image.shape, center_targets.shape, offset_targets.shape, center_weights.shape, offset_weights.shape)
    plt.figure(figsize=(30, 5))
    fontsize = 19
    plt.subplot(1, 6, 1)
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.title('Image', fontsize=fontsize)

    plt.subplot(1, 6, 2)
    plt.imshow(center_targets[0].cpu())
    plt.title('Center targets', fontsize=fontsize)

    plt.subplot(1, 6, 3)
    plt.imshow(offset_targets[0].cpu())
    plt.title('Offset-y targets', fontsize=fontsize)

    plt.subplot(1, 6, 4)
    plt.imshow(offset_targets[1].cpu())
    plt.title('Offset-x targets', fontsize=fontsize)

    plt.subplot(1, 6, 5)
    plt.imshow(center_weights[0].cpu())
    plt.title('Center weights', fontsize=fontsize)

    plt.subplot(1, 6, 6)
    plt.imshow(offset_weights[0].cpu())
    plt.title('Offset weights', fontsize=fontsize)

    plt.show()
    plt.close()


def compute_center_offset(mask, sigma=8.0):

    B, _, H, W = mask.shape

    assert mask.shape[1] == 1, mask.shape
    mask = mask.squeeze(1)

    y_coord, x_coord = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    y_coord = torch.tensor(y_coord).to(mask.device)
    x_coord = torch.tensor(x_coord).to(mask.device)

    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g = torch.tensor(g).to(mask.device)

    center_targets = []
    offset_targets = []
    for i in range(B):
        center = torch.zeros([H, W]).to(mask.device)
        offset = torch.zeros([2, H, W]).to(mask.device)

        mask_index = torch.where(mask[i])

        if len(mask_index[0]) == 0:
            center_targets.append(torch.zeros([1, 1, H, W]).to(mask.device))
            offset_targets.append(torch.zeros([1, 2, H, W]).to(mask.device))
            continue

        center_y, center_x = torch.mean(mask_index[0].float()), torch.mean(mask_index[1].float())

        # generate center heatmap
        y, x = center_y.round().int(), center_x.round().int()

        # upper left
        ul = (x - 3 * sigma - 1).round().int(), (y - 3 * sigma - 1).round().int()
        # bottom right
        br = (x + 3 * sigma + 2).round().int(), (y + 3 * sigma + 2).round().int()

        # start and end indices in default Gaussian image
        gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], W) - ul[0]
        gaussian_y0, gaussian_y1 = max(0, -ul[1]), min(br[1], H) - ul[1]

        # start and end indices in center heatmap image
        center_x0, center_x1 = max(0, ul[0]), min(br[0], W)
        center_y0, center_y1 = max(0, ul[1]), min(br[1], H)
        center[center_y0:center_y1, center_x0:center_x1] = torch.maximum(
            center[center_y0:center_y1, center_x0:center_x1],
            g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
        )

        # generate offset (2, h, w) -> (y-dir, x-dir)
        offset[0][mask_index] = center_y - y_coord[mask_index]
        offset[1][mask_index] = center_x - x_coord[mask_index]

        center_targets.append(center[None, None])
        offset_targets.append(offset[None])

    center_targets = torch.cat(center_targets, dim=0)
    offset_targets = torch.cat(offset_targets, dim=0)

    return center_targets, offset_targets


def _object_id_hash(objects, val=256, dtype=torch.long):
    C = objects.shape[0]
    objects = objects.to(dtype)
    out = torch.zeros_like(objects[0:1, ...])
    for c in range(C):
        scale = val ** (C-1-c)
        out += scale * objects[c:c+1, ...]
    return out


def delta_image(images, thresh=None):
    """
    Get a map of parts of the images that are changing
    :param images: image tensor of shape [T, 3, H, W]
    :return: delta image tensor of shape [T-1, H, W]
    """
    assert len(images.shape) == 4, "image tensor must have 4 dimensions"
    assert images.shape[1] == 3, "1st dimension of the image tensor must be 3"
    assert images.shape[0] > 1, "computing delta image requires at least 2 images"

    intensities = images.mean(1)
    delta_images = (intensities[1:] - intensities[:-1]).abs()

    if thresh is not None:
        delta_images = (delta_images > thresh).float()

    assert delta_images.shape[0] == 1, "current implementation only support sequence len = 1"
    delta_images_floodfill = fill_contour(delta_images[0])
    delta_images_floodfill = torch.as_tensor(np.ascontiguousarray(delta_images_floodfill[None]))
    return delta_images, (delta_images_floodfill >= 1.).float()


def fill_contour(img):
    img_copy = np.copy(img)
    img = img.numpy().astype(np.uint8)

    # Find Contour
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # You need to make a list for cv2.drawContours()! ! ! ! !
    c_max = []
    max_area = 0
    max_cnt = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # find max countour
        if (area>max_area):
            if(max_area!=0):
                c_min = []
                c_min.append(max_cnt)
                cv2.drawContours(img, c_min, -1, (0,0,0), cv2.FILLED)
            max_area = area
            max_cnt = cnt
        else:
            c_min = []
            c_min.append(cnt)
            cv2.drawContours(img, c_min, -1, (0,0,0), cv2.FILLED)

    c_max.append(max_cnt)
    if isinstance(max_cnt, int):
        return img_copy
    else:
        cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)
        return img

def optical_flow(image, next_image):

    gray = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_image.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method

    flow = cv2.calcOpticalFlowFarneback(gray, next_gray,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    return torch.as_tensor(np.ascontiguousarray(magnitude))

def visualize_views(dataset_dict, plot_keys=None):
    plt.figure(figsize=(25, 5))
    if plot_keys is None:
        plot_keys = ['image', 'objects', 'segment_id_map', 'delta_image', 'delta_image_floodfill', 'probe',
                     'dual_delta_images', 'prev_delta_images', 'post_delta_images']#, 'flow', 'objects', 'segment_id_map']

    num_plots = len(plot_keys)

    for key, data in dataset_dict.items():
        if key in plot_keys:
            if len(data.shape) == 4:  # [B, T, D, H, W]
                num_plots += (data.shape[0] - 1)

    count = 1
    for key, data in dataset_dict.items():
        if key in plot_keys:
            print(key, data.shape)
            plt.subplot(1, num_plots, count)
            if len(data.shape) == 2:
                plt.imshow(data)
            else:
                if 'diff' in key:
                    plt.imshow(data.permute(1, 2, 0), cmap='coolwarm')
                elif key == 'offset':
                    plt.imshow(data.permute(1, 2, 0)[..., 0])
                elif len(data.shape) == 4:
                    for t in range(data.shape[0]):
                        plt.subplot(1, num_plots, count)
                        plt.imshow(data[t].permute(1, 2, 0))
                        plt.title('frames: %d' % t)
                        if t < (data.shape[0] - 1):
                            count += 1
                else:
                    plt.imshow(data.permute(1, 2, 0))

            if key in ['segment_id_map', 'objects']:
                plt.title(key+':%d' % len(torch.unique(data)), fontsize=14)
            else:
                plt.title(key, fontsize=14)
            plt.axis('off')
            count += 1
    print('save input images')
    # file_name = dataset_dict['file_name']
    # plt.savefig('./tmp/image_%s.png' % file_name.split('/')[-1], bbox_inches='tight')
    plt.show()
    plt.close()

def gather_tensor(tensor, sample_inds):
    # tensor is of shape [B, N, D]
    # sample_inds is of shape [2, B, T, K] or [3, B, T, K]
    # where the last column of the 1st dimension are the sample indices
    _, N, D = tensor.shape
    dim, B, T, K = sample_inds.shape
    if dim == 2:
        indices = sample_inds[-1].view(B, T * K).unsqueeze(-1).expand(-1, -1, D)
        output = torch.gather(tensor, 1, indices).view([B, T, K, D])
    elif dim == 3:
        tensor = tensor.view(B, N * D)
        node_indices = sample_inds[1].view(B, T * K)
        sample_indices = sample_inds[2].view(B, T * K)
        indices = node_indices * D + sample_indices
        # print('in gather tensor: ', indices.max(), tensor.shape)
        output = torch.gather(tensor, 1, indices).view([B, T, K])
    else:
        raise ValueError
    return output

def reorder_int_labels(x):
    _, y = torch.unique(x, return_inverse=True)
    y -= y.min()
    return y

def visualize_flow(image, flow):

    image = image.permute(1,2,0).cpu().numpy()
    flow = flow.permute(1,2,0).cpu().detach().numpy()

    # map flow to rgb image
    flow = flow_viz.flow_to_image(flow)
    image_flow = np.concatenate([image, flow], axis=0)

    # plt.imshow(image_flow / 255.0)
    # plt.show()
    # plt.close()

    return flow

def sample_image_inds_from_probs(probs, num_points, eps=1e-8):

    B,H,W = probs.shape
    P = num_points
    N = H*W

    probs = probs.reshape(B,N)
    print('probs: ', probs.sum())
    probs = torch.maximum(probs + eps, torch.tensor(0.).to(probs.device)) / (probs.sum(dim=-1, keepdim=True) + eps)
    dist = Categorical(probs=probs)

    indices = dist.sample([P]).permute(1,0).to(torch.int32) # [B,P]

    # indices_h = torch.minimum(torch.maximum(torch.div(indices, W, rounding_mode='floor'), torch.tensor(0)), torch.tensor(H-1))
    # Modification (Honglin): Pytorch 1.7 doesn't have the rounding_mode option
    indices_h = torch.minimum(torch.maximum(indices // W, torch.tensor(0).to(indices)), torch.tensor(H - 1).to(indices))
    indices_w = torch.minimum(torch.maximum(torch.fmod(indices, W), torch.tensor(0).to(indices)), torch.tensor(W-1).to(indices))
    indices = torch.stack([indices_h, indices_w], dim=-1) # [B,P,2]
    return indices


def measure_static_segmentation_metric(out, inputs, size, segment_key,
                                       eval_full_res=False, moving_only=True, exclude_zone=True, eval_motion_crop=False):

    assert len(inputs) == 1

    gt_objects = inputs[0]['segment_id_map'].int()[None]
    assert gt_objects.max() < torch.iinfo(torch.int32).max, gt_objects
    if not eval_full_res:
        gt_objects = F.interpolate(gt_objects.float().unsqueeze(1), size=size, mode='nearest').int()

    gt_objects_copy = gt_objects.clone()

    if moving_only:
        # only evaluate the metric for the moving segments
        if 'moving_id' not in inputs[0].keys():
            exclude_values = []
        else:
            moving_idx = inputs[0]['moving_id'][0, 0]
            exclude_values = inputs[0]['per_segment_id'][0].cpu().numpy().tolist()
            exclude_values.remove(moving_idx)
    elif exclude_zone:
        if 'per_segment_id' not in inputs[0].keys():
            exclude_values = []
        else:
            exclude_values = [inputs[0]['per_segment_id'][:, -2]]  # excluding zone in eval

    else:
        exclude_values = []
    if not isinstance(segment_key, list):
        segment_key = [segment_key]

    segment_metric = {}
    segment_out = {}
    for key in segment_key:
        results = {'mean_ious': [], 'recalls': [], 'boundary_f1_scores': []}
        pred_objects = out[key]
        pred_objects = pred_objects.reshape(pred_objects.shape[0], 1, size[0], size[1])

        if eval_motion_crop:
            # prepare the data for motion cropping
            unique = pred_objects.unique()

            # -- reshape is necessary for the pred objects
            assert gt_objects_copy.shape[-1] == 512 and gt_objects_copy.shape[0] == 1, gt_objects_copy.shape
            pred_objects = F.interpolate(pred_objects.float(), [512, 512], mode='nearest').long()
            assert torch.equal(pred_objects.unique(), unique)
            data = {'pred': pred_objects, 'gt': gt_objects_copy.unsqueeze(1)}
            prob = inputs['delta_image_floodfill'][0]

            # apply motion cropping
            data = MOTION_CROP_MODULE(data, prob, training=False)

            pred_objects, gt_objects = data['pred'], data['gt'].squeeze(1)
            pred_objects = F.interpolate(pred_objects.float(), size, mode='nearest').long()

        metric = SegmentationMetrics(gt_objects=gt_objects.cpu(),
                                     pred_objects=pred_objects.int().cpu(),
                                     size=None if eval_full_res else size,
                                     background_value=0)

        metric.compute_matched_IoUs(exclude_gt_ids=list(set([0] + exclude_values)))
        metric.compute_recalls()
        metric.compute_boundary_f_measures(exclude_gt_ids=list(set([0] + exclude_values)))

        results['mean_ious'].append(metric.mean_ious)
        results['recalls'].append(metric.recalls)
        results['boundary_f1_scores'].append(metric.mean_boundary_f1_scores)

        for k, v in results.items():
            segment_metric[f'metric_{key}_{k}'] = torch.tensor(np.mean(v))
        segment_out[key] = metric.seg_out

    return segment_metric, segment_out


def copy_freeze_params(model_a, model_b):
    # copy the params in model_b from model_a
    # freeze the params in model_b

    model_b.load_state_dict(model_a.state_dict())
    for param_b in model_b.parameters():
        param_b.requires_grad = False     # not update by gradient


def score_prediction(pred_motion, pred_segments, K=256):

    B = pred_segments.shape[0]
    size = pred_segments.shape[-2:]
    R = transforms.Resize(size)

    ## get the motion segment by thresholding
    motion_seg = F.interpolate(pred_motion.float(), size, mode='nearest').squeeze(1)
    # motion_seg = (R(pred_motion).square().sum(-3).sqrt() > self.thresh).float()

    ## get the static segments
    assert pred_segments.max() < K
    static_segs = F.one_hot(pred_segments, K).float()
    # remove the one-hot encoding of the background
    static_segs = static_segs[..., 1:]

    other_unexplained_motion_segments = []
    motion_seg_copy = motion_seg.clone()
    if motion_seg.unique().shape[0] > 2:
        # find the largest motion segment
        assert pred_motion.max() < K
        motion_seg = F.one_hot(motion_seg.long(), K).float()
        area = motion_seg.sum(dim=(-3, -2))
        area[:, 0] = 0 # ignore the background region
        argmax_area = area.argmax(-1).long()

        other_unexplained_motion_segments = list(area[0].nonzero())
        other_unexplained_motion_segments.remove(argmax_area)

        b_inds = torch.arange(B, dtype=torch.long, device=argmax_area.device)
        inds = torch.stack([b_inds[:, None], argmax_area[:, None]], 0)
        motion_seg = motion_seg.permute(0, 3, 1, 2)[list(inds)][:, 0].long()  # [B,H,W]


    ## find the segment that overlaps most with the motion segments
    overlaps = (motion_seg[...,None] * static_segs).sum(dim=(-3,-2))
    best_overlaps = overlaps.argmax(-1).long() # [B]
    b_inds = torch.arange(B, dtype=torch.long, device=best_overlaps.device)
    inds = torch.stack([
        b_inds[:,None],
        best_overlaps[:,None]], 0)
    explained = static_segs.permute(0,3,1,2)[list(inds)][:,0].long() # [B,H,W]
    ## "explain away" by setting remaining pixels to a new value
    unexplained = (motion_seg.long() - explained).clamp(min=0)
    score = unexplained.float().mean()

    out_segments = unexplained + 2 * explained

    if len(other_unexplained_motion_segments) > 0:
        max_idx = out_segments.max()
        for idx in other_unexplained_motion_segments:
            out_segments[motion_seg_copy == idx] = max_idx + 1
            max_idx += 1

    return out_segments, score

def reorder_int_labels(x):
    _, y = torch.unique(x, return_inverse=True)
    y -= y.min()
    return y


def label_connected_component(labels, min_area=20, topk=256):
    size = labels.size()
    assert len(size) == 2
    max_area = size[0] * size[1] - 1

    # per-label binary mask
    unique_labels = torch.unique(labels).reshape(-1, 1, 1)  # [?, 1, 1], where ? is the number of unique id
    unique_labels = unique_labels[unique_labels > 0]  # remove the zero id, which is the background
    binary_masks = (labels.unsqueeze(0) == unique_labels).float()  # [?, H, W]


    # label connected components
    # cc is an integer tensor, each unique id represents a single connected component
    cc = connected_components(binary_masks.unsqueeze(1), num_iterations=500)  # [?, 1, H, W]

    # reorder indices in cc so that cc_area tensor below is a smaller
    cc = reorder_int_labels(cc)

    # area of each connected components
    cc_area = torch.bincount(cc.long().flatten().cpu()).cuda()  # bincount on GPU is much slower
    num_cc = cc_area.shape[0]
    valid = (cc_area >= min_area) & (cc_area <= max_area)  # [num_cc]

    if num_cc < topk:
        selected_cc = torch.arange(num_cc).cuda()
    else:
        _, selected_cc = torch.topk(cc_area, k=topk)
        valid = valid[selected_cc]

    # collapse the 0th dimension, since there is only matched one connected component (across 0th dimension)
    cc_mask = (cc == selected_cc.reshape(1, -1, 1, 1)).sum(0)  # [num_cc, H, W]
    cc_mask = cc_mask * valid.reshape(-1, 1, 1)
    out = cc_mask.argmax(0)
    return out

def apply_cc(mask):
    assert len(mask.shape) == 2
    mask = label_connected_component(mask, min_area=10)  # [H, W]
    mask = reorder_int_labels(mask)  # [H, W]
    return mask

def filter_small_connected_component(labels, min_area=10, invalid_value=0):
    size = labels.size()
    assert len(size) == 2
    # max_area = size[0] * size[1] - 1

    # per-label binary mask
    unique_labels = torch.unique(labels).reshape(-1, 1, 1)  # [?, 1, 1], where ? is the number of unique id
    binary_masks = (labels.unsqueeze(0) == unique_labels).float()  # [?, H, W]

    # filter the binary mask first
    # if the binary mask has area smaller than min_area, then its CC must be smaller than min_area
    area = binary_masks.flatten(1, 2).sum(-1)
    valid_area_mask = area > min_area

    if valid_area_mask.sum() < valid_area_mask.shape[0]:  # filter
        invalid_label_mask = binary_masks[~valid_area_mask].sum(0) > 0
        labels = invalid_label_mask * invalid_value + ~invalid_label_mask * labels
        binary_masks = binary_masks[valid_area_mask]


    # label connected components
    # cc is an integer tensor, each unique id represents a single connected component
    cc = connected_components(binary_masks.unsqueeze(1), num_iterations=200)  # [?, 1, H, W]

    # reorder indices in cc so that cc_area tensor below is a smaller
    cc = reorder_int_labels(cc)

    # area of each connected components
    cc_area = torch.bincount(cc.long().flatten().cpu()).cuda()  # bincount on GPU is much slower
    num_cc = cc_area.shape[0]
    cc_idx = torch.arange(num_cc).cuda()
    assert torch.equal(cc.sum(0), cc.max(0)[0])  # make sure the CCs are mutually exclusive
    cc = cc.sum(0)  # collapse the 0th dimension (note: the CCs must be mutually exclusive)

    # find ccs that are greater than min area and set the invalid ones to zeros
    # there are two ways to implement it -- choose the one that is more memory efficient
    num_valid_segments = (cc_area >= min_area).sum()
    num_invalid_segments = (cc_area < min_area).sum()

    if num_valid_segments < num_invalid_segments:
        valid = cc_area >= min_area
        valid_cc_idx = cc_idx[valid].view(-1, 1, 1)
        valid_mask = (cc == valid_cc_idx).sum(0) > 0
        valid_labels = valid_mask * labels + ~valid_mask * invalid_value
    else:
        invalid = cc_area < min_area
        invalid_cc_idx = cc_idx[invalid].view(-1, 1, 1)
        invalid_mask = (cc == invalid_cc_idx).sum(0) > 0
        valid_labels = ~invalid_mask * labels + invalid_mask * invalid_value

    return valid_labels

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, weights):
        assert weights is None, "weighted version is not implemented yet"
        return self.criterion(inputs, targets)