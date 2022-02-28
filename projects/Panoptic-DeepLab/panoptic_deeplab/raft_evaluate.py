import torch
import argparse
import numpy as np
import sys
import cv2
import os
sys.path.append('./RAFT/core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args, _ = parser.parse_known_args()
from mpl_toolkits.axes_grid1 import make_axes_locatable


class EvalRAFT:
    def __init__(self, ckpt_path='./RAFT/models/raft-things.pth'):
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model = model.module
        model.eval()
        self.model = model.cuda()

    def __call__(self, image0, image1):
        flow_low, flow_up = self.model(image0, image1, iters=20, test_mode=True)
        return flow_up


def viz_flow(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    plt.imshow(img_flo / 255.0)
    plt.show()
    plt.close()

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    # cv2.waitKey()


def viz_flow_seg(img, flo, seg, mag, miou=None, precision=None, save=False, filename=None):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    seg = seg[0].permute(1, 2, 0).cpu().numpy()
    mag = mag.permute(1, 2, 0).cpu().numpy()

    flo = flow_viz.flow_to_image(flo) / 255.
    img = img / 255.

    fontsize=15
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('Image', fontsize=fontsize)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(flo)
    plt.title('Flow', fontsize=fontsize)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    im = plt.imshow(mag)
    plt.title('Magnitude', fontsize=fontsize)
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


    plt.subplot(1, 4, 4)
    plt.imshow(seg)
    miou_string = 'none' if miou is None else '%.2f' % miou
    precision_string = 'none' if precision is None else '%.2f' % precision
    plt.title('Supervision \n (IoU: %s, Prec: %s)' % (miou_string, precision_string), fontsize=fontsize)
    plt.axis('off')
    fig.tight_layout()
    plt.show()

    if save:
        assert filename is not None
        path = './raft_flow/%s.png' % filename

        if not os.path.exists(path.split('frame')[0]):
            os.makedirs(path.split('frame')[0])
        plt.savefig('./raft_flow/%s.png' % filename)
        print('save raft flow visualization', filename)

    plt.close()