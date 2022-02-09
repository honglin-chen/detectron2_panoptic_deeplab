import torch
import numpy as np
import matplotlib.pyplot as plt


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
    plt.title('Offset-x targets', fontsize=fontsize)

    plt.subplot(1, 6, 4)
    plt.imshow(offset_targets[1].cpu())
    plt.title('Offset-y targets', fontsize=fontsize)

    plt.subplot(1, 6, 5)
    plt.imshow(center_weights[0].cpu())
    plt.title('Center weights', fontsize=fontsize)

    plt.subplot(1, 6, 6)
    plt.imshow(offset_weights[0].cpu())
    plt.title('Offset weights', fontsize=fontsize)

    plt.show()
    plt.close()


def compute_center_offset(mask, sigma=8.0):

    B, H, W = mask.shape

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


