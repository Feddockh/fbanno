import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import BoundingBoxes
from PIL import Image
from typing import List, Optional, Dict, Tuple


color_palette = [
    "red", "green", "blue", "cyan", "magenta", "yellow",
    "orange", "purple", "pink", "lime", "brown", "gray",
]

def get_color(index):
    return color_palette[index % len(color_palette)]

def to_uint8(img) -> torch.Tensor:
    # If PIL Image, convert to tensor first
    if isinstance(img, Image.Image):
        img = F.pil_to_tensor(img)  # uint8 [C,H,W]
    # If float tensor, scale to [0,255] and convert manually
    if img.dtype.is_floating_point:
        img = (img * 255.0).clamp(0, 255).to(torch.uint8)
    else:
        img = img.to(torch.uint8)
    return img

def plot(imgs, row_title=None, col_title=None, class_names=None, save_path=None, **imshow_kwargs):
    """
    Plot a grid of images with optional bounding boxes, masks, and labels.

    Args:
        imgs (list or list of lists of tuples): Each element is an (image, target) tuple.
        row_title (list, optional): Titles for each row.
        col_title (list, optional): Titles for each column.
        class_names (list, optional): List of class names (idx 0 = background).
        save_path (str, optional): Path to save the plot.
        imshow_kwargs (dict, optional): Additional arguments for plt.imshow().
    """

    if not isinstance(imgs[0], list):
        imgs = [imgs]

    n_rows, n_cols = len(imgs), len(imgs[0])
    fig, axs = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(4*n_cols, 4*n_rows))

    # Construct a grid of images
    for r, row in enumerate(imgs):
        for c, sample in enumerate(row):
            img = sample
            boxes = masks = labels_t = scores = None

            # If sample is a tuple, unpack it
            if isinstance(sample, tuple):
                img, tgt = sample
                if isinstance(tgt, dict):
                    boxes    = tgt.get("boxes")
                    masks    = tgt.get("masks")
                    labels_t = tgt.get("labels")
                    scores   = tgt.get("scores")
                elif isinstance(tgt, BoundingBoxes):
                    boxes = tgt
                else:
                    raise ValueError(f"Unexpected target type: {type(tgt)}")

            # Convert img to a uint8 tensor [C,H,W]
            img = to_uint8(img)

            if masks is not None:
                # Masks should be Tensor [N,H,W] or a list of H×W bool/uint8
                m = masks if isinstance(masks, torch.Tensor) else torch.stack(masks)
                img = draw_segmentation_masks(
                    img, m.to(torch.bool),
                    colors=[get_color(i) for i in range(m.shape[0])],
                    alpha=0.5
                )

            # Convert boxes to a tensor if they are not already
            if boxes is not None and len(boxes):
                # Unpack boxes if they are a list of tensors
                if hasattr(boxes, "tensor"):
                    boxes = boxes.tensor
                # Convert to tensor if they are a list of lists
                if boxes.dtype.is_floating_point:
                    boxes = boxes.round().to(torch.int32)
                else:
                    boxes = boxes.to(torch.int32)

                # Prepare the labels and scores
                labels_list = labels_t.tolist() if labels_t is not None else [0]*len(boxes)
                scores_list = scores.tolist() if scores is not None else [None]*len(boxes)

                lbls, cols = [], []
                for idx, lab in enumerate(labels_list):
                    nm = class_names[lab] if (class_names and lab < len(class_names)) else str(lab)
                    sc = scores_list[idx]
                    if sc is not None:
                        nm = f"{nm}: {sc:.2f}"
                    lbls.append(nm)
                    cols.append(get_color(lab))

                img = draw_bounding_boxes(
                    img, boxes,
                    labels=lbls,
                    colors=cols,
                    width=3,
                    # font_size=20,
                )

            ax = axs[r, c]
            # C×H×W to H×W×C numpy
            ax.imshow(img.permute(1, 2, 0).cpu().numpy(), **imshow_kwargs)
            ax.set(xticks=[], yticks=[])

    # Add titles to the rows
    if row_title is not None:
        for r in range(n_rows):
            axs[r, 0].set_ylabel(row_title[r], rotation=0, labelpad=40, va="center")

    # Add titles to the columns
    if col_title is not None:
        for c in range(n_cols):
            axs[0, c].set_title(col_title[c], fontsize=16)

    # Add legend for class names
    if class_names:
        handles = [
            mpatches.Patch(color=get_color(i), label=class_names[i])
            for i in range(len(class_names))
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            title="Classes",
            fontsize=12,
            title_fontsize=14,
            frameon=False,
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()

def plot_loss(train_losses: List[float], val_losses: Optional[List[float]] = None, 
              save_path: str = "loss_plot.png") -> None:
    """
    Plot training and validation loss curves and save to disk.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list, optional): List of validation losses per epoch.
        save_path (str): Path to save the plot.
    """
    plt.figure()
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label="Train Loss")
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pr_curves(results, metric, class_names, max_x=1.0, max_y=1.05, save_dir=None):
    """
    This function plots the precision-recall curves for each class and an overall curve.
    
    Args:
        results (dict): The results dictionary containing precision and recall data.
        metric (object): The metric object containing IoU thresholds.
        class_names (list): List of class names for the dataset.
        save_path (str): Path to save the plots.
    """
    # Precision has the shape [T, R, K, A, M]
    prec = results["precision"]
    T, R, K, A, M = prec.shape

    # Get the IoU thresholds used
    iou_ths = metric.iou_thresholds  # [0.50,0.55,…,0.95]

    # Recall thresholds vector [R,]
    rec_ths = np.linspace(0.0, 1.0, R) # [0.0,0.01,…,1.0]

    # Plotting over all areas with max detections
    area_idx = 0
    mdet_idx = M - 1

    for cls_idx, cls_name in enumerate(class_names):
        plt.figure(figsize=(6,4))
        for t, thr in enumerate(iou_ths):
            # Precision curve at IoU=thr for class
            p = prec[t, :, cls_idx, area_idx, mdet_idx].cpu().numpy()
            plt.plot(rec_ths, p, label=f"IoU={thr:.2f}")

        plt.title(f"P–R curve for '{cls_name}' class")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(0, max_y)
        plt.xlim(0, max_x)
        plt.legend(loc="upper right", fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{cls_name}_pr_curves.png"), dpi=300)

    # Also plot overall curve (precision averaged over classes)
    plt.figure(figsize=(6,4))
    avg_prec = prec.mean(axis=2) # shape [T, R, A, M]
    for t, thr in enumerate(iou_ths):
        p = avg_prec[t, :, area_idx, mdet_idx].cpu().numpy()
        plt.plot(rec_ths, p, label=f"IoU={thr:.2f}")
    plt.title("Overall P–R curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, max_y)
    plt.xlim(0, max_x)
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "overall_pr_curves.png"), dpi=300)
    else:
        plt.show()
    plt.close('all')