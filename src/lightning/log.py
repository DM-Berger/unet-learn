import numpy as np
import matplotlib.pyplot as plt
import torch as t

from collections import OrderedDict
from numpy import ndarray
from matplotlib.pyplot import Axes, Figure
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union

"""
For TensorBoard logging usage, see:
https://www.tensorflow.org/api_docs/python/tf/summary

For Lightning documentation / examples, see:
https://pytorch-lightning.readthedocs.io/en/latest/experiment_logging.html#tensorboard

NOTE: The Lightning documentation here is not obvious to newcomers. However,
`self.logger` returns the Torch TensorBoardLogger object (generally quite
useless) and `self.logger.experiment` returns the actual TensorFlow
SummaryWriter object (e.g. with all the methods you actually care about)

For the Lightning methods to access the TensorBoard .summary() features, see
https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.html#pytorch_lightning.loggers.TensorBoardLogger

**kwargs for SummaryWriter constructor defined at
https://www.tensorflow.org/api_docs/python/tf/summary/create_file_writer
^^ these args look largely like things we don't care about ^^
"""


def get_logger(logdir: Path) -> TensorBoardLogger:
    return TensorBoardLogger(logdir, name="unet")


# to be called in LightningModule.train_step
def log_progress():
    pass


def slice_label(i: int, mids: Tensor, slicekey: str):
    quarts = mids // 2  # slices at first quarter of the way through
    quarts3_4 = mids + quarts  # slices 3/4 of the way through
    keymap = {"1/4": quarts, "1/2": mids, "3/4": quarts3_4}
    idx = keymap[slicekey]
    if i == 0:
        return f"[{idx[i]},:,:]"
    if i == 1:
        return f"[:,{idx[i]},:]"
    if i == 2:
        return f"[:,:,{idx[i]}]"

    f"[{idx[i]},:,:]", f"[:,{idx[i]},:]", f"[:,:,{idx[i]}]"
    raise IndexError("Only three dimensions supported.")


# https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
class BrainSlices:
    def __init__(self, img: Tensor, target_: Tensor, prediction: Tensor):
        # lol mypy type inference really breaks down here...
        img_: ndarray = img.cpu().detach().numpy().squeeze()
        targ_: ndarray = target_.cpu().detach().numpy().squeeze()
        pred: ndarray = prediction.cpu().detach().numpy().squeeze()
        mids: ndarray = np.array(img_.shape) // 2
        quarts: ndarray = mids // 2  # slices at first quarter of the way through
        quarts3_4: ndarray = mids + quarts  # slices 3/4 of the way through
        self.mids = mids
        self.quarts = quarts
        self.quarts3_4 = quarts3_4
        self.slice_positions = ["1/4", "1/2", "3/4"]
        self.shape = np.array(img_.shape)

        self.imgs = OrderedDict(
            [
                ("1/4", (img_[quarts[0], :, :], img_[:, quarts[1], :], img_[:, :, quarts[2]])),
                ("1/2", (img_[mids[0], :, :], img_[:, mids[1], :], img_[:, :, mids[2]])),
                ("3/4", (img_[quarts3_4[0], :, :], img_[:, quarts3_4[1], :], img_[:, :, quarts3_4[2]])),
            ]
        )
        self.targets = OrderedDict(
            [
                ("1/4", (targ_[quarts[0], :, :], targ_[:, quarts[1], :], targ_[:, :, quarts[2]])),
                ("1/2", (targ_[mids[0], :, :], targ_[:, mids[1], :], targ_[:, :, mids[2]])),
                ("3/4", (targ_[quarts3_4[0], :, :], targ_[:, quarts3_4[1], :], targ_[:, :, quarts3_4[2]])),
            ]
        )
        self.preds = OrderedDict(
            [
                ("1/4", (pred[quarts[0], :, :], pred[:, quarts[1], :], pred[:, :, quarts[2]])),
                ("1/2", (pred[mids[0], :, :], pred[:, mids[1], :], pred[:, :, mids[2]])),
                ("3/4", (pred[quarts3_4[0], :, :], pred[:, quarts3_4[1], :], pred[:, :, quarts3_4[2]])),
            ]
        )
        self.labels = {
            "1/4": [f"[{quarts[0]},:,:]", f"[:,{quarts[1]},:]", f"[:,:,{quarts[2]}]"],
            "1/2": [f"[{mids[0]},:,:]", f"[:,{mids[1]},:]", f"[:,:,{mids[2]}]"],
            "3/4": [f"[{quarts3_4[0]},:,:]", f"[:,{quarts3_4[1]},:]", f"[:,:,{quarts3_4[2]}]"],
        }

    def visualize(self) -> None:
        mids = self.mids
        nrows, ncols = 3, 3  # one row for each slice position
        n_images = 9  # hardcode for now
        # all_trues, all_targets, all_targets_bin, all_preds, all_preds_bin = [], [], [], [], []
        all_trues, all_targets, all_preds = [], [], []
        for i in range(3):  # We want this first so middle images are middle
            for j, position in enumerate(self.slice_positions):
                img, target = self.imgs[position][i], self.targets[position][i]
                prediction = self.preds[position][i]
                # true = img + target
                # pred = img + prediction
                all_trues.append(img)
                all_targets.append(target)
                # all_targets_bin.append(np.array(target > 0.5, dtype=np.int64))
                all_preds.append(prediction)
                # all_preds_bin.append(np.array(prediction > 0.5, dtype=np.int64))
        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(nrows=4, ncols=1, sharex=False, sharey=False)
        true = np.concatenate(all_trues, axis=1)
        target = np.concatenate(all_targets, axis=1)
        # target_bin = np.concatenate(all_targets_bin, axis=1)
        pred = np.concatenate(all_preds, axis=1)
        # pred_bin = np.concatenate(all_preds_bin, axis=1)

        """The problem here is that, after augmentation, our masks are complicated
        arrays of floats, with the maximum being 1.0, and the minimum being 0.0.

        This makes sense, because when augmenting a mask, you are going to have
        interpolation errors / uncertainties. The only way for augmentation not
        to introduce excessive error is to actaully treat the masks as array of
        floats that correspond to probabilities, based on the original masks.

        We can get around this by cheating with the `alpha` argument of
        matplotlib, which, if an array, works pixel-by-pixel.
        """
        axes[0].imshow(true * target, cmap="inferno")
        axes[0].set_title("Actual Brain Tissue (probability)")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        # axes[0].imshow(target, cmap="inferno", alpha=0.5)
        axes[1].imshow(true * pred, cmap="inferno")
        axes[1].set_title("Predicted Brain Tissue (probability)")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        axes[2].imshow(true * np.round(target), cmap="inferno")
        axes[2].set_title("Actual Brain Tissue (binary)")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        # axes[0].imshow(target, cmap="inferno", alpha=0.5)
        axes[3].imshow(true * np.round(pred), cmap="inferno")
        axes[3].set_title("Predicted Brain Tissue (binary)")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        # axes[1].imshow(pred, cmap="inferno", alpha=0.5)

        # both = np.concatenate([true, pred], axis=1)

        # axes[i][j].imshow(both, cmap="inferno", label="img")
        # axes[i][j].set_title("")
        # axes[i][j].set_xlabel(self.labels[position][i])
        # fig.suptitle("True (top) vs. Predicted (right)")
        plt.show()


# def tboard_img_summary(img: Tensor, target: Tensor, prediction: Tensor) -> Any:
#     mids, slices = get_slices(img, target, prediction)
#     for i, (img, target, pred) in enumerate(zip(slices["img"], slices["target"], slices["pred"])):
#         masked_true = img + target
#         masked_pred = img + pred
