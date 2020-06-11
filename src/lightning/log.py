import io
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import torch as t

from collections import OrderedDict
from matplotlib.pyplot import Axes, Figure
from numpy import ndarray
from pathlib import Path
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, lightning: LightningModule, img: Tensor, target_: Tensor, prediction: Tensor):
        # lol mypy type inference really breaks down here...
        self.lightning = lightning
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

    def plot(self) -> Tuple[Figure, Axes]:
        nrows, ncols = 3, 1  # one row for each slice position
        all_trues, all_targets, all_preds = [], [], []
        for i in range(3):  # We want this first so middle images are middle
            for j, position in enumerate(self.slice_positions):
                img, target = self.imgs[position][i], self.targets[position][i]
                prediction = self.preds[position][i]
                all_trues.append(img)
                all_targets.append(target)
                all_preds.append(prediction)
        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
        true = np.concatenate(all_trues, axis=1)
        target = np.concatenate(all_targets, axis=1)
        pred = np.concatenate(all_preds, axis=1)
        # convert to logits, since we are using BCEWithLogitsLoss. That is,
        # BDEWithLogitsLoss handles the sigmoid + loss internally to avoid
        # imprecision issues, but then this means the output of our network
        # never *actually* passes through a sigmoid. So we do that here.
        pred = t.sigmoid(t.tensor(pred)).numpy()

        # Consistently apply colormap since images are standardized but still
        # vary considerably in maximum and minimum values
        true_args = dict(vmin=0.0, vmax=5.0, cmap="gray")
        mask_args = dict(vmin=0.0, vmax=1.0, cmap="gray", alpha=0.5)

        axes[0].imshow(true, **true_args)
        axes[0].imshow(target, **mask_args)
        axes[0].set_title("Actual Brain Tissue (probability)")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(true, **true_args)
        axes[1].imshow(pred, **mask_args)
        axes[1].set_title("Predicted Brain Tissue (probability)")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        axes[2].imshow(true * np.array(pred > 0.5, dtype=float), **true_args)
        axes[2].set_title("Predicted Brain Tissue (binary)")
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        return fig, axes

    def visualize(self, batch_idx: int, outdir: Path = None) -> None:
        fig, axes = self.plot()
        if self.lightning.show_plots:
            if outdir is None:  # for local debugging
                plt.show()
                plt.close()
                return
            fig.set_size_inches(w=10, h=6)
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(outdir / f"epoch{self.lightning.current_epoch}_batch{batch_idx}.png", dpi=200)
            plt.close()
            return

    def log(self, batch_idx: int) -> None:
        logger = self.lightning.logger
        fig, axes = self.plot()
        summary = f"Epoch {self.lightning.current_epoch + 1} - Batch {batch_idx}"
        logger.experiment.add_figure(summary, fig, close=True)

        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)


def tboard_img_summary(img: Tensor, target: Tensor, prediction: Tensor) -> Any:
    raise NotImplementedError()


def log_weights(module: LightningModule) -> None:
    for name, param in module.named_parameters():
        module.logger.experiment.add_histogram(name, param, global_step=module.global_step)

"""
Actual methods on logger.experiment can be found here!!!
https://pytorch.org/docs/stable/tensorboard.html
"""

def log_all_info(module: LightningModule, img: Tensor, target: Tensor, pred: Tensor, batch_idx: int) -> None:
    """Helper for decluttering training loop. Just performs all logging functions."""
    BrainSlices(module, img, target, pred).log(batch_idx)
    log_weights(module)
