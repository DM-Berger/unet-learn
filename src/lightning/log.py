import numpy as np
import matplotlib.pyplot as plt
import torch as t

from matplotlib.pyplot import Axes, Figure
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor

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


def visualize(img: Tensor, target: Tensor, prediction: Tensor) -> None:
    def slice_label(i: int, mids: Tensor):
        if i == 0:
            return f"[{mids[i]},:,:]"
        if i == 1:
            return f"[:,{mids[i]},:]"
        if i == 2:
            return f"[:,:,{mids[i]}]"
        raise IndexError("Only three dimensions supported.")

    img = img.cpu().detach().numpy().squeeze()
    target = target.cpu().detach().numpy().squeeze()
    pred = prediction.cpu().detach().numpy().squeeze()
    mids = np.array(img.shape) // 2
    img_slices = [img[mids[0], :, :], img[:, mids[1], :], img[:, :, mids[2]]]
    target_slices = [target[mids[0], :, :], target[:, mids[1], :], target[:, :, mids[2]]]
    pred_slices = [pred[mids[0], :, :], pred[:, mids[1], :], pred[:, :, mids[2]]]

    fig: Figure
    axes: Axes
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    for i, (img, target, pred) in enumerate(zip(img_slices, target_slices, pred_slices)):
        ax, pred_ax = axes[0][i], axes[1][i]
        # plot true img and true segmentations
        ax.imshow(img, cmap="inferno", label="img")
        ax.imshow(target, cmap="inferno", label="mask", alpha=0.5)
        ax.set_title("True Segmentation")
        ax.set_xlabel(slice_label(i, mids))

        # plot predicted segmentations on true img
        pred_ax.imshow(img, cmap="inferno", label="img")
        pred_ax.imshow(pred, cmap="inferno", label="pred", alpha=0.5)
        pred_ax.set_xlabel(slice_label(i, mids))
        pred_ax.set_title("Predicted Segmentation")
    plt.show()
