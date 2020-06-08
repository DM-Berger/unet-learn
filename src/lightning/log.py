from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger

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