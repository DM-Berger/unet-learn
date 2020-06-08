import nibabel as nib
import os
import torch
import torch.nn.functional as F
import torchio as tio
import sys

from glob import glob
from pathlib import Path
from pytorch_lightning.core.lightning import LightningModule
from time import ctime
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from warnings import filterwarnings

from lightning.log import visualize
from model.unet import UNet3d
from train.augment import compose_transforms
from train.load import COMPUTE_CANADA, IN_COMPUTE_CAN_JOB, get_cc539_subjects


class LightningUNet3d(LightningModule):
    def __init__(
        self,
        initial_features: int = 32,
        depth: int = 3,
        n_labels: int = 2,
        normalization: bool = True,
        batch_size: int = 1,
    ):
        super().__init__()
        self.unet = UNet3d(
            initial_features, depth=depth, n_labels=n_labels, normalization=normalization
        )
        self.batch_size = batch_size

    def forward(self, x: Tensor) -> Tensor:
        return self.unet.forward(x)

    def training_step(self, batch, batch_idx):
        img = batch["img"][tio.DATA]
        img = F.interpolate(img, size=(128, 128, 128))
        target = batch["label"][tio.DATA]
        target = F.interpolate(target, size=(128, 128, 128))
        prediction = self(img)
        if int(batch_idx) % 50 == 0:
            visualize(img, target, prediction)
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        print(f"{ctime()}:  Creating Dataset...")
        subjects = get_cc539_subjects()
        transform = compose_transforms()
        subj_dataset = tio.ImagesDataset(subjects, transform=transform)
        print(f"{ctime()}:  Creating DataLoader...")
        training_loader = DataLoader(subj_dataset, batch_size=self.batch_size, num_workers=8)
        return training_loader

    def prepare_data(self):
        transform = compose_transforms()
