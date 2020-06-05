import nibabel as nib
import os
import torchio as tio
import sys

from glob import glob
from pathlib import Path
from time import ctime
from torch.utils.data import DataLoader
from warnings import filterwarnings

from train.augment import compose_transforms
from train.load import COMPUTE_CANADA, IN_COMPUTE_CAN_JOB, get_cc539_subjects

print(f"{ctime()}:  Creating Dataset...")
subjects = get_cc539_subjects()
transform = compose_transforms()
subj_dataset = tio.ImagesDataset(subjects, transform=transform)
# batch size has to be 1 for variable-sized inputs
print(f"{ctime()}:  Creating DataLoader...")
training_loader = DataLoader(subj_dataset, batch_size=1, num_workers=8)

filterwarnings("ignore", message="Image.*has negative values.*")
for i, subjects_batch in enumerate(training_loader):
    inputs = subjects_batch["img"][tio.DATA]
    target = subjects_batch["label"][tio.DATA]
    print(f"{ctime()}:  Got subject img and mask {i}")
    if COMPUTE_CANADA and not IN_COMPUTE_CAN_JOB:  # don't run much on node
        if i > 5:
            sys.exit(0)
