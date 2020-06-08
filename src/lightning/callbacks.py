from pathlib import Path
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def checkpointer(checkdir: Path, prefix: str, monitor: str = "train_loss") -> ModelCheckpoint:
    """Setup up a ModelCheckpoint callback.

    Parameters
    ----------
    checkdir: Path
        Directory where checkpoints will be saved.

    prefix: str
        Identifier to prepend to checkpoint filename.

    monitor: str
        Quanitity to monitor
    """
    suffix = "{epoch}__{train_loss:.3f}-{val_less:.3f}"
    fullpath = checkdir.resolve() / prefix
    filepath = str(fullpath) + suffix

    return ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        verbose=True,
        save_top_k=3,
        mode="auto",
        save_weights_only=False,
        period=1,
    )
