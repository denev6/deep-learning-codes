import torch
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from tqdm.auto import tqdm, trange

from model import FlareModule


class EarlyStopping(object):
    """Stop training when loss does not decrease"""

    def __init__(self, patience: int, path_to_save: str):
        self._min_loss = float("inf")
        self._patience = patience
        self._path = path_to_save
        self.__check_point = None
        self.__counter = 0

    def should_stop(self, loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """Check if training should stop and save the check point if needed.

        :param loss: Current validation loss.
        :param model: Model to save (it will compare the model with prior saved model and save if better).
        :param epoch: current epoch (will be used as check point if needed).
        :return: True if training should stop, False otherwise.
        """
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
            self.__check_point = epoch
            torch.save(model.state_dict(), self._path)
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter == self._patience:
                return True
        return False

    def load(self, weights_only=True):
        """Load best model weights"""
        return torch.load(self._path, weights_only=weights_only)

    @property
    def check_point(self):
        """Return check point index

        :return: check point index
        """
        if self.__check_point is None:
            raise ValueError("No check point is saved!")
        return self.__check_point

    @property
    def best_loss(self):
        return self._min_loss


class WarmupScheduler(object):
    """Warmup learning rate and dynamically adjusts learning rate based on validation loss.

    When the loss increases, the learning rate will be divided by decay_factor.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        min_lr: float = 1e-6,
        warmup_steps: int = 10,
        decay_factor: float = 0.1,
    ):
        """Initialize Warmup Scheduler.

        :param optimizer: Optimizer for training.
        :param lr: Learning rate.
        :param min_lr: Minimum learning rate.
        :param warmup_steps: Number of warmup steps.
        :param decay_factor: Factor to multiply learning rate when loss increases.
        """
        self.optimizer = optimizer
        self.initial_lr = lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor

        # If user set warmup_steps=0, then set warmup_steps=1 to avoid ZeroDivisionError.
        self.warmup_steps = max(warmup_steps, 1)

        assert (
            self.initial_lr >= self.min_lr
        ), f"Learning rate must be greater than min_lr({self.min_lr})"
        assert 0 < self.decay_factor < 1, "Decay factor must be less than 1.0."

        self.global_step = 1
        self.best_loss = float("inf")

        # Initialize learning rates
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr * (self.global_step / self.warmup_steps)

    def step(self, loss: float):
        """Update learning rate based on current loss.

        :param loss: Current validation loss.
        """
        self.global_step += 1

        if self.global_step <= self.warmup_steps:
            # Linear warmup
            warmup_lr = self.initial_lr * (self.global_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            # Check if loss increased
            if loss > self.best_loss:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group["lr"] * self.decay_factor, self.min_lr)
                    param_group["lr"] = new_lr
            self.best_loss = min(self.best_loss, loss)

    def get_lr(self):
        """Return current learning rates."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]



def validate(model, device, criterion, val_loader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)

            batch_loss = criterion(output, label)
            val_loss += batch_loss.item()

        return val_loss / len(val_loader)


def _train(
    model: FlareModule,
    device: torch.device,
    model_path: str,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    gradient_step: int,
    patience: int,
    enable_amp: bool,
    amp_precision: torch.dtype,
):
    scheduler = model.scheduler()
    if scheduler is not None:
        assert callable(
            getattr(scheduler, "step", None)
        ), "Scheduler must have a step() method."

    epoch_trange = trange(1, epochs + 1)
    early_stopper = EarlyStopping(patience, model_path)
    scaler = GradScaler(device=str(device), enabled=enable_amp)
    criterion = model.loss_fn()
    optimizer = model.optimizer()

    model.to(device)

    model.zero_grad()

    for epoch_idx in epoch_trange:
        model.train()
        train_loss = 0
        for batch_id, (data, label) in enumerate(train_loader, start=1):
            data = data.to(device)
            label = label.to(device)

            with autocast(
                device_type=str(device), enabled=enable_amp, dtype=amp_precision
            ):
                output = model(data)
                batch_loss = model.loss(output, label)

            train_loss += batch_loss.item()

            # Scale loss to prevent under/overflow
            scaler.scale(batch_loss / gradient_step).backward()

            # Gradient Accumulation
            if batch_id % gradient_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            val_loss = validate(model, device, criterion, val_loader)

        # Validate Training Epoch
        train_loss /= len(train_loader)
        tqdm.write(
            f"Epoch {epoch_idx}, Train-Loss: {train_loss:.5f},  Val-Loss: {val_loss:.5f}"
        )

        # Early stopping
        if early_stopper.should_stop(val_loss, model, epoch_idx):
            break

        # Learning Rate Scheduling
        if scheduler is not None:
            if isinstance(scheduler, WarmupScheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step(epoch_idx)

    return early_stopper.check_point, early_stopper.best_loss


def train(
    model: FlareModule,
    device: torch.device,
    model_path: str,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    gradient_step: int = 1,
    patience: int = 0,
    precision: str = "fp32",
):
    """Train the model and return the best check point.

    Batch accumulation, Early stopping, Warmup scheduler,
    and Learning rate scheduler are included.

    :param model: Model to train.
    :param model_path: Path to save the best model.
    :param device: Torch device (cpu or cuda).
    :param epochs: Maximum number of epochs.
    :param train_loader: Training data loader.
    :param val_loader: Validation data loader.
    :param gradient_step: Set gradient_step=1 to disable gradient accumulation.
    :param patience: Number of epochs to wait before early stopping (default: 0).
    :param precision: Enable Auto mixed precision training (fp32, fp16, bf16) (default: fp32).
    """
    precision = precision.strip().lower()
    if precision == "fp32":
        enable_amp = False
        amp_precision = torch.float32
    elif precision == "fp16":
        enable_amp = True
        amp_precision = torch.float16
    elif precision == "bf16":
        enable_amp = True
        amp_precision = torch.bfloat16
    else:
        raise KeyError("Precision must be one of fp32, fp16 or bf16.")

    if enable_amp:
        assert torch.amp.autocast_mode.is_autocast_available(
            str(device)
        ), "Unable to use autocast on current device. Use fp32 instead."

    check_point, _ = _train(
        model,
        device,
        model_path,
        epochs,
        train_loader,
        val_loader,
        gradient_step,
        patience,
        enable_amp,
        amp_precision,
    )
    return check_point