import torch
from torch import autocast
from torch.amp import GradScaler
from sklearn.model_selection import KFold
from tqdm.auto import tqdm, trange

from data import FlareData


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

        Args:
            loss: Current validation loss.
            model: Model to save (it will compare the model with prior saved model and save if better).
            epoch: current epoch (will be used as check point if needed).

        Returns:
            True if training should stop, False otherwise.
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

        check point index
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

    Args:
        optimizer: Optimizer for training.
        lr: Learning rate.
        min_lr: Minimum learning rate.
        warmup_steps: Number of warmup steps.
        decay_factor: Factor to multiply learning rate when loss increases.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        min_lr: float = 1e-6,
        warmup_steps: int = 10,
        decay_factor: float = 0.1,
    ):
        self.optimizer = optimizer
        self.initial_lr = lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor

        # If user set warmup_steps=0, then set warmup_steps=1 to avoid ZeroDivisionError.
        self.warmup_steps = max(warmup_steps, 1)

        assert (
            self.initial_lr >= self.min_lr
        ), f"Learning rate must be greater than min_lr({self.min_lr})"
        assert 0 < self.decay_factor <= 1, "Decay factor must be less than 1.0."

        self.global_step = 1
        self.best_loss = float("inf")

        # Initialize learning rates
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr * (self.global_step / self.warmup_steps)

    def step(self, loss: float):
        """Update learning rate based on current loss."""
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


def _validate(model, device, criterion, validation_loader, enable_amp, amp_precision):
    """Validate the model on validation set."""
    model.eval()
    val_loss = 0

    with torch.no_grad():
        with autocast(device_type=str(device), enabled=enable_amp, dtype=amp_precision):
            for data, label in validation_loader:
                data = data.to(device)
                label = label.to(device)
                output = model(data)

            batch_loss = criterion(output, label)
            val_loss += batch_loss.item()

        return val_loss / len(validation_loader)


def _train(
    model: torch.nn.Module,
    device: torch.device,
    model_path: str,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    train_set: FlareData,
    validation_set: FlareData,
    gradient_step: int,
    patience: int,
    enable_amp: bool,
    amp_precision: torch.dtype,
    scheduler,
):
    """Train the model

    Returns:
        check point, the best loss, list of (train loss, val loss)
    """
    if scheduler is not None:
        assert callable(
            getattr(scheduler, "step", None)
        ), "Scheduler must have a step() method."

    epoch_trange = trange(1, epochs + 1)
    early_stopper = EarlyStopping(patience, model_path)
    scaler = GradScaler(device=str(device), enabled=enable_amp)
    loss_history = list()

    train_loader = train_set.to_loader()
    val_loader = validation_set.to_loader()

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
                # Compute loss with given precision
                output = model(data)
                batch_loss = criterion(output, label)

            train_loss += batch_loss.item()

            # Scale loss to prevent under/overflow
            scaler.scale(batch_loss / gradient_step).backward()

            # Gradient Accumulation
            if batch_id % gradient_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Validate Training Epoch
        train_loss /= len(train_loader)
        val_loss = _validate(
            model, device, criterion, val_loader, enable_amp, amp_precision
        )
        tqdm.write(
            f"Epoch {epoch_idx}, Train-Loss: {train_loss:.5f},  Val-Loss: {val_loss:.5f}"
        )
        loss_history.append((train_loss, val_loss))

        # Early stopping
        if early_stopper.should_stop(val_loss, model, epoch_idx):
            break

        # Learning Rate Scheduling
        if scheduler is not None:
            if isinstance(scheduler, WarmupScheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step(epoch_idx)

    return early_stopper.check_point, early_stopper.best_loss, loss_history


def _decode_precision(precision: str, device):
    precision = precision.strip().lower()
    amp_options = {
        "fp32": (torch.float32, False),
        "fp16": (torch.float16, True),
        "bf16": (torch.bfloat16, True),
    }
    if precision not in amp_options.keys():
        raise KeyError(f"Precision must be one of {amp_options.keys()}.")
    amp_precision, enable_amp = amp_options[precision]

    if enable_amp:
        assert torch.amp.autocast_mode.is_autocast_available(
            str(device)
        ), "Unable to use autocast on current device. Use 'fp32'(default) instead."

    return enable_amp, amp_precision


def train(
    model: torch.nn.Module,
    device: torch.device,
    model_path: str,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    train_set: FlareData,
    validation_set: FlareData,
    gradient_step: int = 1,
    patience: int = 0,
    precision: str = "fp32",
    scheduler=None,
):
    """Train the model and return the best check point.

    Batch accumulation, Early stopping, Warmup scheduler,
    and Learning rate scheduler are included.

    Args:
        model: Model to train.
        model_path: Path to save the best model.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Torch device (cpu or cuda).
        epochs: Maximum number of epochs.
        train_set: Training data set in FlareData format.
        validation_set: Validation data set in FlareData format.
        gradient_step: Set gradient_step=1 to disable gradient accumulation.
        patience: Number of epochs to wait before early stopping (default: 0).
        precision: Enable Auto mixed precision training (fp32, fp16, bf16) (default: fp32).
        scheduler: Learning rate scheduler (default: None).

    Returns:
        tuple:
            - check point (int)
            - best validation loss (float)
            - list of (train loss, val loss)
    """
    enable_amp, amp_precision = _decode_precision(precision, device)
    check_point, best_loss, loss_history = _train(
        model,
        device,
        model_path,
        optimizer,
        criterion,
        epochs,
        train_set,
        validation_set,
        gradient_step,
        patience,
        enable_amp,
        amp_precision,
        scheduler,
    )
    return check_point, best_loss, loss_history


class CrossValidation:
    def __init__(self, train_data: FlareData, n_splits: int, shuffle: bool = True):
        self._dataset = train_data.dataset
        self._best_val_loss = float("inf")
        self._best_returns = None

        kf = KFold(n_splits=n_splits, shuffle=shuffle)
        self._indices_generator = kf.split(self._dataset)

        self._current_fold = -1
        self._train_set = None
        self._val_set = None

    def __enter__(self):
        return self

    def __iter__(self):
        for fold_idx, (train_idx, val_idx) in enumerate(
            self._indices_generator, start=1
        ):
            self._current_fold = fold_idx
            self._train_set = self._dataset.subset(train_idx)
            self._val_set = self._dataset.subset(val_idx)
            yield fold_idx

    def train(
        self,
        model: torch.nn.Module,
        device: torch.device,
        model_path: str,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epochs: int,
        gradient_step: int = 1,
        patience: int = 0,
        precision: str = "fp32",
        scheduler=None,
    ):
        model_name, ext = model_path.rsplit(".", 1)
        model_path = f"{model_name}_{self._current_fold}.{ext}"

        check_point, best_loss, loss_hist = train(
            model,
            device,
            model_path,
            optimizer,
            criterion,
            epochs,
            self._train_set,
            self._val_set,
            gradient_step,
            patience,
            precision,
            scheduler,
        )

        if best_loss < self._best_val_loss:
            self._best_val_loss = best_loss
            self._best_returns = (
                self._current_fold,
                check_point,
                best_loss,
                loss_hist,
                model_path,
            )

    @property
    def result(self):
        """Return the best check point.

        Returns:
            - Best fold index (int)
            - Best check point (int)
            - Best validation loss (float)
            - Loss history (list of (train loss, val loss))
            - Model path (str)
        """
        return self._best_returns
