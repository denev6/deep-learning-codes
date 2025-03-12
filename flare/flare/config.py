from dataclasses import dataclass, field
from typing import Optional

from .utils import _time_stamp, _format_name


@dataclass()
class Config:
    """Configuration for training and Log with 'flare.log'.

    :param name: name of the experiment
    :param batch: batch size
    :param epochs: number of epochs
    :param lr: learning rate
    :param enable_fp16: whether to use FP16 precision
    :param grad_step: number of gradient accumulation steps
    :param patience: patience for early stopping

    Example:
    >>> config = Config(name="CNN", batch=4, epochs=5, lr=0.01)
    >>> config.add(optimizer="SGD")
    >>> config.optimizer
    'SGD'
    """

    name: str
    batch: int
    epochs: int
    lr: float
    enable_fp16: Optional[bool] = field(default=False)
    grad_step: Optional[int] = field(default=1)
    patience: Optional[int] = field(default=0)

    # Will be filled automatically
    id: str = field(init=False)
    model_path: str = field(init=False)

    def __post_init__(self):
        self.id = _time_stamp()
        self.name = _format_name(self.name)
        self.model_path = f"{self.name}_{self.id}.pt"

        if self.batch < 1:
            raise ValueError("batch must be positive integer.")
        if self.epochs < 0:
            raise ValueError("epochs must be positive integer.")
        if self.lr < 0:
            raise ValueError("lr must be positive float.")

    def add(self, **kwargs):
        """Add extra attributes

        :raises KeyError: if the key already exists in the config.
        """
        for k, v in kwargs.items():
            if k in self.__dict__.keys():
                raise KeyError(f"Duplicate key: {k}")
            object.__setattr__(self, k, v)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        pairs = self.to_dict().items()
        pairs = ", ".join([f"{k}={v}" for k, v in pairs])
        name = str(self.__class__.__name__)
        return f"{name}({pairs})"