from .colab import Colab
from .config import Config
from .data import FlareData
from .evaluation import inference, evaluate, plot_roc, plot_loss
from .training import WarmupScheduler, train, CrossValidation
from .utils import clear_cache, device, ignore_warnings, fix_random_seed, log_as_json

__version__ = "0.0.1"
__author__ = "Jin"
__license__ = "MIT License"
__description__ = "Easily Train Pytorch in Colab."
__all__ = [
    Colab,
    Config,
    FlareData,
    inference,
    evaluate,
    plot_roc,
    plot_loss,
    WarmupScheduler,
    train,
    CrossValidation,
    clear_cache,
    device,
    ignore_warnings,
    fix_random_seed,
    log_as_json,
]
