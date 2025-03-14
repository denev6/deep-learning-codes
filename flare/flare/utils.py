from datetime import datetime
import gc
import warnings
import os
import json
from dataclasses import is_dataclass, asdict

import torch
import numpy as np

from .config import Config


def _time_stamp():
    current_time = datetime.now()
    return current_time.strftime("%y%m%d%H%M%S%f")


def _format_name(name: str, max_len: int = 30) -> str:
    name = name.strip().lower().replace(" ", "-")
    return name[:max_len]


def _obj_to_dict(obj):
    # Do not use 'asdict' since 'flare.config.Config' allows extra values
    # which are not included in dataclasses.fields.
    if isinstance(obj, Config):
        return obj.to_dict()
    elif is_dataclass(obj):
        return asdict(obj)
    else:
        assert isinstance(obj, dict), "Object must be a dictionary or dataclass."
        return obj


def _safe_update_dict(d, k, v):
    if k in d:
        raise KeyError(f"Duplicate key: {k}")
    if v is not None:
        d[k] = v


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def device(force_cuda=True) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if force_cuda:
        assert has_cuda, "CUDA is not available."
        return torch.device("cuda")
    return torch.device("cuda") if has_cuda else torch.device("cpu")


def ignore_warnings():
    warnings.filterwarnings("ignore")


def fix_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_as_json(file_path, *args, **kwargs):
    """Save logs to a JSON file. Dictionary or dataclass is accepted.

    Example:
    >>> log_as_json("log.json", {"tag": "flare"}, config=Config(name="CNN", batch=4, ...))
    {"tag": "flare","config": {"name": "CNN","batch": 4,}}
    """
    _, ext = os.path.splitext(file_path)
    assert ext == ".json", "File path must be a JSON file."

    logs = dict()

    # When raw dictionary is passed
    # i.e. log_as_json({"key": "value"})
    for arg in args:
        arg = _obj_to_dict(arg)
        for name, attr in arg.items():
            _safe_update_dict(logs, str(name).strip(), attr)

    # keyword arguments
    for name, value in kwargs.items():
        value = _obj_to_dict(value)
        _safe_update_dict(logs, str(name).strip(), value)

    with open(file_path, "w") as f:
        json.dump(logs, f, indent=2)

    return logs
