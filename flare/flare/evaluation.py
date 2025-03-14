import torch
from torch import autocast
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
import matplotlib.pyplot as plt

from .data import FlareData
from .training import _decode_precision


def inference(
    model: torch.nn.Module,
    device: torch.device,
    test_set: FlareData,
    precision: str = "fp32",
):
    enable_amp, amp_precision = _decode_precision(precision, device)
    data_loader = test_set.to_loader()

    with torch.no_grad():
        with autocast(device_type=str(device), enabled=enable_amp, dtype=amp_precision):
            model.to(device)
            model.eval()
            y_pred = list()
            y_true = list()

            for data, label in data_loader:
                data = data.to(device)
                output = model(data)
                probs = F.softmax(output.float(), dim=1)
                y_pred.extend(probs.detach().cpu().numpy())
                y_true.extend(label.numpy())

            return y_pred, y_true


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    test_set: FlareData,
    precision: str = "fp32",
):
    y_pred, y_true = inference(model, device, test_set, precision)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "f1-score": f1,
        "recall": recall,
        "auc": auc_value,
    }


def plot_roc(
    model: torch.nn.Module,
    device: torch.device,
    test_set: FlareData,
    precision: str = "fp32",
    title: str = "ROC Curve",
):
    y_pred, y_true = inference(model, device, test_set, precision)
    fpr, tpr, _ = roc_curve(y_true, np.array(y_pred)[:, 1])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")  # Baseline

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid()
    plt.show()


def plot_loss(train_loss, validation_loss, check_point, title="Loss Curve"):
    assert len(train_loss) == len(validation_loss), "Loss length mismatch."
    x_range = np.arange(1, check_point + 1)

    plt.figure(figsize=(6, 6))
    plt.plot(train_loss[:check_point], label="Train Loss")

    # Train Loss
    plt.plot(
        x_range[:check_point],
        train_loss[:check_point],
        "-",
        color="blue",
        label="Train Loss",
    )
    plt.plot(
        x_range[check_point - 1 :], train_loss[check_point - 1 :], "--", color="blue"
    )
    plt.scatter(check_point, train_loss[check_point - 1], color="blue")

    # Validation Loss
    plt.plot(
        x_range[:check_point],
        validation_loss[:check_point],
        "-",
        color="red",
        label="Validation Loss",
    )
    plt.plot(
        x_range[check_point - 1 :],
        validation_loss[check_point - 1 :],
        "--",
        color="red",
    )
    plt.scatter(check_point, validation_loss[check_point - 1], color="red")

    plt.title(title)
    plt.legend()
    plt.show()
