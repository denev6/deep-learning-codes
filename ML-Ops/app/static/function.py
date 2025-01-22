import os

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

MY_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "static/model.pth"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tensor2pil = transforms.ToPILImage()

# Load torch model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 10)
state_dict = torch.load(MODEL_PATH, map_location=MY_DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.to(MY_DEVICE)
model.eval()


def convert_image(img_file) -> torch.Tensor:
    """Convert an image to shape of MobileNet input"""
    img = Image.open(img_file).convert("L")
    img_tensor = transform(img)
    # "Normalize" in "transform" requires Tensor
    # So it returns Tensor, not Image.
    return img_tensor


def save_image(img, img_dir: str, img_name: str) -> str:
    """Save compressed image as .jpg"""
    if isinstance(img, torch.Tensor):
        img = tensor2pil(img)
    file_name, _ = img_name.rsplit(".", 1)
    file_name = f"{file_name}.jpg"
    file_path = os.path.join(img_dir, file_name)
    img.save(file_path, "jpeg", quality=70)
    return file_name


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    prob = exp_logits / np.sum(exp_logits)
    return prob


def predict(img_tensor: torch.Tensor):
    """Predict label of an image"""
    if img_tensor.dim() == 3:
        # Add dim for batch
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(MY_DEVICE)

    with torch.no_grad():
        output = model(img_tensor)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    label_idx = torch.argmax(probs).item()
    label = LABELS[label_idx]
    prob = probs[label_idx].item()
    return label, prob
