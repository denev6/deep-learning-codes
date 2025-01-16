import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

labels = {
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

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 10)
state_dict = torch.load("./static/model.pth", map_location=my_device, weights_only=True)
model.load_state_dict(state_dict)
model.to(my_device)
model.eval()


def convert_image(img_path):
    image = Image.open(img_path).convert("L")
    input_tensors = transform(image)
    input_tensor = input_tensors.unsqueeze(0)
    return input_tensor


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    prob = exp_logits / np.sum(exp_logits)
    return prob


def predict(img_tensor):
    img_tensor = img_tensor.to(my_device)
    with torch.no_grad():
        output = model(img_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    label_idx = torch.argmax(probs).item()
    label = labels[label_idx]
    prob = probs[label_idx]
    return label, prob
