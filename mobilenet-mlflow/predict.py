import argparse

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

from mobilenet import transform, labels, my_device

# arguments 가져오기
args = argparse.ArgumentParser()
args.add_argument("--data_path", type=str)
args.add_argument("--model_path", type=str)
parsed_args = args.parse_args()

image_path = parsed_args.data_path
model_path = parsed_args.model_path

# 이미지 불러오기
image = Image.open(image_path)
input_tensors = transform(image)
input_tensor = input_tensors.unsqueeze(0)
input_tensor.to(my_device)

# 모델 불러오기
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 10) 
model.load_state_dict(torch.load(model_path, map_location=my_device, weights_only=True))
model.eval()

# 결과 예측
with torch.no_grad():
    output = model(input_tensor)

_, predicted = torch.max(output, 1)
print(f"Predicted: {labels[predicted.item()]}")
