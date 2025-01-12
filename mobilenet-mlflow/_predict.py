import argparse

from PIL import Image
import mlflow

from mobilenet import transform, labels, my_device

# arguments 가져오기
args = argparse.ArgumentParser()
args.add_argument("--data_path", type=str)
args.add_argument("--model_path", type=str)
parsed_args = args.parse_args()

image_path = parsed_args.data_path
model_path = parsed_args.model_path
logged_model = "runs:/ffb263113ee64d05928679ef5bcda173/MobileNet_model"

# 이미지 불러오기
image = Image.open(image_path)
input_tensors = transform(image)
input_tensor = input_tensors.unsqueeze(0)

# 모델 예측
loaded_model = mlflow.pytorch.load_model(logged_model, map_location=my_device)
output = loaded_model(input_tensor)
predicted = output.argmax()

print(f"Predicted: {labels[predicted.item()]}")
# Predicted: Ankle boot
