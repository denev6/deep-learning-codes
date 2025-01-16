import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
import mlflow

from mobilenet import transform, labels, my_device

# arguments 가져오기
args = argparse.ArgumentParser()
args.add_argument("--batch_size", type=int)
args.add_argument("--learning_rate", type=float)
args.add_argument("--epoch", type=int)
parsed_args = args.parse_args()

batch_size = parsed_args.batch_size
learning_rate = parsed_args.learning_rate
n_epoch = parsed_args.epoch

print(f"Current Device: {my_device}")
print("Downloading Dataset...")

# 데이터 준비
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 준비
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model = model.to(my_device)

criterion = nn.CrossEntropyLoss()
criterion.to(my_device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습
print("Training...")
for epoch in range(n_epoch):
    model.train()
    train_loss = 0

    for batch_id, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(my_device), labels.to(my_device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"{epoch + 1}. Loss: {train_loss:.6f}")

# torch.save(model.state_dict(), model_path)

# 모델 검증
print("Evaluating...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(my_device), labels.to(my_device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.3f}%")

# MLFlow 등록
print("Uploading current model...")

with mlflow.start_run():
    # mlflow.log_param("Learning-rate", learning_rate)
    # mlflow.log_param("Batch-size", batch_size)
    # mlflow.log_param("Epoch", n_epoch)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.pytorch.log_model(model, "model")
