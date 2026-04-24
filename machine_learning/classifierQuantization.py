# Project: MNIST Classifier + Quantization (Laptop-Friendly)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# Device (CPU only)
device = torch.device("cpu")

# Data
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000
)
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SmallNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

print("FP32 Accuracy:", evaluate(model))
def measure_latency(model, runs=100):
    model.eval()
    dummy = torch.randn(1, 1, 28, 28)
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(dummy)
    return (time.time() - start) / runs

fp32_latency = measure_latency(model)
print("FP32 Latency:", fp32_latency)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
print("INT8 Accuracy:", evaluate(quantized_model))

int8_latency = measure_latency(quantized_model)
print("INT8 Latency:", int8_latency)