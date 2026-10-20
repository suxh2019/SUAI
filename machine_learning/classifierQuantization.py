# Project: MNIST Classifier + Quantization (Laptop-Friendly)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

'''
(1) Demonstrate Dynamic Quantization in PyTorch, a model optimization technique that reduces the memory usage of a neural network 
and can improve CPU inference speed while maintaining similar prediction accuracy.
(2) Train a neural network on the MNIST handwritten digit dataset, then compare its performance before and after converting
 its linear layers from FP32 (32-bit floating-point) to INT8 (8-bit integer) quantization.
(3) Train a neural network using FP32 weights, then quantize its linear layers to INT8 without retraining.
'''

# Device (CPU only)
device = torch.device("cpu")

# Data
# Convert images into PyTorch tensors.Before conversion, images contain pixel intensity values from 0 to 255. 
# After ToTensor(), values are represented as floating-point numbers between 0 and 1.
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

# Define a small model; fully connected feedforward neural network, also called a multilayer perceptron (MLP).
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Converts a 2D image to a vector
            nn.Flatten(),
            #Takes 784 input features and produces 128 output features.
            nn.Linear(28*28, 128),
            #ReLU adds nonlinearity, allowing the neural network to learn more complicated relationships.
            nn.ReLU(),
            # Reduces 128 features to 64
            nn.Linear(128, 64),
            nn.ReLU(),
            # Produces 10 values, one for each digit.
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)
#This creates the neural network and places its parameters on the CPU.
model = SmallNet().to(device)

#Train (fast: 1–2 epochs)
#Cross-entropy measures how different the model's predictions are from the true labels.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        #Before each forward pass, optimizer.zero_grad() clears previously accumulated gradients.
        optimizer.zero_grad()
        #Forward pass Predict digits from input images
        output = model(data)
        # Compute loss: measure prediction error
        loss = criterion(output, target)
        # Calculate gradients of the weights
        loss.backward()
        # Use Adam to adjust weights
        optimizer.step()
# Accuracy function
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    # disables gradient tracking
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

print("FP32 Accuracy:", evaluate(model))

# Measure latency
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
# This creates a dynamically quantized copy of the trained model.
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Evaluate
print("INT8 Accuracy:", evaluate(quantized_model))

int8_latency = measure_latency(quantized_model)
print("INT8 Latency:", int8_latency)