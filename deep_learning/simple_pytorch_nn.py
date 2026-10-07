import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------
# 1. Create simple training data
# --------------------------------------------------

# Each row has 2 input features
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=torch.float32)

# Labels
y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [1.0],
], dtype=torch.float32)


# --------------------------------------------------
# 2. Define a neural network
# --------------------------------------------------

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model = SimpleNN()


# --------------------------------------------------
# 3. Define loss function and optimizer
# --------------------------------------------------

criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

epochs = 1000

for epoch in range(epochs):

    # Forward pass
    predictions = model(X)

    # Calculate loss
    loss = criterion(predictions, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, "
            f"Loss: {loss.item():.4f}"
        )


# --------------------------------------------------
# 5. Test the model
# --------------------------------------------------

with torch.no_grad():

    predictions = model(X)

    print("\nPredicted probabilities:")
    print(predictions)

    predicted_classes = (
        predictions >= 0.5
    ).float()

    print("\nPredicted classes:")
    print(predicted_classes)