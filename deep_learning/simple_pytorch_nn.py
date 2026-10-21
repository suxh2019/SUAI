import torch
import torch.nn as nn
import torch.optim as optim

# binary classification using a neural network in PyTorch; Learn the logical OR function
# --------------------------------------------------
# 1. Create simple training data
# --------------------------------------------------
torch.manual_seed(42)
# This creates a tensor containing four training examples.
# The first dimension represents four samples, and the second represents two features per sample.
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
# This network contains: (1) An input layer with 2 features. (2) One hidden layer with 4 neurons.
# (3) An output layer with 1 neuron. Neural network architecture: 2 → 4 → 1

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # This creates a linear layer with two input features and four output features.
        self.fc1 = nn.Linear(2, 4)
        # This creates a layer with four inputs and one output.
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        # ReLU introduces nonlinearity into the network. Without nonlinear activation functions, stacking 
        # linear layers would be equivalent to a single linear transformation.
        x = torch.relu(self.fc1(x))
        # Sigmoid converts a real-valued number into a value between 0 and 1.
        x = torch.sigmoid(self.fc2(x))
        return x


model = SimpleNN()

# Print number of trainable parameters
total_params = sum(
    p.numel() for p in model.parameters()
)
print("Total trainable parameters:", total_params)

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