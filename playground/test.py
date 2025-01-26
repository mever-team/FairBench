import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

# Data
X = torch.tensor(
    [[3, 0, 0, 2], [1, 2, 0, 2], [1, 2, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]
    + [
        [9, 0, 0, 6],
        [3, 6, 0, 6],
        [3, 6, 3, 3],
        [3, 3, 3, 3],
        [3, 3, 0, 3],
        [0, 3, 3, 3],
    ],
    dtype=torch.float32,
)
y = torch.tensor([1, 0.5, 0.25, 0, 0.25, 0.25] * 2, dtype=torch.float32).view(-1, 1)

Xtest = torch.tensor(
    [
        [6, 0, 0, 4],
        [2, 4, 0, 4],
        [2, 4, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 0, 2],
        [0, 2, 2, 2],
    ],
    dtype=torch.float32,
)
ytest = torch.tensor([1, 0.5, 0.25, 0, 0.25, 0.25], dtype=torch.float32).view(-1, 1)


# Neural Network Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(2, 16)
        self.hidden2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def mlp(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return self.sigmoid(x)

    def forward(self, x):
        # return self.mlp(x)
        x1, x2 = x[:, :2], x[:, 2:]
        x1 = self.mlp(x1)
        x2 = self.mlp(x2)
        return 1 - torch.minimum(x1, x2) / torch.maximum(x1, x2)


# Initialize model, loss, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optional: print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Predict
with torch.no_grad():
    y_pred = model(Xtest)


# Convert to list for comparison with sklearn
y_pred_list = y_pred.squeeze().tolist()
print(y_pred_list)

# Calculate Mean Squared Error
msqrt = mean_squared_error(ytest.tolist(), y_pred_list) ** 0.5
print(f"msqrt: {msqrt}")


import numpy as np
import matplotlib.pyplot as plt

# Generate example inputs for visualization
x1_range = np.linspace(1, 10, 10)
x2_range = np.linspace(1, 10, 10)
x1, x2 = np.meshgrid(x1_range, x2_range)
inputs = torch.tensor(np.column_stack([x1.ravel(), x2.ravel()]), dtype=torch.float32)

# Get predictions from the model
with torch.no_grad():
    outputs = model.mlp(inputs).numpy()

# Reshape outputs for plotting
outputs_reshaped = outputs.reshape(x1.shape)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x1, x2, outputs_reshaped, cmap="viridis", edgecolor="none")
ax.set_title("3D Plot of Model's MLP Output")
ax.set_xlabel("Positives")
ax.set_ylabel("Negatives")
ax.set_zlabel("Base measure")

plt.show()
