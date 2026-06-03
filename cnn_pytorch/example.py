import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. SETUP HARDWARE ACCELERATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA PIPELINE (Downloading & Loading)
# ==========================================
# Transforms convert images to PyTorch Tensors and normalize pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Mean and Std Dev for grayscale
])

# Download the training and testing sets automatically
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaders handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ==========================================
# 3. DEFINE THE CNN ARCHITECTURE
# ==========================================
class MyFirstCNN(nn.Module):
    def __init__(self):
        super(MyFirstCNN, self).__init__()

        # Conv Block 1: Expects 1 input channel (grayscale), applies 16 filters of size 3x3
        # Output spatial size stays 28x28 because padding=1 compensates for the 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # MaxPool halves width and height: 28x28 becomes 14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: Takes 16 channels, outputs 32 channels.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # MaxPool halves width and height again: 14x14 becomes 7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected (Linear) Layers for classification
        # The data entering this layer has 32 feature maps of size 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        # Pass through Conv Block 1
        x = self.pool1(self.relu1(self.conv1(x)))

        # Pass through Conv Block 2
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the 3D tensor (Channels, Height, Width) into a 1D vector
        # x.size(0) keeps the batch size intact, -1 flattens everything else
        x = x.view(x.size(0), -1)

        # Pass through Fully Connected layers
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate model and send its parameters to your CPU/GPU hardware
model = MyFirstCNN().to(device)

# ==========================================
# 4. LOSS FUNCTION AND OPTIMIZER
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
epochs = 3
print("\n--- Starting Training ---")

for epoch in range(epochs):
    model.train()  # Put model in training mode
    running_loss = 0.0

    for images, labels in train_loader:
        # Push tensors to the same hardware device as the model
        images, labels = images.to(device), labels.to(device)

        # Zero out the gradient tracking from the last step
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass (PyTorch calculates all derivatives automatically)
        loss.backward()

        # Update our weight matrices/filters
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {epoch_loss:.4f}")

# ==========================================
# 6. EVALUATION (Testing the Model)
# ==========================================
print("\n--- Evaluating Model ---")
model.eval()  # Put model in evaluation mode (turns off dropout/batchnorm updates)
correct = 0
total = 0

# Turn off gradient tracking to save memory and compute power during evaluation
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Get the index of the highest logit value (this is the predicted digit)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy on 10,000 images: {accuracy:.2f}%")
