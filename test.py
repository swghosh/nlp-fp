import os
os.environ['PJRT_DEVICE']='TPU'

import torch
from torch.utils.data import DataLoader
import torch_xla.core.xla_model as xm
from torchvision import datasets, transforms

device = xm.xla_device()


# Define data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load CIFAR-10 dataset
train_data = datasets.CIFAR10('~/.torch/datasets', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=10240, shuffle=True)

# Define model class (same as previous example)
class MNISTModel(torch.nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)  # Adjust for CIFAR-10 channels
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create PyTorch model and optimizer
model = MNISTModel().to(device)
optimizer = torch.optim.Adam(model.parameters())

# Loss function (CrossEntropyLoss for multi-class classification)
loss_fn = torch.nn.CrossEntropyLoss()

i=0
# Training loop
for epoch in range(5):
  for data, target in train_loader:
    
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    #print(output.shape)
    #print(target.shape)
    
    loss = loss_fn(output, target)
    loss.backward()
    #optimizer.step()
    xm.optimizer_step(optimizer)  # Use XLA optimizer step
    if (i+1) % 100 == 0:
        print(f"Epoch: {epoch+1} [{i+1}/{len(train_loader)} ({100. * (i+1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    i+=1


       
print('Finished Training')