import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import PetNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Your device is: ", device)

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.OxfordIIITPet(
    root="data",
    split="trainval",
    target_types="category",
    download=True,
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

net = PetNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):
    print(f'Training epoch {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0

    net.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Loss: {running_loss / len(train_loader):.4f}')
    print(f'Training Accuracy: {100 * correct / total:.2f}%')

torch.save(net.state_dict(), 'trained_net.pth')
