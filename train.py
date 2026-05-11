import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import PetNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Your device is: ", device)

transform = transforms.Compose([
    transforms.Resize((128,128)), 
    transforms.ToTensor()
])

train_dataset = datasets.OxfordIIITPet(
    root = "data",
    split = "trainval",
    target_types = "category",
    download = True,
    transform = transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle= True) 

net = PetNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):
    print(f'Training epoch {epoch}')

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Loss: {running_loss / len(train_loader):.4f}')
