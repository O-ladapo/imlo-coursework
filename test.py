import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import PetNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((128,128)), 
    transforms.ToTensor()
])

test_dataset = datasets.OxfordIIITPet(
    root = "data",
    split = "test",
    target_types = "category",
    download = True,
    transform = test_transform
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle= False) 

net = PetNet().to(device)
net.load_state_dict(torch.load('trained_net.pth', map_location=device))

correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in test_loader:
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')
