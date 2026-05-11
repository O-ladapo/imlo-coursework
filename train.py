import torch
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

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)
print("First label:", labels[0])