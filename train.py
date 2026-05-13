import torch
import torch.optim as optim
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from model import PetNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Your device is: ", device)

class PetDatasetWithTrimap(Dataset):
    def __init__(self, split, augment=False):
        self.base = datasets.OxfordIIITPet(
            root="data",
            split=split,
            target_types=("category", "segmentation"),
            download=True
        )
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, (label, trimap) = self.base[idx]

        img = TF.resize(img, (128, 128))
        trimap = TF.resize(
            trimap, (128, 128),
            interpolation=transforms.InterpolationMode.NEAREST
        )

        if self.augment:
            if random.random() > 0.5:
                pad = 20
                img = TF.pad(img, pad)
                trimap = TF.pad(trimap, pad)
                i, j, h, w = transforms.RandomCrop.get_params(img, (128, 128))
                img = TF.crop(img, i, j, h, w)
                trimap = TF.crop(trimap, i, j, h, w)

            if random.random() > 0.5:
                img = TF.hflip(img)
                trimap = TF.hflip(trimap)

        img = TF.to_tensor(img)
        img = TF.normalize(img,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        trimap = TF.pil_to_tensor(trimap).float()
        mask = ((trimap == 1) | (trimap == 3)).float()
        img = img * mask

        return img, label

train_dataset = PetDatasetWithTrimap(split="trainval", augment=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

net = PetNet().to(device)
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(net.parameters(), weight_decay=7e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.005,
    steps_per_epoch=len(train_loader),
    epochs=30
)

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
        scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Loss: {running_loss / len(train_loader):.4f}')
    print(f'Training Accuracy: {100 * correct / total:.2f}%')
    print(f'LR: {scheduler.get_last_lr()[0]:.6f}')

torch.save(net.state_dict(), 'trained_net.pth')
