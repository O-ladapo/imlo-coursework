import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from model import PetNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PetDatasetWithTrimap(Dataset):
    def __init__(self, split):
        self.base = datasets.OxfordIIITPet(
            root="data",
            split=split,
            target_types=("category", "segmentation"),
            download=True
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, (label, trimap) = self.base[idx]

        img = TF.resize(img, (128, 128))
        trimap = TF.resize(
            trimap, (128, 128),
            interpolation=transforms.InterpolationMode.NEAREST
        )

        img = TF.to_tensor(img)
        img = TF.normalize(img,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        trimap = TF.pil_to_tensor(trimap).float()
        mask = ((trimap == 1) | (trimap == 3)).float()
        img = img * mask

        return img, label

test_dataset = PetDatasetWithTrimap(split="test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

net = PetNet().to(device)
net.load_state_dict(torch.load('trained_net.pth', map_location=device))
net.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
