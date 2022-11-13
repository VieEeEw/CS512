from typing import Tuple, Union, List, Dict
import torch
from torch import nn
import torchvision as tv
import torchvision.transforms as transforms

MODEL_PATH = "model.pth"
CIFAR10_CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
VGG_CONFIG: Dict[str, Tuple[Union[str, int]]] = {
    "VGG11": (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    "VGG16": (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'),
    "VGG19": (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M')
}
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def vgg_units(vgg_name: str) -> nn.Sequential:
    conf = VGG_CONFIG[vgg_name.upper()]
    layers: List[nn.Module] = []
    in_ch = 3
    for v in conf:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, vgg_name: str):
        super(VGG, self).__init__()
        self.vgg = vgg_units(vgg_name)
        # self.classifier = nn.Linear(512, len(CIFAR10_CLASSES));
        self.avg_pooling = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, len(CIFAR10_CLASSES)),
        )

    def forward(self, x):
        x = self.vgg(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_dataset(batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train = tv.datasets.CIFAR10(root="./data", download=True, transform=tf)
    print(f"Number of training images: {len(train)}")
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)

    test = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
    print(f"Number of testing images: {len(test)}")
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Image shape: {test[0][0].shape}")
    return train_loader, test_loader
