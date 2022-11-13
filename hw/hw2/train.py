from typing import Tuple, List

import matplotlib.pyplot as plt
import torch

from torch import nn, optim
from torch.nn import functional
from tqdm import tqdm
from util import VGG, load_dataset, MODEL_PATH, DEVICE

CIFAR10_CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def train_model(model: nn.Module, loader, epoch: int = 2) -> List[float]:
    loss = functional.cross_entropy
    # opt = optim.Adam(model.parameters())
    opt = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    losses = []
    for e in range(epoch):
        loss_val = 0
        with tqdm(loader, desc=f"Epoch {e}: ") as t:
            for i, data in enumerate(t):
                ipt, label = data
                ipt = ipt.to(DEVICE)
                label = label.to(DEVICE)
                opt.zero_grad()
                output = model(ipt)
                crt_loss = loss(output, label)
                crt_loss.backward()
                opt.step()
                loss_val += crt_loss.item()
                if i % 50 == 49:
                    losses.append(loss_val)
                    loss_val = 0
    return losses


def plot_losses(losses: List[float], fig_path: str = "hw2-7-c.png"):
    plt.plot(losses)
    plt.xlabel("batch number")
    plt.ylabel("loss")
    plt.savefig(fig_path)


def main():
    print("Number of classes: 10")
    train_loader, _ = load_dataset(80)
    vgg = VGG("vgg11").to(DEVICE)
    losses = train_model(vgg, train_loader, epoch=150)
    plot_losses(losses)
    torch.save(vgg.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
