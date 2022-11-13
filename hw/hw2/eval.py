import torch
from torch import nn
from tqdm import tqdm
from util import VGG, load_dataset, MODEL_PATH, DEVICE


def eval_model(model: nn.Module, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in tqdm(loader):
            total += label.size(0)
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            output = model(img)
            _, predict = torch.max(output.data, 1)
            correct += (predict == label).sum().item()
    print(f"Accuracy evaluated on testing dataset: {100 * correct / total}%")


if __name__ == "__main__":
    _, test_loader = load_dataset(80)
    vgg = VGG("vgg11").to(DEVICE)
    vgg.load_state_dict(torch.load(MODEL_PATH))
    vgg.to(DEVICE)
    eval_model(vgg, test_loader)
