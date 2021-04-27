import torch.nn as nn
from torchvision.models import resnet50


class Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.model = resnet50(pretrained=True)
        self.fc = nn.Linear(1000, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x



