import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ranger_adabelief import RangerAdaBelief
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from network import Net
import torch.nn as nn


class LitClassifier(pl.LightningModule):

    def __init__(self, hparams, training=False, batch_size=32,
                 train_path=None, val_path=None,
                 test_path=None, num_classes=4, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.hparams = hparams

        if training:
            self.hparams.learning_rate = learning_rate
            self.batch_size = batch_size
            self.train_path = train_path
            self.val_path = val_path
            self.test_path = test_path
        self.num_classes = num_classes

        self.model = Net(n_channels=3, n_classes=self.num_classes)

    def forward(self, x):
        # use forward for inference/predictions
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        m = nn.LogSoftmax(dim=1)
        nll_loss = nn.NLLLoss()
        loss = nll_loss(m(y_hat), y)
        self.manual_backward(loss)
        self.log('train_loss', loss, on_epoch=True)
        self.trainer.train_loop.running_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        m = nn.LogSoftmax(dim=1)
        nll_loss = nn.NLLLoss()
        loss = nll_loss(m(y_hat), y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # todo argmax
        m = nn.LogSoftmax(dim=1)
        nll_loss = nn.NLLLoss()
        loss = nll_loss(m(y_hat), y)

        self.log('test_loss', loss)

    def train_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomApply([
                transforms.RandomRotation(15)
            ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train = ImageFolder(self.train_path, transform=transform)
        return DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val = ImageFolder(self.val_path, transform=transform)
        return DataLoader(val, batch_size=self.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test = ImageFolder(self.val_path, transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        # return RangerAdaBelief(self.parameters(), lr=self.hparams.learning_rate, eps=1e-12, betas=(0.9, 0.999))
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
