import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ranger_adabelief import RangerAdaBelief
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import  ImageFolder

from torchvision.models import resnet18


class LitClassifier(pl.LightningModule):

    def __init__(self, hparams, training=False,  batch_size=32,
                 train_path=None, val_path=None,
                 test_path=None,  num_classes=2, learning_rate=1e-3):

        super().__init__()
        self.save_hyperparameters()

        if training:
            self.hparams.learning_rate = learning_rate
            self.batch_size = batch_size
            self.train_path = train_path
            self.val_path = val_path
            self.test_path = test_path
        self.num_classes = num_classes

        self.model = resnet18(pretrained=True)
        self.fc = torch.nn.Linear(1000, self.num_classes)


    def forward(self, x):
        # use forward for inference/predictions
        x = self.model(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def train_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomApply([
                transforms.RandomRotation(15)
            ]),
            transforms.ToTensor(),
            transforms.Normalize((0.48232,), (0.23051,))
        ])
        train = ImageFolder(self.train_path, transform=transform)
        return DataLoader(train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.48232,), (0.23051,))
        ])
        val = ImageFolder(self.val_path, transform=transform)
        return DataLoader(val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.48232,), (0.23051,))
        ])
        test = ImageFolder(self.val_path, transform=transform)
        return DataLoader(test, batch_size=self.batch_size)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return RangerAdaBelief(self.parameters(), lr=self.hparams.learning_rate, eps=1e-12, betas=(0.9, 0.999))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitClassifier")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser
