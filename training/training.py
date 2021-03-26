from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision.datasets import  ImageFolder
from network import LitClassifier


def main(hparams):
    seed_everything(11)


    # todo add WANDB.

    train_dataset = ImageFolder()
    val_dataset = ImageFolder(args.val_path)
    test_dataset = ImageFolder(args.test_path)

    # initialize the model
    model = LitClassifier(training=True, hparams=hparams, batch_size=hparams.batch_size,
                         train_path=args.train_path, val_path=args.val_path,
                         test_path=args.test_path,  num_classes=2,learning_rate=hparams.learning_rate)

    # initialize loggers
    checkpoint_callback = ModelCheckpoint(filepath='pl_model/')

    # Intialize the Trainer.
    trainer = Trainer(gpus=1, num_nodes=1,
                       profiler=True, min_epochs=1, max_epochs=hparams.max_epochs,
                      checkpoint_callback=checkpoint_callback, benchmark=True, progress_bar_refresh_rate=20)

    # start  the Training.
    trainer.fit(model)

    # activate testing
    trainer.test(model)

    trainer.save_checkpoint("model" + str(args.max_epochs) + "_" + str(args.learning_rate) + ".ckpt")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--opt', default="ranger", type=str)
    parser.add_argument('--loss', default="ce")
    parser.add_argument('--augmentations', default=False, type=bool)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--train_path',
                        default="/home/mfmezger/data/COVID/")
    parser.add_argument('--val_path',
                        default="/home/mfmezger/data/COVID/")
    parser.add_argument('--test_path',
                        default="/home/mfmezger/data/COVID/")
    args = parser.parse_args()

    main(args)
