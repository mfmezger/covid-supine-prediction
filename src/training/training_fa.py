from fastai.vision.all import *



def main():
    path_dataset = "/home/mfmezger/data/covid-19-dataset-split/train/"
    dls = ImageDataLoaders.from_folder(path_dataset, train='train', valid='val', bs=16)
    dls.show_batch()
    learn = cnn_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(), ps=0.25)


if __name__ == '__main__':
    main()