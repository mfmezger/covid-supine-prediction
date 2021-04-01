import os
import shutil
from pathlib import Path





def main():


    path = "/home/mfmezger/data/covid-19-dataset/"



    path_split = "/home/mfmezger/data/covid-19-dataset-split/"

    # define the paths and create the subfolders.
    Path(path_split).mkdir(parents=False, exist_ok=True)
    Path(path_split + "train").mkdir(parents=False, exist_ok=True)
    Path(path_split + "val").mkdir(parents=False, exist_ok=True)
    Path(path_split + "test").mkdir(parents=False, exist_ok=True)

    Path(path_split + "train/supine").mkdir(parents=False, exist_ok=True)
    Path(path_split + "train/pa").mkdir(parents=False, exist_ok=True)
    Path(path_split + "train/ap").mkdir(parents=False, exist_ok=True)
    Path(path_split + "train/lateral").mkdir(parents=False, exist_ok=True)

    Path(path_split + "test/supine").mkdir(parents=False, exist_ok=True)
    Path(path_split + "test/pa").mkdir(parents=False, exist_ok=True)
    Path(path_split + "test/ap").mkdir(parents=False, exist_ok=True)
    Path(path_split + "test/lateral").mkdir(parents=False, exist_ok=True)

    Path(path_split + "val/supine").mkdir(parents=False, exist_ok=True)
    Path(path_split + "val/pa").mkdir(parents=False, exist_ok=True)
    Path(path_split + "val/ap").mkdir(parents=False, exist_ok=True)
    Path(path_split + "val/lateral").mkdir(parents=False, exist_ok=True)

    # define the percentages for train,  val and test.
    train = 0.8
    val = 0.1
    test = 0.

    # get the amount of images of every class that should be moved.
    amount_supine = os.listdir(path+"supine/")
    amount_ap = os.listdir(path+"ap/")
    amount_pa = os.listdir(path+"pa/")
    amount_lateral = os.listdir(path+"lateral/")


    # calculate the amount of trainings, val and test images.
    len_val_supine = int(len(amount_supine) *0.1)
    len_val_ap = int(len(amount_ap) *0.1)
    len_val_pa = int(len(amount_pa) *0.1)
    len_val_lateral = int(len(amount_lateral) *0.1)




    # move for every folder.

if __name__ == '__main__':
    main()




