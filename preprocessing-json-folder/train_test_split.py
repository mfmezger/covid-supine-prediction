import os
import shutil
from pathlib import Path





def main():


    path = "/home/mfmezger/data/covid-19-dataset/"



    path_split = "/home/mfmezger/data/covid-19-dataset-split/"

    # define the paths and create the subfolders.
    Path(path_split).mkdir(parents=False, exist_ok=True)
    Path(path_split + "train").mkdir(parents=False, exist_ok=True)
    Path(path + "val").mkdir(parents=False, exist_ok=True)
    Path(path + "test").mkdir(parents=False, exist_ok=True)

    # define the percentages for train,  val and test.
    train = 0.8
    val = 0.1
    test = 0.

    # get the amount of images of every class that should be moved.
    amount_supine = os.listdir(path+"supine/")
    amount_ap = os.listdir(path+"supine/")
    amount_pa = os.listdir(path+"supine/")
    amount_lateral = os.listdir(path+"supine/")


    # calculate the amount of trainings, val and test images.
    len_val_supine = int(amount_supine *0.1)
    len_test_supine = int(amount_supine *0.1)





    # move for every folder.





