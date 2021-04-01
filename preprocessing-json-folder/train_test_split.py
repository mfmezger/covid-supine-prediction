import os
import random
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
    supine_names = os.listdir(path+"supine/")
    ap_names = os.listdir(path+"ap/")
    pa_names = os.listdir(path+"pa/")
    lateral_names = os.listdir(path + "lateral/")

    # calculate the amount of trainings, val and test images.
    len_val_supine = int(len(supine_names) *0.1)
    len_val_ap = int(len(ap_names) *0.1)
    len_val_pa = int(len(pa_names) *0.1)
    len_val_lateral = int(len(lateral_names) * 0.1)

    # move for every folder.

    # select the images that should be moved
    # first random shuffle the images.

    random.shuffle(supine_names)
    random.shuffle(ap_names)
    random.shuffle(pa_names)
    random.shuffle(lateral_names)

    # slice the arrays for the test and val files.

    # split image list.
    test_list_supine =supine_names[0:len_val_supine]
    val_list_supine =supine_names[len_val_supine+1:len_val_supine*2]
    train_list_supine =supine_names[len_val_supine*2+1:]

    test_list_ap =ap_names[0:len_val_ap]
    val_list_ap =ap_names[len_val_ap+1:len_val_ap*2]
    train_list_ap =ap_names[len_val_ap*2+1:]

    test_list_pa =pa_names[0:len_val_pa]
    val_list_pa =pa_names[len_val_pa+1:len_val_pa*2]
    train_list_pa =pa_names[len_val_pa*2+1:]

    test_list_lateral =lateral_names[0:len_val_lateral]
    val_list_lateral =lateral_names[len_val_lateral+1:len_val_lateral*2]
    train_list_lateral =lateral_names[len_val_lateral*2+1:]

    # loop over the lists and move the images.
    for n in supine_names:
        if n in train_list_supine:
            target = path_split + "train/supine/" + n
        if n in val_list_supine:
            target = path_split + "val/supine/" + n
        if n in test_list_supine:
            target = path_split + "test/supine/" + n

        origin = path + "supine/" + n
        shutil.copyfile(origin, target)


if __name__ == '__main__':
    main()




