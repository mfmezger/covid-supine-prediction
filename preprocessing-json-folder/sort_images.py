import os
import pandas as pd
import shutil
from pathlib import Path

from glob import glob



def main():
    df = pd.read_csv("images.csv", index_col=0)

    # create new folder structure.

    path = "/home/mfmezger/data/covid-19-dataset/"
    dataset_loc_old = "/home/mfmezger/data/covid-19-chest-x-ray-dataset/images/"
    Path(path).mkdir(parents=False, exist_ok=True)
    Path(path+"supine").mkdir(parents=False, exist_ok=True)
    Path(path+"pa").mkdir(parents=False, exist_ok=True)
    Path(path+"ap").mkdir(parents=False, exist_ok=True)
    Path(path+"lateral").mkdir(parents=False, exist_ok=True)
    Path(path+"axial").mkdir(parents=False, exist_ok=True)
    Path(path+"coronal").mkdir(parents=False, exist_ok=True)


    # create subfolders.

    for index, row in df.iterrows():


        # copy the file into the new folderstructure.
        name = row[0].split(".")[0]
        folder_name = row[1]

        name += ".png"
        origin = dataset_loc_old+name
        target = path+folder_name+"/"+ name
        shutil.copyfile(origin, target)







if __name__ == '__main__':
    main()