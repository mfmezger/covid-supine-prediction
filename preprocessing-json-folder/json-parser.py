import json
import os

import pandas as pd

path = "/home/mfmezger/data/covid-19-chest-x-ray-dataset/releases/covid-only/annotations/"


def main():
    # crawl the path.
    files = os.listdir(path)
    files = sorted(files)

    lst = []
    for i in files:
        with open(path + i) as f:
            json_file = json.load(f)
            data = json_file["annotations"]
            view_name = ""
            for x in data:
                if ("view/ap" in x["name"].lower()):
                    view_name = "ap"
                if ("view/ap_supine" in x["name"].lower()):
                    view_name = "supine"
                if ("view/pa" in x["name"].lower()):
                    view_name = "pa"
                if ("view/lateral" in x["name"].lower()):
                    view_name = "lateral"
                if ("view/coronal" in x["name"].lower()):
                    view_name = "coronal"
                if ("view/axial" in x["name"].lower()):
                    view_name = "axial"

            if view_name != "":
                # get the other data of the patient.
                image_name = i
                lst.append([image_name, view_name])

    df = pd.DataFrame(lst, columns=["filename", "position"])
    df.to_csv("images.csv")


if __name__ == '__main__':
    main()
