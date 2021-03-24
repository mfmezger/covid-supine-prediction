import os

from PIL import Image


def main():
    path = "/home/mfmezger/data/covid-19-chest-x-ray-dataset/images/"

    files = os.listdir(path)

    for f in files:

        if f.split(".")[1] == "jpg" or f.split(".")[1] == "jpeg" or f.split(".")[1] == "JPG" or f.split(".")[1] \
                == "PNG":
            img = Image.open(path + f)
            img.save(path + f.split(".")[0] + ".png")
            os.remove(path + f)


if __name__ == '__main__':
    main()
