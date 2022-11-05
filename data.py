import mat73
import pickle
from os import walk
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    TRAIN = "/home/matt/Kidney/VisibleAligned/"
    LABEL = "/home/matt/Kidney/SatoskarKidneyROIs/"

    SKIP_LABEL = ["AM6_roisResized.mat"]
    SKIP_TRAIN = ["DM8.mat", "DM9.mat", "DM10.mat"]
    train_data = list(next(walk(TRAIN), (None, None, []))[2])
    train_label = list(next(walk(LABEL), (None, None, []))[2])#, key=lambda x: x[0:-16], reverse=False)

    # print (len(train_data))
    # print (len(train_label))
    # for skip in SKIP_LABEL:
        # train_label.remove(skip)
    # for skip1 in SKIP_TRAIN:
        # train_data.remove(skip1)

    train_data.sort(key=lambda x: x[0:-4])
    train_label.sort(key=lambda x: x[0:-16])
    
    for i, _ in tqdm(enumerate(train_label)):
        # with open("/home/rshb/myspinner/kidney/data/train/" + train_data[i][0:-4] + ".pkl", "wb") as f:
        #     data = None
        #     try:
        #         data = mat73.loadmat(TRAIN + train_data[i])
        #     except:
        #         data = sio.loadmat(TRAIN + train_data[i])
        #     pickle.dump(data, f)
        if (train_label[i][0:-16] in ['DM10', 'DM9', 'DM8']):
            with open("/home/rshb/myspinner/kidney/data/label/" + train_label[i][0:-4] + ".pkl", "wb") as f:
                data = None
                try:
                    data = mat73.loadmat(LABEL + train_label[i])
                except:
                    data = sio.loadmat(LABEL + train_label[i])
                pickle.dump(data, f)
