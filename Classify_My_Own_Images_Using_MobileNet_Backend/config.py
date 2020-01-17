#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2020-January-16
#------------------------------------------------------------------------------------------#

import numpy as np

# Base Configuration Class
class Config(object):

    # Class Names , LABLES, UNICODE (if provided and just for print)
    CLASS_NAMES =  ["spadesuit", "clubsuit", "Join", "Omega", "Phi", "Psi", "heartsuit"]
    CLASS_LABELS = [          0,          1,      2,       3,     4,     5,          6,]
    CLASS_UNICODES = [u"\u2663", u"\u2665", u"\u2A1D", u"\u03A9", u"\u03D5", u"\u03C8", u"\u2660"]

    # Number of classes
    N_CLASSES = len(CLASS_NAMES)

    # Dataset Directorty - Place your own dataset here or make dataset using make_symbol_texture_image_dataset.py
    DATA_DIR = "./dataset"

    # Number of samples per class for train and test splits
    N_TRAIN_PER_CLASS = 800
    N_TEST_PER_CLASS = 200

    # Total number of train and test split
    N_TRAIN = N_CLASSES * N_TRAIN_PER_CLASS
    N_TEST = N_CLASSES * N_TEST_PER_CLASS

    # Image Size
    IMG_HIGHT = 224
    IMG_WIDTH = 224

    # Training Parameters
    BATCH_SIZE = 32
    N_EPOCHS = 10

    # whether finetune backend architecture or leave the weights intact.
    FINETUNE = True 

    