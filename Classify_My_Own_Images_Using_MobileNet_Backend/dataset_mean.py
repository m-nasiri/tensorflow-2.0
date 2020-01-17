#------------------------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow-2.0/
# date: 2020-January-16
#------------------------------------------------------------------------------------------#

import numpy as np
import os
import skimage.io as io
from config import Config as cfg


train_dir = os.path.join(cfg.DATA_DIR, "train")
img_mean = [0, 0, 0]
im_count = 0
for c in cfg.CLASS_NAMES:
    class_dir = os.path.join(train_dir, c)
    im_list = os.listdir(class_dir)
    print(len(im_list))
    for im in im_list:
        im_count +=1
        filename = os.path.join(class_dir, im)
        img = io.imread(filename)
        img = img[:,:,0:3]
        img_mean_ = np.mean(img, axis=0)
        img_mean_ = np.mean(img_mean_, axis=0)
        img_mean += img_mean_
            
img_mean /= im_count

print("mean of training split: ", img_mean) # [232.0220225  232.35545796 232.25618846]

