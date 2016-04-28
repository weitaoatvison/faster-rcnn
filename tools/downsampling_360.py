#!/usr/bin/env pythonfrom utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse

import os
import sys
import time
import cv2
import string
global idx
idx = 0


def gci(filepath, gap):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            gci(fi_d, gap)
        else:
            if os.path.isfile(fi_d) and (os.path.splitext(fi_d)[1] == '.png'):
                global idx
                idx=idx+1
                if idx%gap == 0:
                    print fi_d
                    os.remove(fi_d)

if __name__ == '__main__':
    filepath = '/home/lbin/workspace/Data/DLData/Vehicle_Pedestrian_Training'
    gap = 2
    gci(filepath, gap)
