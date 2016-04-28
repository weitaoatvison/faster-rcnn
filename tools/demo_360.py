#!/usr/bin/env python

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__','Car', 'Van', 'Truck', 'Cyclist','Pedestrian', 'Person_sitting', 'Tram', 'Misc' ,'DontCare')

def vis_detections(im, class_name, dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        for li in range(4):
            bbox[li] = int(bbox[li])
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), thickness=2)
        cv2.putText(im, '%s %.3f' % (class_name, score), (int(bbox[0]), int(bbox[1] - 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,0,255), thickness=2)

def demo(net, image_name,idx,savedName):
    image_name = cv2.resize(image_name,(1280,720))

    scores, boxes = im_detect(net, image_name)

    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    iCount = 0;
    ori_img = image_name.copy()
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if(cls == 'Cyclist' or cls == 'Pedestrian' or cls =='Person_sitting'):
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            iCount = iCount + len(inds)
        vis_detections(image_name, cls, dets, thresh=CONF_THRESH)

    print iCount
    cv2.imshow('res', image_name)
    if(iCount>=1):
        cv2.imwrite(savedName,ori_img)
    cv2.waitKey(1)

import os
import sys
import time
import cv2
import string
global g_idx
g_idx = 0


def gci(filepath, savedPath, gap):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            gci(fi_d, savedPath, gap)
        else:
            if os.path.isfile(fi_d) and (os.path.splitext(fi_d)[1] == '.MOV' or os.path.splitext(fi_d)[1] == '.MP4'):
                prefix = os.path.basename(fi_d)
                videoCapture = cv2.VideoCapture(fi_d)
                idx = 0
                success, frame = videoCapture.read()
                while success:
                    if idx%string.atoi(gap) == 0:
                        savedPath_sub = savedPath + '/%05d' %(g_idx/1000)
                        if os.path.exists(savedPath_sub) == False:
                            os.mkdir(savedPath_sub)
                        savedFileName=savedPath_sub+ '/' + prefix + '_%d.png' %(idx/string.atoi(gap))
                        print savedFileName
                        if idx/string.atoi(gap)!=0:
                            demo(net,frame,idx,savedFileName)
                        global g_idx
                        g_idx=g_idx+1
                    success, frame = videoCapture.read()
                    idx=idx+1


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = os.path.join('/home/lbin/workspace/DL/py-faster-rcnn/models/kitti/VGG16/faster_rcnn_end2end/test.prototxt')
    caffemodel = os.path.join('/home/lbin/workspace/DL/py-faster-rcnn/data/faster_rcnn_models/vgg16_faster_rcnn_iter_70000_kitti.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/' 'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    filepath = sys.argv[1]
    savedPath = sys.argv[2]
    gap = sys.argv[3]

    cv2.namedWindow('res',cv2.WINDOW_NORMAL)

    gci(filepath, savedPath, gap)

    # print '\n\nLoaded network {:s}'.format(caffemodel)
    # im_path = '/home/lbin/workspace/Data/3rdparty/kitti/detection/data_object_image_2/training/image_2/'

    # for i in range(7518):
    #     im_name = im_path + '%06d.png' %i
    #     demo(net, im_name,i)
