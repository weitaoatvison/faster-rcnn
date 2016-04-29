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

NETS = {'vgg16': ('VGG16', 'VGG16_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        for li in range(4):
            bbox[li] = int(bbox[li])
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), thickness=2)
        # cv2.putText(im, '%s %.3f' % (class_name, score), (int(bbox[0]), int(bbox[1] - 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,0,255), thickness=2)


def demo(net, image_name,idx):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    print image_name
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    cv2.imshow('res', im)
    fileName = '/home/lbin/Desktop/imgs/%05d.jpg'%i
    cv2.waitKey(1)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join('/home/lbin/workspace/DL/py-faster-rcnn/models/kitti/VGG16/test.prototxt')
    caffemodel = os.path.join('/home/lbin/workspace/DL/py-faster-rcnn/data/faster_rcnn_models/vgg16_faster_rcnn_iter_100000.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    im_path = '/home/lbin/workspace/Data/3rdparty/kitti/visual odometry/dataset_color/sequences/03/image_2/'
    cv2.namedWindow('res',cv2.WINDOW_NORMAL)
    for i in range(10000):
        im_name = im_path + '%06d.png' %i
        demo(net, im_name,i)
