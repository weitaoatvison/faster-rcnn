#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

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
import os
import cPickle



CLASSES = ('__background__','Car', 'Van', 'Truck', 'Cyclist','Pedestrian', 'Person_sitting', 'Tram', 'Misc' ,'DontCare')

NETS = {'vgg16': ('VGG16', 'VGG16_faster_rcnn_final.caffemodel')}

DETECTION_DEGREE={"easy": {'pixel': 40.0, 'occ': 0, 'trunc': 0.15},
                  "moderate": {'pixel': 25.0, 'occ': 1, 'trunc': 0.3},
                  "hard": {'pixel': 25.0, 'occ': 2, 'trunc': 0.5}}
CLASS_OVERLAP={"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5, "__background__": 0.5, "Van": 0.5,
               "Truck": 0.5, "Person_sitting": 0.5, "Tram": 0.5, "Misc": 0.5, "DontCare": 0.5}


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
        cv2.putText(im, '%s %.3f' % (class_name, score), (int(bbox[0]), int(bbox[1] - 2)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,0,255), thickness=2)


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
    CONF_THRESH = 0.6
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
    # cv2.imwrite(fileName,im)
    cv2.waitKey(0)

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

def Reformat(net,Input_Path,Output_Path):
    FileSet=os.listdir(Input_Path)
    all_boxes = [[[] for _ in xrange(len(FileSet))]
                 for _ in xrange(len(CLASSES))]

    for file in FileSet:
        file_name=Input_Path+file
        im=cv2.imread(file_name)
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()
        print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.6
        NMS_THRESH = 0.3
        store_file = file[:-3] + 'txt'
        store_file_path = Output_Path + store_file
        f = open(store_file_path, "a+")
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                write_string=cls+' '+'-1 '+'-1 '+'-1 '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])\
                            +' '+'-1 '+'-1 '+'-1 '+'-1 '+'-1 '+'-1 '+'-1 '+str(score)+'\n'
                f.write(write_string)
        f.close()


def _write_voc_results_file(all_boxes,__image_index):
    for cls_ind, cls in enumerate(CLASSES):
        if cls == '__background__':
            continue
        print 'Writing {} VOC results file'.format(cls)
#        filename = self._get_voc_results_file_template().format(cls)
        filename='/home/radmin/dl/data/kitti/result/comp4_'+cls
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(__image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
'''
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects
'''

def parse_rec(filename):
    objects= []
    f=open(filename)
    for line in f.readlines():
        object_struct={}
        line=line.strip().split()
        object_struct['name']=line[0]
        object_struct['bbox']=[float(line[4]),float(line[5]),float(line[6]),float(line[7])]
        object_struct['score']=float(line[-1])
        object_struct['truncated']=float(line[1])
        object_struct['occluded']=int(line[2])
        objects.append(object_struct)
    return objects

def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             mode="easy"):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            # parse_rec need to be rectify
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    detection_minBB = DETECTION_DEGREE[mode]["pixel"]
    detection_occ = DETECTION_DEGREE[mode]["occ"]
    detection_trunc = DETECTION_DEGREE[mode]["trunc"]

    # extract gt objects for this class
    class_recs = {}
    npos = 0

    #change this variable can switch different debug branch, choose_debug=1 means test as voc's rules,
    #choose_debug=0 means test as kitti's rules
    choose_debug=0
    for imagename in imagenames:
        if choose_debug==1:

            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R)
            #        npos = npos + sum(~difficult)
            npos = npos + bbox.shape[0]
            class_recs[imagename] = {'bbox': bbox,
                                     'det': det}

        else:
            R = [obj for obj in recs[imagename] if obj['name'] == classname and obj['occluded']==detection_occ and
                    obj['truncated']<detection_trunc and min(obj['bbox'][2]-obj['bbox'][0],obj['bbox'][3]-obj['bbox'][1])>=detection_minBB]
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R)
        #        npos = npos + sum(~difficult)
            npos=npos+bbox.shape[0]
            class_recs[imagename] = {'bbox': bbox,
                                     'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    if choose_debug==1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    else:
        splitlines=[]
        for x in lines:
            xx=x.strip().split(' ')
            min_val=min(float(xx[4]) - float(xx[2]), float(xx[5]) - float(xx[3]))
            if(min_val>=detection_minBB):
                splitlines.append(xx)

    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        '''
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
        '''

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    ####################################################################
    '''
    # go down dets and mark TPs and FPs
#    nd=23000
    tp1 = np.zeros(nd)
    fp1 = np.zeros(nd)

    for d in range(nd):
        R = class_recs1[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp1[d] = 1.
                R['det'][jmax] = 1
            else:
                fp1[d] = 1.
        else:
            fp1[d] = 1.

    # compute precision recall
    fp1 = np.cumsum(fp1)
    tp1 = np.cumsum(tp1)
    rec1= tp1 / float(npos1)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec1= tp1 / np.maximum(tp1 + fp1, np.finfo(np.float64).eps)
    ap1 = voc_ap(rec1, prec1)
    '''
    return rec, prec, ap

def _do_python_eval(output_dir='output'):
    annopath='/home/radmin/dl/data/kitti/training/label_2/{:s}.txt'
    imagesetfile='/home/radmin/dl/data/kitti/training/image_2/train.txt'

    cachedir = os.path.join('/home/radmin/dl/data/kitti', 'annotations_cache')
    for _mode in ["easy","moderate","hard"]:
        aps = []
        # The PASCAL VOC metric changed in 2010
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(CLASSES):
            if cls == '__background__':
                continue
            #filename = self._get_voc_results_file_template().format(cls)
            filename = '/home/radmin/dl/data/kitti/result/comp4_' + cls
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=CLASS_OVERLAP[cls],mode=_mode)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Mean AP new= {:.4f}'.format(np.mean(aps[:-1])))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~'+_mode)



def evaluate_detections(all_boxes, output_dir,_image_index):
    _write_voc_results_file(all_boxes,_image_index)
    _do_python_eval(output_dir)
    '''
    if self.config['matlab_eval']:
        self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
        for cls in self._classes:
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            os.remove(filename)
    '''


def _load_image_set_index():
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
#    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
#                                  self._image_set + '.txt')
    image_set_file='/home/radmin/dl/data/kitti/training/image_2/train.txt'
    assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    return image_index


def image_path_at(i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return image_path_from_index(image_index[i])


def image_path_from_index(index):
    """
    Construct an image path from the image's "index" identifier.
    """
#    image_path = os.path.join(self._data_path, 'JPEGImages',
#                              index + self._image_ext)
    ##########################################################################
    image_path=os.path.join('/home/radmin/dl/data/kitti/training/image_2',index+'.png')
    assert os.path.exists(image_path), \
        'Path does not exist: {}'.format(image_path)
    return image_path



#def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
def test_net(net, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    image_index=_load_image_set_index()
    num_images = len(image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(len(CLASSES))]

#    output_dir = get_output_dir(imdb, net)
    output_dir='/home/radmin/running_data'
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    ########################################################################
    #num_images=1
    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
#        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
#            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        image_path = os.path.join('/home/radmin/dl/data/kitti/training/image_2', image_index[i] + '.png')
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        im=cv2.imread(image_path)

#        im = cv2.imread(image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, len(CLASSES)):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                #vis_detections(im, imdb.classes[j], cls_dets)
                pass
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, len(CLASSES))])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, len(CLASSES)):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    evaluate_detections(all_boxes, output_dir,image_index)


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = '/home/radmin/faster_rcnn_end2end/test.prototxt'
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              'vgg16_faster_rcnn_iter_70000_kitti.caffemodel')
    Input_Path='/home/radmin/dl/data/kitti/testing/image_2/'
    Output_Path='/home/radmin/dl/data/kitti/testing/label_2/'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/' 'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
#    im_path='/home/radmin/dl/data/kitti/testing/image_2/'
    cv2.namedWindow('res',cv2.WINDOW_NORMAL)
#   record the detect result
#   Reformat(net,Input_Path,Output_Path)
    test_net(net, max_per_image=100, thresh=0.05, vis=False)

