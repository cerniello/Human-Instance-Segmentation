from __future__ import division

import os
import argparse
import socket
import timeit
from datetime import datetime

# import pytorch/opencv/matplotlib and other random utils
import torch
import torchvision
import torch
import cv2
import random
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pybgs as bgs
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import imageio
import pybgs
import re


PID = 3
MASK_RCNN = False
FRAMES_DIR = './data/crowds_zara02_frames/'
OUTPUT_DIR = './data/'
TEXT_DIR = './data/crowds_zara02.txt'
START_FRAME = 0
DEBUG = False
DISTANCE = 40


class mapper():
    def __init__(self, df, H=576, W=720):
        minx, maxx = np.min(df.x), np.max(df.x)
        miny, maxy = np.min(df.y), np.max(df.y)

        self.H = H
        self.W = W
        self.Rx = (W - 0)/(maxx - minx)
        self.Ry = (H - 0)/(maxy - miny)

    def World2Pix(self, xy_coord):
        assert len(xy_coord) == 2, "xy should be a point tuple/list"
        x = xy_coord[0]
        y = xy_coord[1]

        xhat = x * self.Rx
        yhat = self.H - y * self.Ry

        return [int(xhat), int(yhat)]

    def Pix2World(self, xy_coord):
        assert len(xy_coord) == 2, "xy should be a point tuple/list"
        x = xy_coord[0]
        y = xy_coord[1]

        xhat = x / self.Rx
        yhat = (self.H - y) / self.Ry

        return [xhat, yhat]


def parse_args():
    parser = argparse.ArgumentParser(description='Masking creating for OSVOS')
    parser.add_argument('--pID',
                        default=3,
                        help="Person id to be selected from the dataframe")
    parser.add_argument(
        '--use_mask_rcnn',
        default=False,
        help='Define the type of masking, available: mask RCNN or BGS')
    parser.add_argument('--output_folder',
                        default='./data/',
                        help='Where the results will be located')
    parser.add_argument('--frames_folder',
                        default='./data/crowds_zara02_frames/',
                        help='Folder where the frames are contained')
    parser.add_argument('--text_folder',
                        default='./data/crowds_zara02.txt',
                        help='Folder where the frame annotations are contained')
    parser.add_argument('--start_frame',
                        default=0,
                        help='Frame from which starting the annotation')
    parser.add_argument('--debug',
                        default=False,
                        help='Save bounding boxes')
    parser.add_argument('--distance',
                        default=40,
                        help='Save bounding boxes')

    args = parser.parse_args()
    return args


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_path, threshold):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i]
                  for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def instance_segmentation_api(img_path, threshold=0.5, rect_th=1, text_size=2, text_th=1):
    """
    instance_segmentation_api
      parameters:
        - img_path - path to input image
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(masks)):
        if(pred_cls[i] == 'person'):
            rgb_mask = random_colour_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.2, 0)

            cv2.rectangle(img, boxes[i][0], boxes[i][1],
                          color=(0, 255, 0), thickness=rect_th)
            #cv2.circle(img, (379, 346), radius=2, color=(255, 0, 0), thickness=-1)

    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def find_bounding_box_mask(img_path, x_gt, y_gt, threshold=0.5):

    masks, boxes, pred_cls = get_prediction(
        img_path, threshold)  # pred mask, box and class
    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dist_to_comp = 999999999
    final_mask = []
    final_bb = []

    for i in range(len(boxes)):
        if(pred_cls[i] == 'person'):

            bb_x_sup = (boxes[i][0][0] + boxes[i][1][0])/2
            bb_y_sup = boxes[i][0][1]

            # dist measure
            dist = np.sqrt((x_gt - bb_x_sup)**2 + (y_gt - bb_y_sup)**2)

            if(dist < dist_to_comp):
                dist_to_comp = dist
                final_mask = masks[i]
                final_bb = boxes[i]

    return dist_to_comp, final_mask, final_bb


def frames_pID(pID, start_frame, output_path, frames_path, distance):

    data_tmp = data[data['pID'] == pID].copy()
    data_tmp['pID'] = data_tmp['pID'].apply(lambda x: int(x))
    data_tmp['frame'] = data_tmp['frame'].apply(lambda x: int(x))
    data_tmp = data_tmp[(data_tmp['frame'] >= start_frame)
                        & (data_tmp['frame'] < 350)]

    os.makedirs(output_path + '/JPEGImages/pID' + str(pID), exist_ok=True)
    os.makedirs(output_path + '/Annotations/pID' + str(pID), exist_ok=True)

    dict_masks_bb = {}

    # data_tmp['frame'].iloc[0], data_tmp['frame'].iloc[-1]
    for idx, f in enumerate(range(data_tmp['frame'].iloc[0], data_tmp['frame'].iloc[-1])):
        # Copy and rename of the images from frames folder
        shutil.copy2(frames_path + 'frame' + str(f) + ".jpg", output_path +
                     '/JPEGImages/pID' + str(pID) + '/' + str(idx).zfill(5) + '.jpg')

    # for idx, f in enumerate(data_tmp['frame']):
        if(f in list(data_tmp['frame'])):
            # Conversion from world coordinates of the ground truth to actual pixel coordinates
            curr_x, curr_y = m.World2Pix([data_tmp[data_tmp['frame'] == f]['x'], data_tmp[data_tmp['frame'] == f]['y']])
                
            # Extraction of the closest bounding box to the GT, with relative mask
            dist, mask, bb = find_bounding_box_mask(output_path + '/JPEGImages/pID' + str(pID) + '/' + str(idx).zfill(5) + '.jpg',
                                                    curr_x, curr_y, threshold=0.20)
            if(dist < distance):
                dict_masks_bb[f] = {'id_annotation': str(idx).zfill(
                    5), 'dist': dist, 'mask': mask, 'bb': bb, 'x_GT': curr_x, 'y_GT': curr_y}

    # return of the dataset with the frames having onli the pID selected
    return data_tmp, dict_masks_bb


if __name__ == '__main__':
    # get the pretrained model from torchvision.models
    # Note: pretrained=True will get the pretrained weights for the model.
    # model.eval() to use the model for inference
    working_path = os.getcwd()  # /content/path/

    args = parse_args()

    PID = int(args.pID)
    MASK_RCNN = args.use_mask_rcnn
    OUTPUT_DIR = args.output_folder
    FRAMES_DIR = args.frames_folder
    TEXT_DIR = args.text_folder
    START_FRAME = args.start_frame
    DEBUG = args.debug
    DISTANCE = args.distance

    print(' - Download model')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    data_path = OUTPUT_DIR

    print(' - Create subfolders')
    os.makedirs(data_path + '/JPEGImages', exist_ok=True)
    os.makedirs(data_path + '/Annotations', exist_ok=True)

    data = pd.read_csv(TEXT_DIR, sep='\t', header=None)
    data.columns = ['frame', 'pID', 'x', 'y']

    print(' - Selecting the pictures with id: ' + str(PID))
    m = mapper(data)
    data_pID, dict_masks_bb = frames_pID(pID=PID, start_frame=START_FRAME, output_path=OUTPUT_DIR,  frames_path=FRAMES_DIR, distance=DISTANCE)

    if(MASK_RCNN):
        print(' - I will mask with mask RCNN')
    else:
        print(' - I will mask with BGS')

    if(MASK_RCNN):
        for key in dict_masks_bb.keys():
            plt.imsave(OUTPUT_DIR+'Annotations/pID'+str(PID)+'/'+dict_masks_bb[key]['id_annotation']+'.png',
                       dict_masks_bb[key]['mask'].astype(np.uint8), cmap=cm.binary.reversed())
    else:
        pathIn = FRAMES_DIR

        algorithm = bgs.LBAdaptiveSOM()
        # PixelBasedAdaptiveSegmenter
        # MultiLayerBGS
        # LBAdaptiveSOM
        # DPWrenGABGS
        # MixtureOfGaussianV2BGS
        # .LBSimpleGaussian
        # LOBSTER

        for frame_path in [f for f in os.listdir(pathIn) if re.match(r'[0-9]+.*\.jpg', f)]:
            try:
                frame = cv2.imread(os.path.join(pathIn, frame_path))
                img_output = algorithm.apply(frame)
            except:
                print(frame_path)
                print(frame.shape)
                pass

        for key in dict_masks_bb.keys():
            frame = cv2.imread(os.path.join(pathIn, 'frame'+str(key)+'.jpg'))
            img_output = algorithm.apply(frame)
            x_sup = int(dict_masks_bb[key]['bb'][0][0])
            y_sup = int(dict_masks_bb[key]['bb'][0][1])
            x_inf = int(dict_masks_bb[key]['bb'][1][0])
            y_inf = int(dict_masks_bb[key]['bb'][1][1])

            final_img = np.zeros(img_output.shape)
            final_img[y_sup:y_inf, x_sup:x_inf,
                      :] = img_output[y_sup:y_inf, x_sup:x_inf, :]
            plt.imsave(OUTPUT_DIR+'Annotations/pID'+str(PID)+'/'+dict_masks_bb[key]['id_annotation']+'.png',
                       final_img.astype(np.uint8), cmap='binary')

    if(DEBUG):
        os.makedirs(OUTPUT_DIR+'BoundingBox/', exist_ok=True)
        for key in dict_masks_bb.keys():
            img = cv2.imread(FRAMES_DIR + 'frame' + str(key) + ".jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.rectangle(img, dict_masks_bb[key]['bb'][0], dict_masks_bb[key]['bb'][1], color=(
                0, 255, 0), thickness=2)

            os.makedirs(OUTPUT_DIR+'BoundingBox/pID' +
                        str(PID)+'/', exist_ok=True)
            plt.imsave(OUTPUT_DIR+'BoundingBox/pID'+str(PID)+'/'+dict_masks_bb[key]['id_annotation']+'.jpg',
                       img)
