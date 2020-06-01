from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize

from dataloaders.helpers import *
from torch.utils.data import Dataset

class dataloader(Dataset):
    """Pytorch DATALOADER ORIGINALLY TAKEN FROM OSVOS-PyTorch github repository"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='/content/OSVOS-PyTorch/data',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """

        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        
        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'

        if self.seq_name is None:

            # Initialize the original data splits for training the parent network
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/', seq.strip())))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/', seq.strip(), x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else:

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/', str(seq_name))))
            img_list = list(map(lambda x: os.path.join('JPEGImages/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/', str(seq_name))))
            labels = [os.path.join('Annotations/', str(seq_name), name_label[0].split('.')[0] + '.png')]
            labels.extend([None]*(len(names_img)-1))
            
            if self.train:
                # take only JPEGImages and Annotation images in Annotations
                # note: image and relative gt must have the same name!
                # i.e. ../JPEGImages/00023.jpg => ../Annotations/00023.png
                new_img_list = []
                new_label_list = []

                for idx in range(len(names_img)):
                  img_temp = names_img[idx]
                  if img_temp.split('.')[0] in list(map(lambda x: x.split('.')[0], name_label)):
                    new_img_list.append(os.path.join('JPEGImages/', str(seq_name), img_temp))

                    label_temp = img_temp.split('.')[0] + '.png'
                    new_label_list.append(os.path.join('Annotations/', str(seq_name), label_temp))
                
                img_list = new_img_list.copy()
                labels = new_label_list.copy()

                
                #img_list = [img_list[0]]
                #labels = [labels[0]]
                pass 

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        # debug
        # print('reading_IMG : ' + os.path.join(self.db_root_dir, self.img_list[idx]))

        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))

        # DEBUG
        #print('\t', img.shape)

        if self.labels[idx] is not None:
            # DEBUG
            #print('reading_LABEL : ' + os.path.join(self.db_root_dir, self.labels[idx]))
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
            #print('\t', label.shape)

        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
                gt = np.array(label, dtype=np.float32)
                gt = gt/np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])