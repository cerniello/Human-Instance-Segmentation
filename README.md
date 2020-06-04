# Human Instance Segmentation (given trajectories)

### Fabio Montello, Francesco Russo and Michele Cernigliaro
Project for the course Advanced Machine Learning @LaSapienza

## Abstract

## Introduction

In this repo we provide the code for Human-Instance-Segmentation from surveillance cameras.
Our work is based on trying to segment specific persons along a video sequence, given their trajectory annotations.

## Related work
The main implementation is based on [OSVOS-PyTorch](https://github.com/kmaninis/OSVOS-PyTorch) code which implements the One-Shot-Video-Object-Segmentation algorithm (you may want to refer to the paper [here](https://arxiv.org/abs/1611.05198))

## Proposed method explained
The code is based on 2 steps:

Given a video sequence (frames in .jpg format) and a trajectory dataset (see **first data setup** section):

    1a. Choose a specific person within the video sequence (with a certain pID)
    1b. Create ground truth annotations (binary masks in .png format) and JPEG images
            - This procedure is done only if you have trajectory annotations
    2a. Perform OSVOS online training with gt and original frame images
    2b. Perform the istance segmentation for the whole lenght of the person's frames.

*Note*: the two tasks are separated. You can perform OSVOS with any other dataset following the **second data setup** section.

Our work is suitable for the following video sequences datasets:

    - UCY
    - ETH


## Setup environment

Install
``` bash
pip install tensorboardx==2.0
pip install scipy==1.2.1
pip install pybgs==3.0.0.post2
```

Copy this repository 
``` bash
git clone https://github.com/cerniello/Human-Instance-Segmentation.git
```

## Download data

- OSVOS PARENT MODEL
    - Download the [parent model](https://data.vision.ee.ethz.ch/kmaninis/share/OSVOS/Downloads/models/pth_parent_model.zip) pre-trained on DAVIS-2016 dataset and unzip it under `models/`
    
- VIDEO SEQUENCES
    - We provide a demo with `crowds_zara02` video sequence. The frames inside the folder should be in .jpg format
    
- VIDEO SEQUENCES ANNOTATIONS
    - we provide a demo with `crowds_zara02.txt`
    - annotations datasets can be found at [Trajnet](http://trajnet.stanford.edu/data.php?n=1)
    
    
## Setup data

### Data preparation for the setup
If you already have annotations and frames, you can skip this part and refer directly to the second setup.
Otherwise, you will need a folder with the video sequence:

    data  
    ├── crowds_zara02_frames
    │   ├── frame1.jpg  
    │   ├── frame2.jpg
    │   ├── ...
    ├── crowds_zara02.txt
    ├── homography_matrix
    │   ├── ucy_zara02.txt

The annotation structure in the .txt file should have the following format:
``` bash
frame pID x  y
10   1    3.4 12.8
```

Where frame indicates the frame of the person with certain pID (person Identifier) and (x,y) indicates his world coordinates within that frame. You can refer i.e. to [Trajnet](http://trajnet.stanford.edu) webpage for more details.


The folder homography matrix contains the 3x3 matrices H in order to perform World2Pix and Pix2World coordinates transformations. We took the matrices for ucy and eth sequences from [this repo](https://github.com/trungmanhhuynh/Scene-LSTM). 
Notice that our code works also without homography matrix (approximating the coordinates conversion using olny the annotations file, even if it is highly reccomended to use them whenever provided). 

### Data setup

After running the first script, your folder should be like this:

    data
    ├── video_sequence_frames
    │   ├── frame1.jpg  
    │   ├── frame2.jpg
    ├── vide_sequence_person_annotations.txt
    ├── JPEGImages
    │   ├── pID1  
    │   │   ├── 00000.jpg 
    │   │   ├── 00001.jpg
    │   │   ├── 00002.jpg
    │   │   ├── ...
    │   ├── pID7  
    │   │   ├── 00000.jpg
    │   │   ├── 00001.jpg 
    │   │   ├── 00002.jpg
    │   │   ├── ...
    ├── Annotations 
    │   ├── pID1  
    │   │   ├── 00004.png
    │   │   ├── 00010.png
    │   │   ├── ...
    │   ├── pID7  
    │   │   ├── 00001.png
    │   │   ├── 00008.png
    │   │   ├── ...

## Run the demo!

#### Create JPEGImages and Binary masks
First, in order to create the annotations for the person with i.e. pID=7, run run_masking_preprocessing.py :

```
python run_masking_preprocessing.py --pID=7 --frames_folder=../path/crowds02frames/ --text_folder=../path/crowds_annotations.txt --homography=../path/ucy_zara02.txt
```

For all the list of commands you can run it with --help
The script will perform the task 1a and 1b and will create person pID7 sequences in `JPEGImages/` and related gt binary masks in `Annotations/`

#### Run OSVOS algorithm

```
python run_OSVOS_pretrained.py --seq_name=pID7 --epochs=200 --output_folder="results" --batch_size=2
```

The algorithm will run with Cuda, if possible. An highly number of epochs is suggested. 

## References
[S. Caelles*, K.K. Maninis*, J. Pont-Tuset, L. Leal-Taixé, D. Cremers, and L. Van Gool - 
One-Shot Video Object Segmentation, Computer Vision and Pattern Recognition (CVPR), 2017.](http://people.ee.ethz.ch/~cvlsegmentation/osvos/)

[K.K. Maninis*, S. Caelles*, Y.Chen, J. Pont-Tuset, L. Leal-Taixé, D. Cremers, and L. Van Gool - 
Video Object Segmentation Without Temporal Information, Transactions of Pattern Analysis and Machine Intelligence (T-PAMI), 2018.](http://people.ee.ethz.ch/~cvlsegmentation/osvos-s/)


[Kaiming He and Georgia Gkioxari and Piotr Doll \'ar and Ross B. Girshick - Mask R-CNN, 2017](https://arxiv.org/pdf/1703.06870.pdf)

[Sobral, Andrews and Bouwmans, Thierry - BGS Library: A Library Framework for Algorithm’s Evaluation in Foreground/Background Segmentation, 2014](https://www.researchgate.net/publication/259574448_BGS_Library_A_Library_Framework_for_Algorithm's_Evaluation_in_ForegroundBackground_Segmentation)

[Sadeghian, Amir and Kosaraju, Vineet and Gupta, Agrim and Savarese, Silvio and Alahi, Alexandre - TrajNet: Towards a Benchmark for Human Trajectory Prediction, 2018](http://trajnet.stanford.edu/)

### Useful links
[OSVOS PyTorch Implementation](https://github.com/kmaninis/OSVOS-PyTorch)   
[Mask RCNN Pytorch Implementation](https://github.com/spmallick/learnopencv/tree/master/PyTorch-Mask-RCNN)   
[BGSLibrary](https://github.com/andrewssobral/bgslibrary)
[Parent model pretrained on DAVIS](https://data.vision.ee.ethz.ch/kmaninis/share/OSVOS/Downloads/models/pth_parent_model.zip)  
[Video dataset - Pedestrians in front of the Zara store in Makariou Street, Nicosia, Cyprus](https://repo.vi-seem.eu/bitstream/handle/21.15102/VISEEM-316/Zara.zip?sequence=1&isAllowed=y)
