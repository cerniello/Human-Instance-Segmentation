# Human Instance Segmentation (given trajectories)

### Fabio Montello, Francesco Russo and Michele Cernigliaro

Project for the course Advanced Machine Learning @LaSapienza

In this repo we provide the code for Human-Instance-Segmentation from surveillance cameras.
Our work is based on trying to segment specific persons along a video sequence, given their trajectory annotations.

The main implementation is based on [OSVOS-PyTorch](https://github.com/kmaninis/OSVOS-PyTorch) which implements the One-Shot-Video-Object-Segmentation algorithm (you may want to refer to the paper [here](https://arxiv.org/abs/1611.05198))

The code is based on 2 steps:

Given a video sequence (frames in .jpg format) and a trajectory dataset (see *first data setup* section):

    1a. Select a specific person (with certain pID)
    1b. Create ground truth annotations (binary masks in .png format) and JPEG images
            - This procedure is done only if you have trajectory annotations
    2a. Perform OSVOS online training with gt and original frame images
    2b. Perform the istance segmentation for the whole lenght of the person's frames.
    

Our algorithm is suitable for the following video sequences datasets (with related annotations):
- 

## Setup environment

Install
``` bash
pip install ..
pip install ..

```

## Download data

- OSVOS PARENT MODEL [here]()
    
- VIDEO SEQUENCES
    - Other UCY
    
- VIDEO SEQUENCES ANNOTATIONS
    - i.e. annotations from [Trajnet](http://trajnet.stanford.edu/data.php?n=1)
    
    
## Setup data

### first data setup
If you already have annotations and frames, you can skip this part and refer to the second setup.
Otherwise, you will need a folder with the video sequence.

    data  
    ├── video_sequence_frames
    │   ├── frame1.jpg  
    │   ├── frame2.jpg
    │   │   ├── ...
    ├── vide_sequence_person_annotations.csv

The annotation structure in the .csv file should have the following:
``` bash
frame pID x  y
10   1    3.4 12.8
```

Where frame indicates the frame of the person with certain pID (person Identifier) and (x,y) indicates his world coordinates within that frame. You can refer to [Trajnet](http://trajnet.stanford.edu) webpage for more details.


### data second setup

After running the first script, your folder should be like this:

    data
    ├── video_sequence_frames
    │   ├── frame1.jpg  
    │   ├── frame2.jpg
    ├── vide_sequence_person_annotations.csv
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
