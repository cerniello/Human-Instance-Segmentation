# Human-Instance-Segmentation
Project for the course Advanced Machine Learning @LaSapienza

## Setup environment

``` bash
pip install ..
pip install ..

```

## Download data

- OSVOS PARENT MODEL
    - 
    - [COCO 2017 Val images [5K/1GB]](http://images.cocodataset.org/zips/val2017.zip)
    - [COCOPersons Train Annotation (person_keypoints_train2017_pose2seg.json) [166MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_train2017_pose2seg.json)
    - [COCOPersons Val Annotation (person_keypoints_val2017_pose2seg.json) [7MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_val2017_pose2seg.json)
    
- VIDEO SEQUENCES
    - Other UCY
    - [images [667MB] & annotations](https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman)
    
    
## Setup data

### First setup
If you already have annotations and frames, you can skip this part and refer to the second setup.
Otherwise, you will need a folder with the video sequence

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

Where frame indicates the frame of the person with certain pID (person Identifier) and x y indicates the world coordinates.
You can refer to [Trajnet](http://trajnet.stanford.edu) webpage


### Second setup

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
