## YOLO-M2LA: Advancing Bus Front-View Object Detection for Intelligent Public Transportation

## Highlights
1.	Construct BFVD, a high-resolution bus front-view dataset with five traffic-related categories and fixed evaluation splits.
2.	Propose YOLO-M2LA, integrating information-preserving downsampling and multi-scale linear attention for front-view object detection.
3.	Introduce a CBS-SPD module to reduce early spatial information loss and enhance small-object representation.
4.	Design an M2LA block that combines multi-scale dilated attention with efficient linear global aggregation.
5.	Demonstrate consistent improvements on BFVD and public benchmarks under unified training settings.




## ABSTRACT:
Object detection from forward-facing bus cameras presents distinctive visual challenges that are largely overlooked by existing benchmarks designed from the private-car perspective. In bus front-view scenes, pedestrians and riders frequently appear as small, densely distributed, and heavily occluded targets, while vehicles dominate medium- and large-scale regions, resulting in complex and heterogeneous traffic patterns near bus stops and intersections. To address this gap, we introduce the Bus Front-View Dataset (BFVD), a high-resolution dataset collected from bus-mounted cameras under diverse environmental conditions. BFVD contains 8,131 images and 56,137 annotated objects across five categories, with particular emphasis on dense pedestrian regions and frequent occlusions. To handle these challenges, we propose YOLO-M2LA, a multi-scale detection framework enhanced with a lightweight Mamba-based linear attention module. The design improves small-object representation through multi-scale feature refinement and enables efficient long-range dependency modeling with low computational overhead. Extensive experiments on BFVD and public benchmarks show consistent improvements over strong baselines, especially for small and occluded targets. The dataset and code are publicly available to support reproducible research.


## installation
## Setup with Anaconda
step 1.Create conda environmrnt and install pytorch
```python
conda create -n pytorch2 python=3.9
conda activate pytorch2
```

step 2.Install torch and matched torchvision from pytorch.org.
The code was tested using conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

step 3. Install YOLO-M2LA
```bash
git clone https://github.com/wenzijingzi/YOLO-M2LA.git
cd YOLO-M2LA
pip install -r requirements.txt
```

## Bus Front-View Dataset (BFVD)

https://drive.google.com/drive/folders/1W7lCf7hhiq1xI5iRfB_51-dr1emCBYIA?usp=sharing
```python
<dataets_dir>
      │
      ├── images
      │      ├── train
      │      └── val
      │      └── test        
      ├── labels
      │      ├── train
      │      └── val
      │      └── test
      │      └── classes.txt
      └── imageset
             ├── train.txt
             └── val.txt
```
Train
```python
cd <YOLO-M2LA_dir>
python train.py
```
Detect
```python
cd <YOLO-M2LA_dir>
python detect.py
```
