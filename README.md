# SupReMap-Lawin

This repository is based on https://github.com/yan-hao-tian/lawin, and has been extended to function as a downstream semantic segmentation evaluation baseline of data generated with the SupReMap project, as described in https://github.com/gsaltintas/RemoteSensingData.

## Setup

Please follow the setup instructions provided below by the original Lawin authors.

## Dataset

Download our datasets into `/data/` and the ImageNet-1K-pretrained Lawin-B5 model into `pretrained/` before training:
```
mkdir pretrained
wget https://algvrithm.com/files/supremap/lawin_trained_on_imagenet1k.pth -O pretrained/mit_b5.pth
mkdir data && cd data
wget https://algvrithm.com/files/supremap/supremap_lawin_swisstopo_dataset_real.zip
wget https://algvrithm.com/files/supremap/supremap_lawin_swisstopo_dataset_generated.zip
unzip supremap_lawin_swisstopo_dataset_real.zip
unzip supremap_lawin_swisstopo_dataset_generated.zip
```

## Training 

This repository assumes your machine uses a single GPU.

Train on real Swisstopo data:

  `python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 ./tools/train.py local_configs/segformer/B5/lawin.b5.256x256.supremap_lawin_generated.py`


Train on Swisstopo data generated from OpenStreetMap vector maps using a pix2pixHD model with a style encoder (see https://github.com/algvr/supremap-imaginaire):

  `python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 ./tools/train.py local_configs/segformer/B5/lawin.b5.256x256.supremap_lawin_real.py`


## Inference

`python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 ./tools/test.py local_configs/segformer/B5/lawin.b5.256x256.supremap_lawin_<real|generated>.py <path_to_your_checkpoint> --show-dir=<path_to_output_directory>`

## Pretrained Models

Lawin-B5 trained on real Swisstopo data:

`https://algvrithm.com/files/supremap/lawin_trained_on_swisstopo_iter_20000.pth`


Lawin-B5 trained on generated Swisstopo data:

`https://algvrithm.com/files/supremap/lawin_trained_on_pix2pixhd_with_style_encoder_iter_20000.pth`


## Results

We provide the following results achieved on the above datasets for reference. Visualizations are available at https://algvrithm.com/supremap-vis-v1/.

### Real Swisstopo data

Iteration 20K:

```  
Global results:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 67.34 | 77.23 | 81.56 |
+--------+-------+-------+-------+


Per-class results:

+------------+-------+-------+
| Class      | IoU   | Acc   |
+------------+-------+-------+
| background | 73.46 | 87.34 |
| building   | 75.4  | 85.13 |
| road       | 53.81 | 67.08 |
| green      | 53.9  | 63.87 |
| water      | 79.98 | 83.13 |
| beach      | 69.57 | 74.95 |
+------------+-------+-------+
```


Best mean IoU up to iteration 20K (iteration 9600):

```
Global results:

+--------+-------+-------+------+
| Scope  | mIoU  | mAcc  | aAcc |
+--------+-------+-------+------+
| global | 68.28 | 77.27 | 81.3 |
+--------+-------+-------+------+

Per-class results:

+------------+-------+-------+
| Class      | IoU   | Acc   |
+------------+-------+-------+
| background | 73.14 | 87.23 |
| building   | 75.07 | 84.92 |
| road       | 53.49 | 66.49 |
| green      | 53.01 | 63.22 |
| water      | 79.1  | 82.41 |
| beach      | 75.84 | 79.38 |
+------------+-------+-------+
```

Best mean accuracy up to iteration 20K (iteration 4800):


```
Global results:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 66.08 | 79.96 | 80.95 |
+--------+-------+-------+-------+

Per-class results:

+------------+-------+-------+
| Class      | IoU   | Acc   |
+------------+-------+-------+
| background | 72.74 | 86.71 |
| building   | 74.52 | 83.99 |
| road       | 52.88 | 66.02 |
| green      | 53.03 | 65.48 |
| water      | 78.14 | 80.72 |
| beach      | 65.17 | 96.87 |
+------------+-------+-------+

```

Best average accuracy up to iteration 20K (iteration 18300):


```
Global results:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 67.22 | 76.62 | 81.67 |
+--------+-------+-------+-------+

Per-class results:

+------------+-------+-------+
| Class      | IoU   | Acc   |
+------------+-------+-------+
| background | 73.53 | 87.69 |
| building   | 75.52 | 85.31 |
| road       | 53.64 | 66.28 |
| green      | 53.62 | 63.2  |
| water      | 79.26 | 81.99 |
| beach      | 65.84 | 67.79 |
+------------+-------+-------+
```

### Generated Swisstopo data

Iteration 20K (best mean IoU & best mean accuracy up to 20K):

```  
Global results:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 63.43 | 70.29 | 86.42 |
+--------+-------+-------+-------+

Per-class results:

+------------+-------+-------+
| Class      | IoU   | Acc   |
+------------+-------+-------+
| background | 79.54 | 90.79 |
| building   | 88.33 | 92.71 |
| road       | 64.32 | 76.6  |
| green      | 54.61 | 65.24 |
| water      | 79.04 | 81.55 |
| beach      | 14.74 | 14.84 |
+------------+-------+-------+
```

Best average accuracy up to iteration 20K (iteration 15700):


```  
Global results:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 62.34 | 69.13 | 86.63 |
+--------+-------+-------+-------+

Per-class results:

+------------+-------+-------+
| Class      | IoU   | Acc   |
+------------+-------+-------+
| background | 79.48 | 91.58 |
| building   | 88.34 | 93.34 |
| road       | 64.4  | 76.58 |
| green      | 51.03 | 57.85 |
| water      | 77.58 | 79.96 |
| beach      | 10.24 | 10.29 |
+------------+-------+-------+
```

# Original Lawin README

## Lawin Transformer

[Paper](https://arxiv.org/abs/2201.01615)

Lawin Transformer: Improving Semantic Segmentation Transformer with Multi-Scale Representations via Large Window Attention (*Under Review*).<br>


### Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Important requirements: ```CUDA 11.6``` and  ```pytorch 1.8.1``` 

```
pip install torchvision==0.9.1
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
pip install einops
cd lawin && pip install -e . --user
```

### Evaluation
Download trained [models](https://drive.google.com/drive/folders/187xf3Ase-NGjnMmi2gq0Q222FxYlB-wm?usp=sharing).
```
# Single-gpu testing
python tools/test.py local_configs/segformer/B2/lawin.b2.512x512.ade.160k.py /path/to/checkpoint_file
```

### Training

Download [weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) pretrained on ImageNet-1K, and put them in a folder ```pretrained/```.

Example: train ```lawin-B2``` on ```ADE20K```:

```
# Multi-gpu training
./tools/dist_train.sh local_configs/segformer/B2/lawin.b2.512x512.ade.160k.py <GPU_NUM> --work-dir <WORK_DIRS_TO_SAVE_WEIGHTS&LOGS> --options evaluation.interval=320000
```

### Citation
```
@article{yan2022lawin,
  title={Lawin transformer: Improving semantic segmentation transformer with multi-scale representations via large window attention},
  author={Yan, Haotian and Zhang, Chuang and Wu, Ming},
  journal={arXiv preprint arXiv:2201.01615},
  year={2022}
}
```
