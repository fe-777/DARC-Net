# DARC-Net for Semi-Supervised Medical Image Segmentation

<!-- Replace the author names below with the actual author list. -->


## Introduction

This repository provides the official PyTorch implementation of:

**DARC-Net for Semi-Supervised Medical Image Segmentation**

<!-- Add the paper link after the preprint or published paper becomes available. -->

DARC-Net is a semi-supervised medical image segmentation framework built upon two student networks and an exponential moving average teacher. It aims to improve pseudo-label reliability, foreground representation, and model robustness under limited annotations.

## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.6.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.




## Dataset

We provide the training code, data split files, and model definitions for the ACDC, LA, and NIH Pancreas datasets.

The original medical datasets are not included in this repository. Please obtain them from their official sources and comply with the corresponding licenses and usage requirements.

The recommended dataset structure is:

```text
data/
├── ACDC/
│   ├── data/
│   └── data_split/
├── LA/
│   ├── data/
│   └── data_split/
└── Pancreas/
    ├── data/
    └── data_split/
```

The actual dataset path can be modified in the corresponding training scripts or configuration files.

Medical images, patient information, and other private clinical data must not be uploaded to this repository.


## Usage

### ACDC

To train DARC-Net on the ACDC dataset:

```bash
python ./code/ACDC_DARC_train.py
```

To evaluate the trained model:

```bash
python ./code/test_ACDC.py
```






## Acknowledgements

Our code is largely based on[BCP](https://github.com/DeepMed-Lab-ECNU/BCP) and [mean teacher](https://github.com/CuriousAI/mean-teacher) frameworks.

We sincerely thank the authors of these projects for making their code publicly available and contributing to the medical image segmentation community.

Please cite the corresponding original papers when using code or experimental settings derived from these repositories.

