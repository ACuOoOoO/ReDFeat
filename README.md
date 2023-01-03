# ReDFeat


This repository code for multimodal feature ReDFeat. [&#34;RedFeat: Recoupling detection and description for multimodal feature learning&#34;](https://arxiv.org/abs/2205.07439)

## Requirements

The code is build on Pytorch 1.10 and Korina 0.6.2. Later version should also be compatible.

## Datasets

Please clone the [mutilmodal_feature_evaluation benchmark](https://github.com/ACuOoOoO/Multimodal_Feature_Evaluation) by

```bash
git clone https://github.com/ACuOoOoO/Multimodal_Feature_Evaluation
```

And then, follow [README.md](https://github.com/ACuOoOoO/Multimodal_Feature_Evaluation/blob/main/README.md) to build up train and test data.

## Training

run 

```bash
python train.py --image_type=VIS_IR --name=IR
```

The major parameters include:
**image_type**: type of modal (VIS_NIR,VIS_IR,VIS_SAR)
**name**: name of checkpoint
**datapath**: path for training data

## Evaluation

Evaluation codes for feature extraction, matching and transform estimation are included in [mutilmodal_feature_evaluation benchmark](https://github.com/ACuOoOoO/Multimodal_Feature_Evaluation):

**extract_ReDFeat.py**: extract ReDFeat for three types of modals.
**match.py**: reproduce feature matching experiments in the paper.
**reproj.py**: reproduce image registration experiments in the paper.
