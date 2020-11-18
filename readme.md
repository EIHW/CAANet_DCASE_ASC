# CAA-Net: Conditional Atrous CNNs with Attention
A Python code for the subtask B of the task 1 in DCASE 2018/2019.


# Data
DCASE 2018 Task 1 - Acoustic Scene Classification, containing two tasks:

subtask A: data from device A

subtask B: data from device A, B, and C

This code is working on the dataset of subtask B.


# Preparation
channels:
  - pytorch
  - defaults
dependencies:
  - matplotlib=2.2.2
  - numpy=1.14.5
  - h5py=2.8.0
  - pytorch=0.4.0
  - pip:
    - audioread==2.1.6
    - librosa==0.6.1
    - scikit-learn==0.19.1
    - soundfile==0.10.2


# File structure
- CAANet_DCASE_ASC
  - pytorch
  - utils-pred
  - runme.sh

Note: 
- The folders "pytorch-pred" and "utils-pred" are corresponding to multi-task conditional training.

- The folders "pytorch-wopred" and "utils-wopred" are corresponding to teacher forcing conditional training.

- Please change the folder names as "pytorch" and "utils-pred" before running the code.


# Run 
sh runme.sh

In runme.sh, please run the following files:
1. feature extracttion: utils/features.py
2. training a model, and evaluation: main_pytorch.py

# Cite
If the user referred the code, please cite our paper:

Z. Ren, Q. Kong, J. Han, M. D. Plumbley and B. W. Schuller, "CAA-Net: Conditional Atrous CNNs with Attention for Explainable Device-robust Acoustic Scene Classification," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2020.3037534.





Zhao Ren

Chair of Embedded Intelligence for Health Care and Wellbeing

University of Augsburg

18.11.2020

