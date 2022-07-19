# Learning Invariant Visual Representations for CZSL

This repository provides dataset splits and code for Paper:

Learning Invariant Visual Representations for Compositional Zero-Shot Learning, ECCV 2022 [Paper (arXiv)](https://arxiv.org/pdf/2206.00415.pdf)



## Usage 

1. Clone the repo 

2. We recommend using Anaconda for environment setup. To create the environment and activate it, please run:
```
    conda env create --file environment.yml
    conda activate czsl
```

3. The dataset and splits can be downloaded from: [CZSL-dataset](https://drive.google.com/drive/folders/1ZSw4uL8bjxKxBhrEFVeG3rgewDyDVIWj).


4. To run IFL for UT-Zappos dataset:
```
    Training:
    python train.py --config config/zappos.yml

    Testing:
    python test.py --logpath LOG_DIR
```


**Note:** Most of the code is an improvement based on https://github.com/ExplainableML/czsl.

## References
If you find this paper useful in your research, please consider citing:
```
@article{zhang2022learning,
  title={Learning Invariant Visual Representations for Compositional Zero-Shot Learning},
  author={Zhang, Tian and Liang, Kongming and Du, Ruoyi and Sun, Xian and Ma, Zhanyu and Guo, Jun},
  journal={arXiv preprint arXiv:2206.00415},
  year={2022}
}
```
