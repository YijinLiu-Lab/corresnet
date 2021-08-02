# corresnet
Corresnet is a deep-learning-based image jitter correction method for synchrotron nano-resolution tomographic reconstruction.


## Abstract

Nano-resolution full-field transmission X-ray microscopy has been successfully applied to a wide range of research fields thanks to its capability of non-destructively reconstructing the 3D structures with high resolution. Due to constraints in the practical implementation, such as imperfections in the hardware , the nano-tomography data is often associated with random image jitter. Without a proper image registration process prior to reconstruction, the quality of the resulting reconstructions will be compromised. Here we present a deep-learning-based image jitter correction method which registers the projection images with high efficiency and accuracy, facilitating a high-quality tomographic reconstruction. We demonstrate and validate our development using synthetic and experimental datasets. Our method is effective and readily applicable to a broad range of applications. 
## Workflow 

![image](https://github.com/SSRL-LiuGroup/corresnet/blob/main/Fig/ss1.png)

## Network structure 

![image](https://github.com/SSRL-LiuGroup/corresnet/blob/main/Fig/ss2.png)


## Installation
1 Install dependencies
```
pip install torch
pip install dxchange
pip install os
```

## Citation 
Use this bibtex to cite this repository:
```
@article{jiang_lib_segmentation2020,
  title={Deep-learning-based image registration for nano-resolution tomographic reconstruction},
  author={Tianyu Fu, Kai Zhang， Yan Wang, Jizhou Li, Jin Zhangab, Chunxia Yao, Qili He, Shanfeng Wang, Wanxia Huang, Qingxi Yuan, Piero Pianetta, and Yijin Liu},
  journal={Nature Communications},
  year={2021},
  publisher={Journal of Synchrotron Radiation}
}
```

## Training on your own dataset
