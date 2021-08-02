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
  title={Machine-Learning-Revealed Statistics of the Particle-Carbon/Binder Detachment in Li-Ion Battery Cathodes},
  author={Z. Jiang, J. Li, Y.Yang, L. Mu, C. Wei, X. Yu, P. Pianetta, K. Zhao, P. Cloetens, F. Lin and Y. Liu},
  journal={Nature Communications},
  year={2020},
  volume={11},
  number={2310},
  doi={10.1038/s41467-020-16233-5},
  publisher={Nature Publishing Group}
}
```
