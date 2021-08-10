# corresnet
Corresnet is a deep-learning-based image jitter correction method for synchrotron nano-resolution tomographic reconstruction.


## Abstract

Nano-resolution full-field transmission X-ray microscopy has been successfully applied to a wide range of research fields thanks to its capability of non-destructively reconstructing the 3D structures with high resolution. Due to constraints in the practical implementation, such as imperfections in the hardware , the nano-tomography data is often associated with random image jitter. Without a proper image registration process prior to reconstruction, the quality of the resulting reconstructions will be compromised. Here we present a deep-learning-based image jitter correction method which registers the projection images with high efficiency and accuracy, facilitating a high-quality tomographic reconstruction. We demonstrate and validate our development using synthetic and experimental datasets. Our method is effective and readily applicable to a broad range of applications. 

<img src="https://github.com/SSRL-LiuGroup/corresnet/blob/main/fig/fig3.png" width="600px">


## Workflow 
The workflow is shown in the figure below:

<img src="https://github.com/SSRL-LiuGroup/corresnet/blob/main/fig/fig1.png" width="600px">


## Network structure 
The network structure is shown in the figure below:

<img src="https://github.com/SSRL-LiuGroup/corresnet/blob/main/fig/fig2.png" width="600px">

## Getting Started

![train.py](https://github.com/SSRL-LiuGroup/corresnet/blob/main/train.py) shows how to train network on your own dataset. 

![model.py](https://github.com/SSRL-LiuGroup/corresnet/blob/main/model.py) These files contain the main network implementation.

![res_data.py](https://github.com/SSRL-LiuGroup/corresnet/blob/main/res_data.py) This notebook visualizes the different pre-processing steps to prepare the training data.

![evaluation.py](https://github.com/SSRL-LiuGroup/corresnet/blob/main/evaluation.py) shows how to evaluation network on your own dataset. 

## Installation
1.Clone this repository via git clone https://github.com/SSRL-LiuGroup/corresnet.git
2.Install dependencies and current repo
```
pip install -r requirements.txt
```

## Training on your own dataset

Train a new model starting from your own dataset:
```
python3 train.py train --train_dataroot=/path/to/data/train/ --test_dataroot=/path/to/data/test/ --model_dir=/path/to/your/model/
```
evaluat a new model starting from your own dataset:
```
python3 evaluation.py train --test__path=/path/to/data/evaluation/ --model_dir=/path/to/your/model/
```

## Citation 
Use this bibtex to cite this repository:
```
@article{jiang_lib_segmentation2020,
  title={Deep-learning-based image registration for nano-resolution tomographic reconstruction},
  author={Tianyu Fu, Kai Zhang， Yan Wang, Jizhou Li, Jin Zhangab, Chunxia Yao, Qili He, Shanfeng Wang, Wanxia Huang, Qingxi Yuan, Piero Pianetta, and Yijin Liu},
  journal={Journal of Synchrotron Radiation},
  year={2021},
}
```

## Contributing
Contributions to this repository are always welcome. Examples of things you can contribute:

* Accuracy Improvements.
* Training on your own data and release the trained models.
* Visualizations and examples.

## Requirements
Python 3.7, torch 1.2.0 and other common packages listed in requirements.txt.
