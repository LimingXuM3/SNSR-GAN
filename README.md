# SNSR-GAN
A PyTorch(python 3.6) and OpenCV test implementation of the paper 'Low-dose Pulmonary CT Super-resolution using Generative Adversarial Nets with Spectral Normalization'.

## Requirements
- PyTorch
- tqdm
- opencv
- os

## Datasets and trained models

### Chest X-ray 2017 Dataset
Download the dataset from [here](https://data.mendeley.com/datasets/rscbjbr9sj/3), and then extract it into `data/train` directory.

### Chest X-ray 14 Dataset
Download the dataset from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC), and then extract it into `data/train` directory.

### Trained models and testing data
Download the trained models and testing data from [here](https://pan.baidu.com/s/1PXVnLlGv_tvBXeGjGCuLdQ) password: 3gtz , and extract them into corresponding directory.

## Usage

### Test Single Image
'''
python test_image1.py         test the trained SNSR-GAN 

python test_image2.py         test the trained SNSR-GAN variant without spectral normalization or label information
'''

The output super resolution image are on the same directory.

### Evaluate the SSIM, FSIM and MSSIM
'''
run 'FeatureSIM.m' in the plat of Matlab         compute the FSIM value (set the same channal)

python msim.py                                   compute the MSIM value 

python ssim.py                                   compute the SSIM value
'''

## Citation
@article{2020Low,

  title={Low-dose chest X-ray image super-resolution using generative adversarial nets with spectral normalization},
  
  author={ Xu, Liming and Zeng, Xianhua and Huang, Zhiwei and Li, Wweisheng and Zhang, He},
  
  journal={Biomedical Signal Processing and Control},
  
  volume={55},
  
  pages={101600},
  
  year={2020},
  
}
