# SNSR-GAN
A PyTorch(python 3.6) and OpenCV test implementation of SNSRGAN
Super-resolution Pulmonary CT synthesis using generative adversarial nets with spectral normalization (under review).

## Requirements
- PyTorch
- tqdm
- opencv
- os

## Datasets and trained models

### Chest X-ray 2017 Dataset
Download the dataset from [here](https://data.mendeley.com/datasets/rscbjbr9sj/3), and then extract it into `data` directory.

### Chest X-ray 14 Dataset
Download the dataset from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC), and then extract it into `data` directory.

### Trained models
Download the trained models from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC), and then extract it into `trained_models` directory.

## Usage

### Test Benchmark Datasets
'''
python test_benchmark1.py         test the trained SNSR-GAN 
python test_benchmark2.py         test the trained SNSR-GAN variant without spectral normalization or label information
'''
The output super resolution images are on `benchmark_results` directory.

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

