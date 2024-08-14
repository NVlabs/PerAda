# Datasets Preparation

For reference, the folder structure is:
```
data/
├── CheXpert-v1.0-small/
│   ├── train/
│   ├── train.csv
│   ├── valid/
│   ├── valid.csv
├── office_home_dg/
│   ├── art/
│   ├── clipart/
│   ├── product/
│   └── real_world/
├── cifar10.1/
│   ├── cifar10.1_v6_data.npy
│   └── cifar10.1_v6_labels.npy
├── cifar10/
├── CIFAR-10-C/
│   ├── brightness.npy
│   ├── contrast.npy
│   ├── corruptions.txt
│   ├── defocus_blur.npy
│   ├── elastic_transform.npy
│   ├── fog.npy
│   ├── frost.npy
│   ├── gaussian_blur.npy
│   ├── gaussian_noise.npy
│   ├── glass_blur.npy
│   ├── impulse_noise.npy
│   ├── jpeg_compression.npy
│   ├── labels.npy
│   ├── motion_blur.npy
│   ├── pixelate.npy
│   ├── saturate.npy
│   ├── shot_noise.npy
│   ├── snow.npy
│   ├── spatter.npy
│   ├── speckle_noise.npy
│   └── zoom_blur.npy
├── cifar100/
└── stl10/
```


## Cifar10
CIFAR-10 will be downloaded automatically.

OOD datasets:
- Download [CIFAR-10.1 dataset](https://github.com/modestyachts/CIFAR-10.1)
- Download [CIFAR-10-C dataset](https://github.com/hendrycks/robustness)


Knowledge distillation datasets will be automatically downloaded:
- CIFAR-100
- STL-10


## Office-Home
The original [Office-Home dataset](http://hemanthdv.org/OfficeHome-Dataset) is used.
When runing the training script, this dataset `office_home_dg.zip`(129M)  will be automatically downloaded from the [Google Drive link](https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa)


## CheXpert
1. Download `CheXpert-v1.0-small.zip`(11.47 GB) from [CheXpert dataset](http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip).

2. Unzip the file.
3. Create label non-i.i.d FL data distribution.
The FL data parition used in our paper is provided:
- `chexpert_partition_0.3.json`  ($\alpha$=0.3 )
- `chexpert_partition_1.json` ($\alpha$=1)

To generate FL data parition under other $\alpha$, run the following command
```
python datasets/chexpert_data_partition.py
```
Specifically, configure the dataset path in `datasets/chexpert_data_partition.py` as below:
```
root='data'
train_file=os.path.join(root, 'CheXpert-v1.0-small/train.csv')
```
Then, configure the FL dataset non-iid degree and number of users:
```
alpha=0.3 
num_users=20
```
