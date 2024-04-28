# CS3511-01-Project
## Directory Structure
- ```data``` where you put data folder in.
- ```ckpt``` where you put trained checkpoint in.
- ```src``` dataset, augmentation method, model etc all in one folder.
- ```evaluation_pretrained_model.py``` generate masks used for submit
- ```refine_files.py``` generate nii.gz to submit
- ```generate_csv.py``` align image-mask training pair
- ```train.py``` train file
- ```train.sh``` training bash file
- ```README.md```
- ```environment.yml```

## Contents
- [Clone The Repo](#clone-the-repo)
- [Download Dataset](#download-dataset)
- [Environment](#environment)
- [Usage](#usage)

## Clone The Repo
Using the following command line can clone this repo into your machine.<br>
```bash 
git clone https://github.com/3116218257/CS3511-01-Project.git
cd CS3511-01-Project
```

## Download Dataset
You need to download the Task A dataset, you can get it from https://data.mendeley.com/datasets/s8kbw25s3x/1. If you have downloaded the zipped file, use this command. Then push the dataset downloaded into ```data```.

## Environment
Just create a virtual environment for our project using command line.<br>
```bash
conda env create -f environment.yml
```
If there still are some missing package, download manually the packages in the ```environment.yml```.

## Usage
Just simply run 
```bash
sh train.sh
```
Then, you can choose to use MedSegDiff-v2 model to generate another version of the mask, put them into right folder with right name, then
you can change the hyper-parameters, be ware of some pre-requisite package and paths. Then use ```evaluation_pretrained_model.py``` and finally ```refine_files.py```, will generate nii.gz file.