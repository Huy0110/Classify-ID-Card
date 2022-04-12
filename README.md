# Classify-ID-Card

# Requirement
Please install:
- timm
- torch
- tqdm.notebook
- matplotlib
- sklearn
- PIL


# Dataset

## The formated dataset: 
The formated dataset is simiar to this link : https://drive.google.com/drive/folders/1_wcXUxM5_iPQ8j9uz-PyjkWxT4YgCKM7?usp=sharing

## The original dataset:
The original dataset is similar to 3 forder in git : Train, Val, Test

Note that the name of each image in the original dataset must folow the rules: [label]_[something else] 

Run this command to process the data to the formated dataset (If you don't have the formated dataset) :
```bash
python effi_data_process.py
```
# Run
## Train
```bash
python train.py
```

## Test
```bash
python test.py
```

