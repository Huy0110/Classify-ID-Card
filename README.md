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

Run this command to process the data to the formated dataset (If you have the original dataset) :
```bash
python effi_data_process.py
```

If you want to controll the number of data in each forder, you can run this command (If you have the original dataset):
```bash
python effi_data_process_controll.py --number_of_train 1000 --number_of_val 120 --number_of_train 120
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

# References
https://colab.research.google.com/drive/172TfOusbVN4y6f3sZGgTovSpVA53Aa9u?usp=sharing#scrollTo=Y0mKfl38NYkb

