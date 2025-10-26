

## Overview
This repository contains training and evaluation scripts for transformer-based models 
(e.g., BERT, GPT-2).  
The code demonstrates how **context-aware** and **priority-aware prompt tuning** 
can improve model understanding of natural-language driving commands.
The dataset included here is a small synthetic subset inspired by the Talk2Car commands to illustrate the experimental setup for reproducibility and educational use.

Please keep in mind that this implementation is for reproducibility only and the dataset provided is a sample of the tested data. The general process of text changes and labeling type are also mentioned in other files
##  How to Run
#please config the salmon setting, Please pay attention to these settings.
```bash
pip install -r requirements.txt
python src/train.py --config config.yaml --train_path data/train_mimic2000_hist.csv --test_path data/test_mimic40_hist.csv

.
