

## Overview
This repository contains training and evaluation scripts for transformer-based models 
(e.g., BERT, GPT-2) used in the experiments described in the paper.  
The code demonstrates how **context-aware** and **priority-aware prompt tuning** 
can improve model understanding of natural-language driving commands.
The dataset included here is a small synthetic subset inspired by the Talk2Car commands to illustrate the experimental setup for reproducibility and educational use.


##  How to Run
```bash
pip install -r requirements.txt
python src/train.py --config config.yaml --train_path data/train_mimic2000_hist.csv --test_path data/test_mimic40_hist.csv

