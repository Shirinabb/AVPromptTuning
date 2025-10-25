# AVPromptTuning

This repository provides training and evaluation scripts 
for context-aware and priority-aware prompt tuning of transformer models.  
Example datasets are included for demonstration and educational purposes.

## How to Run
```bash
pip install -r requirements.txt
python src/train.py --config config.yaml --train_path data/train_mimic2000_hist.csv --test_path data/test_mimic40_hist.csv
