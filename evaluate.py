import pandas as pd, numpy as np, os, argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json, random

parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir", default="results/")
parser.add_argument("--runs", type=int, default=10)
args = parser.parse_args()

files = [os.path.join(args.csv_dir, f) for f in os.listdir(args.csv_dir) if f.endswith(".csv")]
all_runs = []
for f in files:
    df = pd.read_csv(f)
    all_runs.append(df)

df_all = pd.concat(all_runs)
summary = df_all.groupby(["model","mode"]).agg(["mean","std"]).reset_index()
summary.columns = ["model","mode","acc_mean","acc_std","prec_mean","prec_std","rec_mean","rec_std","f1_mean","f1_std"]
summary.to_csv(os.path.join(args.csv_dir,"summary_table7.csv"),index=False)
print(summary)
