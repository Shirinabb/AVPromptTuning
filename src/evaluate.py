import os, argparse, pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default="results/")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    
    files = [os.path.join(args.csv_dir, f) for f in os.listdir(args.csv_dir) if f.endswith(".csv")]
    if not files:
        raise SystemExit("No CSV files found in results/. Run train.py first.")

    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # DataClassification
    grp = df_all.groupby(["model", "mode"]).agg(["mean", "std"])
    grp.columns = ["acc_mean","acc_std","prec_mean","prec_std","rec_mean","rec_std","f1_mean","f1_std"]
    summary = grp.reset_index()

    out = os.path.join(args.csv_dir, "summary_table.csv")
    summary.to_csv(out, index=False)
    print("Saved:", out)
    print(summary)

