
import argparse, json, random, pandas as pd, numpy as np, os
from pathlib import Path

TRAFFIC = ["low","medium","high"]
WEATHER = ["clear","rainy","foggy","snowy"]
URGENCY = ["normal","urgent","critical"]
ROAD = ["highway","urban","rural"]

SAFETY_KEYWORDS = ["stop", "brake", "emergency", "pedestrian", "collision", "bicycle", "blind spot", "slow down"]

def is_safety(command: str) -> int:
    c = command.lower()
    for k in SAFETY_KEYWORDS:
        if k in c:
            return 1
    return 0

def infer_urgency(command: str) -> str:
    c = command.lower()
    if any(w in c for w in ["emergency","now","immediately","urgent","brake"]):
        return "critical"
    if any(w in c for w in ["please","asap","hurry"]):
        return "urgent"
    return "normal"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns: id,command")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    df = pd.read_csv(args.input)
    rows = []
    for _, r in df.iterrows():
        cmd = str(r["command"])
        rid = str(r["id"])
        record = {
            "id": rid,
            "command": cmd,
            "labels": {
                "traffic_density": random.choice(TRAFFIC),
                "weather": random.choice(WEATHER),
                "urgency": infer_urgency(cmd),
                "road": random.choice(ROAD),
            },
            "safety": is_safety(cmd),  # 1=safety-critical, 0=non-safety
        }
        rows.append(record)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} records to {out_path}")

if __name__ == "__main__":
    main()
