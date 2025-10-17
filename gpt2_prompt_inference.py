
import argparse, json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

def score_priority(safety:int, urgency:str, alpha=0.7, beta=0.3):
    s = float(safety)
    u_map = {"normal":0.0, "urgent":0.5, "critical":1.0}
    u = u_map.get(urgency, 0.0)
    return alpha*s + beta*u

# Example inference using a lightweight GPT-2 model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="augmented.jsonl")
    ap.add_argument("--bert", required=False, help="optional path to fine-tuned BERT for safety classification")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)
    args = ap.parse_args()

    tok_gpt = AutoTokenizer.from_pretrained("gpt2")
    tok_gpt.pad_token = tok_gpt.eos_token
    gpt = AutoModelForCausalLM.from_pretrained("gpt2")

    clf = None
    tok_clf = None
    if args.bert:
        tok_clf = AutoTokenizer.from_pretrained(args.bert)
        clf = AutoModelForSequenceClassification.from_pretrained(args.bert).eval()

    def build_prompt(ex):
        safety = ex["safety"]
        urgency = ex["labels"]["urgency"]
        prio = score_priority(safety, urgency, args.alpha, args.beta)
        tags = f"[WEATHER={ex['labels']['weather']}][TRAFFIC={ex['labels']['traffic_density']}][ROAD={ex['labels']['road']}]"
        priotag = "SAFETY=HIGH" if prio >= 0.6 else "SAFETY=LOW"
        return f"{tags}[{priotag}] USER: {ex['command']}\nASSISTANT:"

    with open(args.data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            # optional re-classify safety with BERT if provided
            if clf is not None:
                inputs = tok_clf(ex["command"], return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = clf(**inputs).logits
                    pred = int(logits.argmax(-1).item())
                ex["safety"] = pred

            prompt = build_prompt(ex)
            inputs = tok_gpt(prompt, return_tensors="pt")
            out = gpt.generate(**inputs, max_new_tokens=40, do_sample=True, top_p=0.9, temperature=0.7)
            text = tok_gpt.decode(out[0], skip_special_tokens=True)
            print("="*80)
            print(text)

if __name__ == "__main__":
    main()
