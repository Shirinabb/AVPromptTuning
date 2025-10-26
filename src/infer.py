import os, argparse
from prompt_router import Router
from retriever import load_corpus, BM25Retriever, EmbeddingRetriever

def read_template(prompts_dir, cat):
    path = os.path.join(prompts_dir, f"{cat}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(tpl, command_text, context_text, retrieved):
    snips = "\n".join([f"- {r['text']}" for r in retrieved])
    return tpl.format(
        command_text=command_text,
        context_text=context_text or "",
        retrieved_snippets=snips
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router_ckpt", required=True)
    ap.add_argument("--prompts_dir", default="prompts")
    ap.add_argument("--corpus_dir", default="corpus")
    ap.add_argument("--retriever", default="embedding", choices=["embedding","bm25"])
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--command", required=True)
    ap.add_argument("--context", default="")
    args = ap.parse_args()

    router = Router(ckpt_dir=args.router_ckpt, device=args.device)
    cat = router.predict(args.command, args.context)
    print(f"[ROUTER] category={cat}")

    corpus = load_corpus(args.corpus_dir, cat)
    retr = EmbeddingRetriever(corpus) if args.retriever=="embedding" else BM25Retriever(corpus)
    retrieved = retr.topk(args.command + " " + args.context, k=args.top_k)

    tpl = read_template(args.prompts_dir, cat)
    final_prompt = build_prompt(tpl, args.command, args.context, retrieved)

    print("\n===== FINAL PROMPT =====\n")
    print(final_prompt)

if __name__ == "__main__":
    main()
