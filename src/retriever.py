import os, json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def load_corpus(corpus_dir, category):
    path = os.path.join(corpus_dir, f"{category}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

class BM25Retriever:
    def __init__(self, corpus):
        self.docs = [c["text"] for c in corpus]
        self.meta = corpus
        self.v = TfidfVectorizer(ngram_range=(1,2))
        self.X = self.v.fit_transform(self.docs)
    def topk(self, query, k=5):
        q = self.v.transform([query])
        scores = (self.X @ q.T).toarray().ravel()
        idx = np.argsort(-scores)[:k]
        return [self.meta[i] for i in idx]

class EmbeddingRetriever:
    def __init__(self, corpus, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.meta = corpus
        self.emb = self.model.encode([c["text"] for c in corpus], normalize_embeddings=True)
    def topk(self, query, k=5):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = (self.emb @ q)
        idx = np.argsort(-sims)[:k]
        return [self.meta[i] for i in idx]
