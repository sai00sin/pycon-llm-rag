import argparse, faiss, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

class Retriever:
    def __init__(self, emb_npz="ckpts/embeddings.npz", index_path="ckpts/index.faiss"):
        z = np.load(emb_npz, allow_pickle=True)
        self.meta = z["meta"].astype(str)
        self.vocab = z["vocab"].astype(object)
        self.vec = TfidfVectorizer(vocabulary=self.vocab)
        self.index = faiss.read_index(index_path)

    def search(self, query: str, k=3):
        q = self.vec.transform([query]).astype(np.float32)
        q = q.toarray()
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        return I[0], D[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="社内Wikiのバックアップ手順")
    args = ap.parse_args()
    r = Retriever()
    I, D = r.search(args.query)
    print(I, D)
