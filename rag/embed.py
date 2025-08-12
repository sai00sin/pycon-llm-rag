import argparse
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 既定は軽量なTF-IDF。必要になったら sentence-transformers をローカル導入して差し替え。

def chunk_text(text: str, chunk_size=500, stride=200):
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+chunk_size])
        i += max(1, chunk_size - stride)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--out", default="ckpts/embeddings.npz")
    ap.add_argument("--chunks", type=int, default=500)
    ap.add_argument("--stride", type=int, default=200)
    args = ap.parse_args()

    texts = []
    meta  = []
    for p in Path(args.docs).rglob("*.txt"):
        t = Path(p).read_text(encoding='utf-8')
        for ch in chunk_text(t, args.chunks, args.stride):
            texts.append(ch)
            meta.append(str(p))

    vec = TfidfVectorizer(max_features=20000)
    X = vec.fit_transform(texts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X.astype(np.float32), meta=np.array(meta), vocab=np.array(vec.get_feature_names_out(), dtype=object))
    print(f"saved: {args.out}  (shape={X.shape})")
