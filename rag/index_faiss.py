import argparse, faiss, numpy as np
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="ckpts/embeddings.npz")
    ap.add_argument("--out", default="ckpts/index.faiss")
    args = ap.parse_args()

    z = np.load(args.emb, allow_pickle=True)
    X = z["X"].astype(np.float32).toarray()  # TF-IDF → dense
    index = faiss.IndexFlatIP(X.shape[1])
    # 内積最大化 → TF-IDFはL2正規化すると良い
    faiss.normalize_L2(X)
    index.add(X)
    faiss.write_index(index, args.out)
    print(f"saved: {args.out}")
