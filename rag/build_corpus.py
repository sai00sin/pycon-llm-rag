from pathlib import Path
import argparse, re

def normalize(txt: str) -> str:
    txt = txt.replace('\u3000',' ').replace('\t',' ')
    txt = re.sub(r"[ ]{2,}", " ", txt)
    return txt.strip()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/docs")
    ap.add_argument("--out", default="data/docs_corpus.txt")
    args = ap.parse_args()

    parts = []
    for p in Path(args.src).rglob("*.txt"):
        parts.append(normalize(Path(p).read_text(encoding='utf-8')))
    Path(args.out).write_text("\n\n".join(parts), encoding='utf-8')
    print(f"built: {args.out}")
