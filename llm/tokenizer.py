from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import argparse

# 学習: BPE / 読み書き: tokenizers の JSON 形式

def train_bpe(corpus_glob: str, vocab_size: int, out_path: str):
    files = [str(p) for p in Path('.').glob(corpus_glob)]
    assert files, f"No training files matched: {corpus_glob}"

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>","<bos>","<eos>"])
    tokenizer.train(files, trainer)
    tokenizer.save(out_path)
    print(f"saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/pretrain/*.txt")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--out", default="ckpts/tokenizer.json")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    train_bpe(args.train, args.vocab_size, args.out)
