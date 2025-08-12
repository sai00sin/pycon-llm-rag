import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from llm.model import MiniTransformer
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, paths, tokenizer_path, seq_len=256):
        self.files = []
        for p in paths:
            self.files += Path('.').glob(p)
        assert self.files, "No data files found"
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.seq_len = seq_len
        self.buf = []
        for f in self.files:
            text = Path(f).read_text(encoding='utf-8')
            ids = self.tok.encode(text).ids
            self.buf += ids

    def __len__(self):
        return max(0, len(self.buf)-self.seq_len-1)

    def __getitem__(self, i):
        x = torch.tensor(self.buf[i:i+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.buf[i+1:i+1+self.seq_len], dtype=torch.long)
        return x, y


def train(tokenizer, data_glob, out, epochs=1, batch_size=16, seq_len=256, lr=3e-4, device='cuda'):
    ds = TextDataset([data_glob], tokenizer, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    vocab_size = Tokenizer.from_file(tokenizer).get_vocab_size()
    model = MiniTransformer(vocab_size=vocab_size, max_len=seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for e in range(epochs):
        pbar = tqdm(dl, desc=f"epoch {e+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab_size": vocab_size, "seq_len": seq_len}, out)
    print(f"saved: {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="ckpts/tokenizer.json")
    ap.add_argument("--data", default="data/pretrain/*.txt")
    ap.add_argument("--out", default="ckpts/pretrain.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    train(args.tokenizer, args.data, args.out, args.epochs, args.batch_size, args.seq_len, args.lr, args.device)
