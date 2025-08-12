import argparse, json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from llm.model import MiniTransformer
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_path, max_len=256):
        self.samples = [json.loads(l) for l in Path(jsonl_path).read_text(encoding='utf-8').splitlines()]
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        prompt = f"質問: {s['question']}\n回答: "
        target = s['answer']
        ids_inp = self.tok.encode(prompt).ids
        ids_out = self.tok.encode(target).ids
        ids = ids_inp + ids_out
        ids = ids[:self.max_len]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        return x, y


def finetune(base_ckpt, tokenizer, data_jsonl, out, epochs=1, batch_size=8, lr=2e-4, device='cuda'):
    ckpt = torch.load(base_ckpt, map_location=device)
    model = MiniTransformer(vocab_size=ckpt['vocab_size'], max_len=ckpt['seq_len']).to(device)
    model.load_state_dict(ckpt['model'])
    ds = QADataset(data_jsonl, tokenizer, max_len=ckpt['seq_len'])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda b: _pad_batch(b, device))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for e in range(epochs):
        pbar = tqdm(dl, desc=f"sft {e+1}/{epochs}")
        for x, y in pbar:
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    torch.save({"model": model.state_dict(), **ckpt}, out)
    print(f"saved: {out}")


def _pad_batch(batch, device):
    xs, ys = zip(*batch)
    T = max(x.size(0) for x in xs)
    pad = lambda t: torch.cat([t, torch.zeros(T - t.size(0), dtype=torch.long)], dim=0)
    X = torch.stack([pad(x) for x in xs]).to(device)
    Y = torch.stack([pad(y) for y in ys]).to(device)
    return X, Y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ckpts/pretrain.pt")
    ap.add_argument("--tokenizer", default="ckpts/tokenizer.json")
    ap.add_argument("--data", default="data/finetune/qa.jsonl")
    ap.add_argument("--out", default="ckpts/sft.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    finetune(args.base, args.tokenizer, args.data, args.out, args.epochs, args.batch_size, args.lr, args.device)
