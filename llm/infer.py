import torch, argparse
from tokenizers import Tokenizer
from llm.model import MiniTransformer

@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=128, temperature=0.8, top_k=40, device='cpu'):
    ids = tok.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]
        logits = logits / max(1e-6, temperature)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    return tok.decode(x[0].tolist())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/sft.pt")
    ap.add_argument("--tokenizer", default="ckpts/tokenizer.json")
    ap.add_argument("--prompt", default="質問: このプロジェクトの目的は？\n回答: ")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    model = MiniTransformer(vocab_size=ckpt['vocab_size'], max_len=ckpt['seq_len']).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    out = generate(model, tok, args.prompt, device=args.device)
    print(out)
