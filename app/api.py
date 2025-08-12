from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from tokenizers import Tokenizer
import torch
from llm.model import MiniTransformer
from rag.retrieve import Retriever

app = FastAPI()

class AskReq(BaseModel):
    query: str
    top_k: int = 3

# Lazy load
_tok = None
_model = None
_ret = None

@app.on_event("startup")
def load_all():
    global _tok, _model, _ret
    _tok = Tokenizer.from_file("ckpts/tokenizer.json")
    ckpt = torch.load("ckpts/sft.pt", map_location="cpu")
    _model = MiniTransformer(vocab_size=ckpt['vocab_size'], max_len=ckpt['seq_len'])
    _model.load_state_dict(ckpt['model'])
    _model.eval()
    _ret = Retriever()

@app.post("/ask")
def ask(req: AskReq):
    I, D = _ret.search(req.query, k=req.top_k)
    # 参考資料（簡易表示: ファイルパスのみ）
    refs = [str(Path("ckpts/embeddings.npz")), *(map(str, I))]

    prompt = f"あなたは社内文書に詳しいアシスタントです。\n質問: {req.query}\n参考: 省略\n回答: "

    from llm.infer import generate
    out = generate(_model, _tok, prompt, device="cpu")
    return {"answer": out, "refs": refs}
