# pycon-llm-rag — Minimal, Local, API-free Starter Kit

外部APIを使わず、ローカル完結で「自作ミニLLM × シンプルRAG」デモを動かすための最小構成。
まずは動く最小限、その後に拡張してください。

## クイックスタート
1) `pip install -r requirements.txt`
2) `python llm/tokenizer.py --train data/pretrain/*.txt --vocab_size 16000 --out ckpts/tokenizer.json`
3) `python llm/train.py --tokenizer ckpts/tokenizer.json --data data/pretrain/*.txt --epochs 1 --out ckpts/pretrain.pt`
4) `python llm/finetune.py --base ckpts/pretrain.pt --tokenizer ckpts/tokenizer.json --data data/finetune/qa.jsonl --epochs 1 --out ckpts/sft.pt`
5) `python rag/embed.py --docs data/docs --out ckpts/embeddings.npz` → `python rag/index_faiss.py --emb ckpts/embeddings.npz --out ckpts/index.faiss`
6) `uvicorn app.api:app --reload` と `streamlit run app/ui.py`
