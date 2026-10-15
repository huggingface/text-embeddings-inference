import json, torch, numpy as np, torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

MODEL="BAAI/bge-m3"
TEXTS=["What is machine learning?",
       "Кузьма — корпоративный ассистент на Go.",
       "hybrid retrieval combines dense and sparse vectors"]

# ---- independent reimpl (what the candle port will do) ----
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModel.from_pretrained(MODEL); model.eval()
sd=torch.load(hf_hub_download(MODEL,"sparse_linear.pt"),map_location="cpu",weights_only=True)
W,b=sd["weight"].float(), sd["bias"].float()           # [1,1024], [1]
print(f"[sparse_linear] weight{tuple(W.shape)} bias{tuple(b.shape)} w[0,:4]={W[0,:4].tolist()} bias={b.item():.6f}")
UNUSED={tok.cls_token_id,tok.eos_token_id,tok.pad_token_id,tok.unk_token_id}
print(f"[special ids] cls={tok.cls_token_id} eos={tok.eos_token_id} pad={tok.pad_token_id} unk={tok.unk_token_id}")

def mine(text):
    enc=tok(text,return_tensors="pt",truncation=True,max_length=512)
    with torch.no_grad(): h=model(**enc).last_hidden_state[0]      # [L,1024]
    tw=torch.relu(h@W.t()+b).squeeze(-1)                            # [L]
    ids=enc["input_ids"][0].tolist()
    res=defaultdict(float)
    for w,idx in zip(tw.tolist(),ids):
        if idx not in UNUSED and w>0 and w>res[str(idx)]: res[str(idx)]=w
    dense=F.normalize(h[0],dim=-1)                                  # CLS + L2
    return dict(res), dense.numpy()

# ---- reference (FlagEmbedding) ----
from FlagEmbedding import BGEM3FlagModel
ref=BGEM3FlagModel(MODEL,use_fp16=False)
out=ref.encode(TEXTS,return_dense=True,return_sparse=True,batch_size=1,max_length=512)

golden=[]
print("\n=== PARITY (mine vs FlagEmbedding) ===")
for i,t in enumerate(TEXTS):
    m_lex,m_dense=mine(t)
    r_lex={k:float(v) for k,v in out["lexical_weights"][i].items()}
    r_dense=np.asarray(out["dense_vecs"][i],dtype=np.float32)
    ks_m,ks_r=set(m_lex),set(r_lex)
    jac=len(ks_m&ks_r)/max(1,len(ks_m|ks_r))
    vdiff=max((abs(m_lex[k]-r_lex[k]) for k in ks_m&ks_r),default=0.0)
    cos=float(np.dot(m_dense,r_dense)/(np.linalg.norm(m_dense)*np.linalg.norm(r_dense)))
    print(f"[{i}] {t[:40]!r:42} idx:{len(ks_m)}/{len(ks_r)} Jaccard={jac:.4f} maxΔval={vdiff:.2e} denseCos={cos:.6f}")
    golden.append({"text":t,"lexical_weights":r_lex,"dense_head5":r_dense[:5].tolist()})

json.dump({"model":MODEL,"spec":"relu(sparse_linear(h)); drop{cls,eos,pad,unk},w>0; max-dedup by token_id",
           "cases":golden}, open("m3_golden.json","w"), ensure_ascii=False, indent=1)
print("\ngolden fixture -> m3_golden.json (", len(golden), "cases )")
