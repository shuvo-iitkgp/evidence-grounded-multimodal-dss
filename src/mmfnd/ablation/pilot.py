"""Small real-data pilot: frozen CLIP encoders, trainable fusion, local evidence.

This is a feasibility experiment, NOT the manuscript's RoBERTa/VGG/T5 model.
The corpus is training post text; agreement is not independently verified truth.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from sklearn.metrics import f1_score
from transformers import (CLIPModel, CLIPTokenizerFast, CLIPImageProcessor,
                          AutoTokenizer, AutoModelForSequenceClassification)
from .fusion import dtf

CLIP = "openai/clip-vit-base-patch32"
NLI = "cross-encoder/nli-MiniLM2-L6-H768"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class FusionHead(nn.Module):
    def __init__(self, hierarchical, text_dim=512, image_dim=768, hidden=64):
        super().__init__()
        self.hierarchical = hierarchical
        self.text = nn.Linear(text_dim, hidden)
        self.image = nn.Linear(image_dim, hidden)
        self.text_attn = nn.Linear(hidden, 1)
        self.image_attn = nn.Linear(hidden, 1)
        self.cross = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(),
                                        nn.Dropout(.2), nn.Linear(hidden, 2))

    def forward(self, text, image, mask):
        text, image = torch.tanh(self.text(text)), torch.tanh(self.image(image))
        if self.hierarchical:
            weights = self.text_attn(text).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
            t = (text*weights.unsqueeze(-1)).sum(1)
            v = (image*self.image_attn(image).softmax(1)).sum(1)
            modalities = torch.stack([t, v], dim=1)
            attended, _ = self.cross(modalities, modalities, modalities, need_weights=False)
            fused = (modalities+attended).flatten(1)
        else:
            t = (text*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True)
            v = image.mean(1)
            fused = torch.cat([t, v], dim=1)
        return self.classifier(fused)


def features(frame, root, device, batch_size):
    signature = hashlib.sha256(frame[["id", "text", "image_hash"]].to_csv(index=False).encode()).hexdigest()
    cache = root/"features.npz"
    if cache.exists():
        stored = np.load(cache)
        if str(stored["signature"]) == signature:
            return {k: stored[k] for k in ("text", "image", "mask", "text_embed", "image_embed")}
    try:
        model = CLIPModel.from_pretrained(CLIP, cache_dir="data/model_cache", local_files_only=True)
        tokenizer = CLIPTokenizerFast.from_pretrained(CLIP, cache_dir="data/model_cache", local_files_only=True)
    except OSError:
        model = CLIPModel.from_pretrained(CLIP, cache_dir="data/model_cache")
        tokenizer = CLIPTokenizerFast.from_pretrained(CLIP, cache_dir="data/model_cache")
    model = model.to(device).eval()
    processor = CLIPImageProcessor()
    (root/"encoder_provenance.json").write_text(json.dumps({"model":CLIP,
        "resolved_revision":getattr(model.config,"_commit_hash",None),
        "torch":torch.__version__,"frozen":True},indent=2),encoding="utf-8")
    arrays = {k: [] for k in ("text", "image", "mask", "text_embed", "image_embed")}
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start+batch_size]
            tokens = tokenizer(batch.text.tolist(), max_length=77, truncation=True,
                               padding="max_length", return_tensors="pt").to(device)
            images = []
            for p in batch.image_path:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
            text = model.text_model(**tokens)
            vision = model.vision_model(pixel_values=pixels)
            arrays["text"].append(text.last_hidden_state.cpu().numpy().astype(np.float16))
            arrays["image"].append(vision.last_hidden_state.cpu().numpy().astype(np.float16))
            arrays["mask"].append(tokens.attention_mask.cpu().numpy().astype(bool))
            for key, value in (("text_embed", model.text_projection(text.pooler_output)),
                                ("image_embed", model.visual_projection(vision.pooler_output))):
                arrays[key].append(nn.functional.normalize(value, dim=-1).cpu().numpy())
            print(f"features {min(start+batch_size,len(frame))}/{len(frame)}", flush=True)
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    np.savez_compressed(cache, signature=signature, **arrays)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays


def train_heads(frame, arrays, root, device, seeds, epochs, batch_size):
    data = {k: torch.from_numpy(arrays[k]).to(torch.bool if k == "mask" else torch.float32)
            for k in ("text", "image", "mask")}
    y = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
    train = np.flatnonzero(frame.split == "train")
    val = np.flatnonzero(frame.split == "val")
    test = np.flatnonzero(frame.split == "test")
    def forward(model, idx):
        return model(*(data[k][idx].to(device) for k in ("text", "image", "mask")))
    scores, timings = {}, []
    for seed in seeds:
        for name, hierarchical in (("early", False), ("hier", True)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = FusionHead(hierarchical).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
            counts = np.bincount(y[train].numpy(), minlength=2)
            weights = torch.tensor(len(train)/(2*counts), dtype=torch.float32, device=device)
            best, patience, best_state, best_epoch = -1., 0, None, 0
            for epoch in range(epochs):
                model.train()
                order = np.random.permutation(train)
                for start in range(0, len(order), batch_size):
                    idx = order[start:start+batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(forward(model, idx), y[idx].to(device), weight=weights)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    pred = torch.cat([forward(model, val[i:i+batch_size]).cpu()
                                      for i in range(0, len(val), batch_size)]).argmax(-1).numpy()
                f1 = f1_score(y[val], pred, average="macro", labels=[0,1], zero_division=0)
                if f1 > best:
                    best, best_epoch, patience = f1, epoch+1, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience += 1
                if patience >= 5:
                    break
            model.load_state_dict(best_state)
            torch.save({"state": best_state, "seed": seed, "best_epoch": best_epoch,
                        "validation_macro_f1": best, "architecture": name}, root/f"{name}_{seed}.pt")
            model.eval()
            with torch.inference_mode():
                logits = torch.cat([forward(model, np.arange(i,min(i+batch_size,len(frame)))).cpu()
                                    for i in range(0,len(frame),batch_size)])
                scores[(seed,name)] = (logits[:,1]-logits[:,0]).numpy()
                # Batch-one head-only latency. Explicitly excludes frozen encoder/retrieval.
                for _ in range(10):
                    forward(model, test[:1])
                sync(device)
                for idx in test:
                    start = time.perf_counter()
                    forward(model, np.array([idx]))
                    sync(device)
                    timings.append({"seed": seed, "model": name, "id": frame.iloc[idx].id,
                                    "head_ms": 1000*(time.perf_counter()-start)})
            print(f"seed={seed} {name} best_epoch={best_epoch} val_f1={best:.4f}", flush=True)
    pd.DataFrame(timings).to_csv(root/"head_latency.csv", index=False)
    return scores


def evidence(frame, arrays, root, device, batch_size):
    cache = root/"evidence.csv"
    # Features cache signature binds image/text/order. Evidence is tied to the same manifest.
    signature = hashlib.sha256(frame[["id", "text", "image_hash", "split"]].to_csv(index=False).encode()).hexdigest()
    stamp = root/"evidence_signature.txt"
    if cache.exists() and stamp.exists() and stamp.read_text() == signature:
        return pd.read_csv(cache, dtype={"id":str})
    train = np.flatnonzero(frame.split == "train")
    query = np.flatnonzero(frame.split != "train")
    # Training captions are a local retrieval corpus, not independent fact checks.
    image_sim = arrays["image_embed"][query] @ arrays["image_embed"][train].T
    text_sim = arrays["text_embed"][query] @ arrays["text_embed"][train].T
    candidates = np.argsort(-image_sim, axis=1)[:, :5]
    rows, pairs = [], []
    for q, candidate in enumerate(candidates):
        best = candidate[np.argmax(text_sim[q,candidate])]
        source, target = frame.iloc[train[best]], frame.iloc[query[q]]
        rows.append({"id": target.id, "s_max": float(image_sim[q,candidate[0]]),
                     "text_similarity": float(text_sim[q,best]), "evidence_id": source.id,
                     "candidate_ids": json.dumps(frame.iloc[train[candidate]].id.tolist()),
                     "evidence_available": 1})
        # Evidence=premise, claim=hypothesis. No corpus labels enter this calculation.
        pairs.append((source.text, target.text))
    tokenizer = AutoTokenizer.from_pretrained(NLI, cache_dir="data/model_cache")
    model = AutoModelForSequenceClassification.from_pretrained(NLI, cache_dir="data/model_cache").to(device).eval()
    (root/"verifier_provenance.json").write_text(json.dumps({"model":NLI,
        "resolved_revision":getattr(model.config,"_commit_hash",None),
        "class_order":["contradiction","entailment","neutral"]},indent=2),encoding="utf-8")
    probs = []
    with torch.inference_mode():
        for start in range(0,len(pairs),batch_size):
            batch = pairs[start:start+batch_size]
            tokens = tokenizer([x[0] for x in batch], [x[1] for x in batch],
                               truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
            probs.append(model(**tokens).logits.softmax(-1).cpu().numpy())
    # Model card's documented class order: contradiction, entailment, neutral.
    probabilities = np.concatenate(probs)
    result = pd.DataFrame(rows)
    result[["p_ref","p_sup","p_nei"]] = probabilities
    result.to_csv(cache,index=False)
    stamp.write_text(signature)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--seeds", default="11,22,33,44,55")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--encoder-batch-size", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-evidence", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    root, source = Path(args.out), Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(source/(split+".csv"),dtype={"id":str}).assign(split=split)
              for split in ("train","val","test")]
    frame = pd.concat(frames,ignore_index=True)
    if frame.id.duplicated().any():
        raise ValueError("Repeated post IDs across splits")
    if frame.groupby("image_hash").split.nunique().max() > 1:
        raise ValueError("Exact image overlap between splits")
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    manifest = {"mode":"PILOT", "device":str(device), "torch":torch.__version__,
                "gpu":torch.cuda.get_device_name() if device.type == "cuda" else None,
                "encoders":CLIP, "frozen":True, "verifier": None if args.no_evidence else NLI,
                "retrieval":"train-only CLIP cosine top5, CLIP text reranking (not DPR)",
                "coreference":False, "args":vars(args),
                "data_manifest":json.loads((source/"manifest.json").read_text()),
                "limitations":["Not full manuscript architecture", "No independent evidence relevance/stance annotations",
                               "English encoders: Chinese Weibo performance is not a valid multilingual benchmark",
                               "Training captions are not verified external evidence", "Head-only timing excludes encoders and retrieval"]}
    (root/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    arrays = features(frame,root,device,args.encoder_batch_size)
    scores = train_heads(frame,arrays,root,device,seeds,args.epochs,args.batch_size)
    if args.no_evidence:
        ev = frame[frame.split != "train"][["id"]].copy()
        for col,value in {"s_max":0.,"p_sup":0.,"p_ref":0.,"p_nei":1.,"evidence_available":0}.items():
            ev[col] = value
    else:
        ev = evidence(frame,arrays,root,device,args.encoder_batch_size)
    histories = {}
    for row in frames[0].itertuples():
        for domain in set(json.loads(row.domains)):
            counts = histories.setdefault(domain,[0,0])
            counts[int(row.label)] += 1
    selected = frame[frame.split != "train"].copy()
    selected["dtf"], selected["domain_n"] = 0., 0.
    for idx,row in selected.iterrows():
        counts = [histories[d] for d in set(json.loads(row.domains)) if d in histories]
        if counts:
            # Mean per-domain trust; mean support avoids inflating repeated hyperlinks.
            selected.loc[idx,"dtf"] = np.mean([dtf(r,f) for f,r in counts])
            selected.loc[idx,"domain_n"] = np.mean([r+f for f,r in counts])
    outputs = []
    for seed in seeds:
        result = selected.copy()
        result["early_logit"] = scores[(seed,"early")][selected.index]
        result["hier_logit"] = scores[(seed,"hier")][selected.index]
        result = result.merge(ev,on="id",validate="one_to_one")
        result["seed"], result["dataset"] = seed, source.name
        outputs.append(result)
    pd.concat(outputs,ignore_index=True).to_csv(root/"scores.csv",index=False)
    manifest["whole_run_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    manifest["whole_run_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved() if device.type == "cuda" else None
    (root/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Saved real pilot scores to {root/'scores.csv'}",flush=True)


if __name__ == "__main__":
    main()
