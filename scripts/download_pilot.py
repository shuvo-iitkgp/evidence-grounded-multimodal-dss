"""Download official benchmark metadata and small, reproducible image subsets.

Run from repository root. Raw data stays in ignored data/. No fabricated images.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
import requests
from sklearn.model_selection import train_test_split

RAW = "https://raw.githubusercontent.com/"


def download(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=90, stream=True)
    response.raise_for_status()
    tmp = path.with_suffix(path.suffix+".part")
    with tmp.open("wb") as f:
        for chunk in response.iter_content(1024*1024):
            f.write(chunk)
    tmp.replace(path)
    return path


def domains(text):
    # Shorteners/CDNs/platform hosts are not external publisher histories.
    excluded = {"t.co", "bit.ly", "twitter.com", "weibo.com", "reddit.com",
                "i.redd.it", "i.imgur.com", "imgur.com", "pic.twitter.com", "t.cn", "url.cn"}
    hosts = {urlparse(x).hostname for x in re.findall(r'https?://[^\s<>]+', str(text))}
    return json.dumps(sorted(h for h in hosts if h and h not in excluded))


def source_domain(value):
    """Fakeddit original submission-domain metadata, not its thumbnail host."""
    if pd.isna(value):
        return "[]"
    host = str(value).strip().lower()
    excluded = ("redd.it","reddit.com","reddituploads.com","imgur.com","twimg.com",
                "gfycat.com","youtube.com","youtu.be","web.archive.org","archive.is",
                "500px.org","sli.mg")
    if "." not in host or host.startswith("self.") or any(host == d or host.endswith("."+d) for d in excluded):
        return "[]"
    return json.dumps([host])


def fetch_image(row, root):
    path = root/"images"/(str(row["id"])+".jpg")
    try:
        if not path.exists():
            url = row["image_url"].replace("http://", "https://", 1)
            r = requests.get(url, timeout=12, headers={"User-Agent": "research-pilot/1.0"})
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            if min(im.size) < 32:
                raise ValueError("Image too small")
            im.thumbnail((512, 512))
            im.save(path, quality=90)
        with Image.open(path) as im:
            im.verify()
        row["image_path"] = str(path.resolve())
        row["image_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return row
    except Exception:
        return None


def limit(frame, n, seed=2026):
    if len(frame) <= n:
        return frame.sample(frac=1, random_state=seed)
    selected, _ = train_test_split(frame, train_size=n, stratify=frame.label, random_state=seed)
    return selected


def internet_subset(frame, n, root):
    # Candidate pool is chosen before download failures; log attrition explicitly.
    pool = limit(frame, min(len(frame), n*5))
    quotas = {1: int(round(n*frame.label.mean()))}
    quotas[0] = n-quotas[1]
    rows, seen, attempted = [], set(), 0
    retained = {0:0,1:0}
    candidates = pool.to_dict("records")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0,len(candidates),16):
            batch = [r for r in candidates[start:start+16] if retained[r["label"]] < quotas[r["label"]]]
            attempted += len(batch)
            for row in executor.map(lambda r: fetch_image(r, root),batch):
                if row and row["image_hash"] not in seen and retained[row["label"]] < quotas[row["label"]]:
                    rows.append(row)
                    seen.add(row["image_hash"])
                    retained[row["label"]] += 1
            if start % 160 == 0:
                print(f"images retained={len(rows)}/{n}, attempted={attempted}",flush=True)
            if len(rows) >= n:
                break
    available = pd.DataFrame(rows)
    print(f"download: {len(rows)}/{attempted} retained unique images", flush=True)
    if available.empty or available.label.nunique() < 2:
        raise RuntimeError("Not enough accessible images from both classes")
    available = available.drop_duplicates("image_hash")
    return limit(available, min(n, len(available)))


def fakeddit(root, sizes):
    ids = {"train": "1XsOkD3yhxgWu8URMes0S9LVSwuFJ4pT6",
           "val": "1Z99QrwpthioZQY2U6HElmnx8jazf7-Kv",
           "test": "1p9EewIKVcFbipVRLZNGYc2JbSC7A0SWv"}
    output = {}
    for split, file_id in ids.items():
        path = download(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t", root/(split+".tsv"))
        frame = pd.read_csv(path, sep="\t", dtype={"id": str},
                            usecols=["id","clean_title","domain","6_way_label","image_url"])
        # Explicit Fakeddit mapping: 6-way category 0 is true, other categories false.
        frame["label"] = (frame["6_way_label"] == 0).astype(int)
        frame["text"] = frame.clean_title.fillna("")
        frame["domains"] = frame.domain.map(source_domain)
        frame = frame[frame.image_url.notna() & frame.text.str.len().gt(0)]
        output[split] = internet_subset(frame[["id", "text", "label", "image_url", "domains"]], sizes[split], root)
    return output


def weibo(root, sizes):
    output = {}
    for split in ("train", "test"):
        rows = []
        for kind, label in (("rumor", 0), ("nonrumor", 1)):
            path = download(RAW+f"wangzhuang1911/Weibo-dataset/main/{split}_{kind}.txt", root/f"{split}_{kind}.txt")
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            for start in range(0, len(lines)-2, 3):
                urls = [u for u in lines[start+1].split("|") if u.startswith("http")]
                if urls and lines[start+2].strip():
                    rows.append({"id": lines[start].split("|")[0], "text": lines[start+2],
                                 "image_url": urls[0], "label": label, "domains": domains(lines[start+2])})
        n = sizes[split]+(sizes["val"] if split == "train" else 0)
        output[split] = internet_subset(pd.DataFrame(rows).drop_duplicates("id"), n, root)
    train, val = train_test_split(output["train"], test_size=sizes["val"],
                                  stratify=output["train"].label, random_state=2026)
    output.update(train=train, val=val)
    return output


def mediaeval(root, sizes):
    output = {}
    for split, folder, archive in (("train", "devset", "MediaEval2015_DevSet_Images.rar"),
                                    ("test", "testset", "MediaEval2015_TestSetImages.rar")):
        base = RAW+f"MKLab-ITI/image-verification-corpus/master/mediaeval2015/{folder}/"
        path = download(base+"tweets.txt", root/(split+".tsv"))
        rar = download(base+archive, root/archive)
        target = root/folder
        target.mkdir(exist_ok=True)
        if not list(target.rglob("*.jpg")):
            subprocess.run(["tar", "-xf", str(rar), "-C", str(target)], check=True)
        images = {p.stem: p for p in target.rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")}
        frame = pd.read_csv(path, sep="\t", dtype=str)
        rows = []
        for row in frame.to_dict("records"):
            if row["label"] not in ("real", "fake"):
                continue
            image_ids = re.split(r'[,\s]+', row.get("imageId(s)", row.get("imageId", "")))
            image = next((images[x] for x in image_ids if x in images), None)
            if image is None:
                continue
            rows.append({"id": row["tweetId"], "text": row["tweetText"], "label": int(row["label"] == "real"),
                         "domains": domains(row["tweetText"]), "image_path": str(image.resolve()),
                         "image_hash": hashlib.sha256(image.read_bytes()).hexdigest()})
        output[split] = pd.DataFrame(rows).drop_duplicates("id")
    # Split dev images into train/validation groups to avoid identical-image leakage.
    groups = output["train"].drop_duplicates("image_hash")
    _, val_groups = train_test_split(groups, test_size=.2, stratify=groups.label, random_state=2026)
    is_val = output["train"].image_hash.isin(val_groups.image_hash)
    output["val"] = output["train"][is_val]
    output["train"] = output["train"][~is_val]
    return {split: limit(frame, sizes[split]) for split, frame in output.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["mediaeval", "fakeddit", "weibo"])
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    args = ap.parse_args()
    root = Path("data/pilot")/args.dataset
    (root/"images").mkdir(parents=True, exist_ok=True)
    sizes = {s: getattr(args, s) for s in ("train", "val", "test")}
    output = globals()[args.dataset](root, sizes)
    # Keep official test unchanged; remove duplicate IDs/images from earlier splits.
    seen_ids, seen_images = set(), set()
    manifest = {"dataset": args.dataset, "subset_seed": 2026, "requested": sizes, "splits": {}}
    for split in ("test", "val", "train"):
        frame = output[split]
        frame = frame[~frame.id.isin(seen_ids) & ~frame.image_hash.isin(seen_images)]
        if frame.label.nunique() != 2:
            raise RuntimeError(f"{split}: both classes required after deduplication")
        seen_ids.update(frame.id)
        seen_images.update(frame.image_hash)
        frame.sort_values("id").to_csv(root/(split+".csv"), index=False)
        manifest["splits"][split] = {"n": len(frame), "real": int(frame.label.sum()),
            "unique_images": int(frame.image_hash.nunique()),
            "sha256": hashlib.sha256((root/(split+".csv")).read_bytes()).hexdigest()}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
