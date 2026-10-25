"""Extract frozen image features for one dataset.

    python scripts/extract_features.py eyepacs --backbone dinov2_s

Writes features/<dataset>_<backbone>.npz with the feature matrix (float16), the
ICDR grade, image and patient identifiers and the original split.
"""
import argparse
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import timm
from PIL import Image

from paths import DATA, FEATURES
from pacr.fundus import crop_and_pad

BACKBONES = {
    "dinov2_s": dict(model_name="vit_small_patch14_dinov2.lvd142m", img_size=224),
    "dinov2_b": dict(model_name="vit_base_patch14_dinov2.lvd142m", img_size=224),
    "resnet50": dict(model_name="resnet50.a1_in1k"),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Each reader yields (image_id, patient_id, split, grade, load_image).
def read_aptos():
    df = pq.read_table(DATA / "aptos/data/train-00000-of-00001.parquet").to_pandas()
    for _, r in df.iterrows():
        iid = r["image"]["path"].rsplit(".", 1)[0]
        yield iid, iid, "all", int(r["label_code"]), lambda b=r["image"]["bytes"]: Image.open(io.BytesIO(b))


def read_idrid():
    files = {"train": "train-00000-of-00001-d81b05cdbfbe95cd", "test": "test-00000-of-00001-3eaaa0286c6f240e"}
    for split, f in files.items():
        df = pq.read_table(DATA / f"idrid/data/{f}.parquet").to_pandas()
        for _, r in df.iterrows():
            # file names repeat between the IDRiD train and test sets
            iid = split + "_" + r["image"]["path"].rsplit(".", 1)[0]
            yield iid, iid, split, int(r["label"]), lambda b=r["image"]["bytes"]: Image.open(io.BytesIO(b))


def read_ddr():
    base = DATA / "ddr/DR_grading"
    for split in ["train", "valid", "test"]:
        rows = [ln.split() for ln in open(base / f"{split}.txt") if ln.strip()]
        for name, grade in rows:
            path = base / split / name
            if int(grade) == 5 or not path.exists():  # grade 5 = ungradable
                continue
            iid = name.rsplit(".", 1)[0]
            yield iid, iid, split, int(grade), lambda p=path: Image.open(p)


def read_eyepacs():
    labels = pd.read_csv(DATA / "eyepacs/trainLabels.csv")
    labels = dict(zip(labels["image"], labels["level"]))
    zf = zipfile.ZipFile(DATA / "eyepacs/EYEPACS.zip")
    for info in zf.infolist():
        name = info.filename.rsplit("/", 1)[-1]
        if info.is_dir() or not name.lower().endswith((".jpeg", ".jpg", ".png")):
            continue
        iid = name.rsplit(".", 1)[0]
        if iid in labels:
            patient = iid.rsplit("_", 1)[0]  # "10_left" -> "10"
            yield iid, patient, "all", int(labels[iid]), lambda i=info: Image.open(io.BytesIO(zf.read(i)))


READERS = {"aptos": read_aptos, "idrid": read_idrid, "ddr": read_ddr, "eyepacs": read_eyepacs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(READERS))
    ap.add_argument("--backbone", default="dinov2_s", choices=list(BACKBONES))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    model = timm.create_model(pretrained=True, num_classes=0, **BACKBONES[args.backbone]).eval()

    feats, meta, batch, batch_meta = [], [], [], []
    n_bad, t0 = 0, time.time()

    def flush():
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
        with torch.inference_mode():
            feats.append(model(x).float().numpy().astype(np.float16))
        meta.extend(batch_meta)
        batch.clear()
        batch_meta.clear()

    for iid, pid, split, grade, load in READERS[args.dataset]():
        try:
            img = crop_and_pad(load())
        except Exception:
            n_bad += 1
            continue
        batch.append((np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD)
        batch_meta.append((iid, pid, split, grade))
        if len(batch) == args.batch_size:
            flush()
            if len(meta) % 640 == 0:
                print(f"{len(meta)} images, {len(meta) / (time.time() - t0):.1f} img/s", flush=True)
    if batch:
        flush()

    FEATURES.mkdir(exist_ok=True)
    out = FEATURES / f"{args.dataset}_{args.backbone}.npz"
    ids, pids, splits, grades = map(np.array, zip(*meta))
    np.savez_compressed(out, feats=np.concatenate(feats), label=grades, image_id=ids,
                        patient_id=pids, split=splits)
    print(f"{args.dataset}: {len(meta)} images ({n_bad} unreadable) -> {out}")


if __name__ == "__main__":
    main()
