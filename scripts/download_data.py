"""Download the four public datasets from their Hugging Face mirrors.

    python scripts/download_data.py aptos idrid eyepacs ddr

DDR is large (about 15 GB at full resolution), so its images are fetched one at a
time, cropped and stored as 512 px JPEGs, and the originals are deleted. The DDR
step can be interrupted and re-run; finished images are skipped.
"""
import argparse
import os
import time

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from paths import DATA
from pacr.fundus import crop_and_pad

DDR_REPO = "ctmedtech/DDR-dataset"


def get(repo, filename, out):
    hf_hub_download(repo, filename, repo_type="dataset", local_dir=out)


def download_aptos():
    get("bumbledeep/aptos", "data/train-00000-of-00001.parquet", DATA / "aptos")


def download_idrid():
    for f in ["data/train-00000-of-00001-d81b05cdbfbe95cd.parquet",
              "data/test-00000-of-00001-3eaaa0286c6f240e.parquet"]:
        get("amin-nejad/idrid-disease-grading", f, DATA / "idrid")


def download_eyepacs():
    for f in ["trainLabels.csv", "EYEPACS.zip"]:
        get("ctmedtech/EYEPACS", f, DATA / "eyepacs")


def download_ddr(retries=8):
    raw, out = DATA / "ddr_raw", DATA / "ddr"
    api = HfApi()
    files = []
    for split in ["train", "valid", "test"]:
        tree = api.list_repo_tree(DDR_REPO, repo_type="dataset", path_in_repo=f"DR_grading/{split}")
        files += [f.path for f in tree if f.path.lower().endswith((".jpg", ".jpeg", ".png"))]
        get(DDR_REPO, f"DR_grading/{split}.txt", out)
    print(f"DDR: {len(files)} images")

    for i, path in enumerate(files, 1):
        dst = out / path
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            try:
                src = hf_hub_download(DDR_REPO, path, repo_type="dataset", local_dir=raw)
                crop_and_pad(Image.open(src), size=512).save(dst, quality=95)
                os.remove(src)
                break
            except Exception as e:  # rate limits and truncated downloads
                print(f"retry {path}: {e!r}"[:200])
                time.sleep(min(30 * (attempt + 1), 180))
        if i % 500 == 0:
            print(f"DDR {i}/{len(files)}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    jobs = {"aptos": download_aptos, "idrid": download_idrid,
            "eyepacs": download_eyepacs, "ddr": download_ddr}
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=list(jobs))
    for name in ap.parse_args().datasets:
        jobs[name]()
        print(name, "done")
