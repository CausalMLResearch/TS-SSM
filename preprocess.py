#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS-SSM preprocessing.
Prepare chunked parquet data, embeddings, temporal bins, and train-only statistics.
"""

import os
import gc
import json
import glob
import pickle
import shutil
import warnings
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset, DownloadConfig

warnings.filterwarnings("ignore")

# Config

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

CATEGORY = "Toys_and_Games"
DATASET_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
OUTPUT_DIR = "./preprocessed_sequential"
TEMP_DIR = "./temp_chunks"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Core filtering (10-core)
MIN_USER_REVIEWS = 10
MIN_ITEM_REVIEWS = 10

# Time filtering
MIN_YEAR = 2014  # Discard events before this year

# Chunking
CHUNK_SIZE = 100_000

# MNAR heuristics
IMAGE_UNUSUAL_THRESHOLD = 0.10
TEXT_LENGTH_UNUSUAL_ALPHA = 2.0

# Embeddings
USE_TEXT_EMB = True
USE_IMAGE_EMB = True

IMAGE_CACHE_ENABLED = True
IMAGE_CACHE_PATH = os.path.join(OUTPUT_DIR, "image_emb_cache.pkl")

TITLE_EMB_DIM = 384
TEXT_EMB_DIM = 384
IMAGE_EMB_DIM = 512

EMB_DTYPE = np.float16
TEXT_EMB_BATCH_SIZE = 512

# Output control
KEEP_RAW_TEXT = False
KEEP_IMAGE_URL = False
KEEP_DATETIME_STR = False

# Cleanup
KEEP_TEMP_DIR = False

# Reserved padding index
PAD_IDX = 0

# Time bin and group config
# Group thresholds for image_rate
GROUP_THRESHOLDS = [0.0, 0.2, 1.01]  # never, rare, frequent
GROUP_NAMES = ["never", "rare", "frequent"]
N_GROUPS = len(GROUP_NAMES)

# Shrinkage parameter
SHRINKAGE_LAMBDA = 50

# Last merged bin (2023-04 to 2023-08 -> one bin)
LAST_REGULAR_MONTH = "2023-03"

download_config = DownloadConfig(
    max_retries=20,
    resume_download=True,
    force_download=False,
)

# Global model cache

_text_model = None
_clip_model = None
_clip_preprocess = None

def get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Loading Sentence-BERT (all-MiniLM-L6-v2) on {device}...")
        _text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    return _text_model

def get_clip_model():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import torch
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Loading CLIP (ViT-B/32) on {device}...")
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_model.eval()
    return _clip_model, _clip_preprocess

# Helpers

def simple_tokenize(text: str) -> int:
    if not text:
        return 0
    return len(text.split())

def extract_review_record(ex: Dict) -> Optional[Dict]:
    user_id = ex.get("user_id")
    item_id = ex.get("asin")
    rating = ex.get("rating")
    timestamp = ex.get("timestamp")
    if not all([user_id, item_id, rating is not None, timestamp is not None]):
        return None

    title = (ex.get("title", "") or "").strip()
    text = (ex.get("text", "") or "").strip()
    images = ex.get("images", []) or []

    helpful_vote = ex.get("helpful_vote", 0) or 0
    verified_purchase = ex.get("verified_purchase", False)

    has_title = 1 if len(title) > 0 else 0
    has_text = 1 if len(text) > 0 else 0
    has_image = 1 if (images and len(images) > 0) else 0
    num_images = min(len(images), 10) if images else 0

    image_url = ""
    if has_image:
        first_img = images[0]
        if isinstance(first_img, dict):
            image_url = first_img.get("large_image_url", "") or ""

    return {
        "user_id": user_id,
        "item_id": item_id,
        "rating": float(rating),
        "timestamp": int(timestamp),
        "title": title,
        "text": text,
        "image_url": image_url,
        "has_title": has_title,
        "has_text": has_text,
        "has_image": has_image,
        "num_images": num_images,
        "len_title_tokens": simple_tokenize(title),
        "len_text_tokens": simple_tokenize(text),
        "helpful_vote": int(helpful_vote),
        "verified_purchase": int(bool(verified_purchase)),
    }

def process_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps and filter by MIN_YEAR."""
    if len(df) == 0:
        return df

    median_ts = df["timestamp"].median()
    ts_unit = "ms" if median_ts > 1e11 else "s"
    df["datetime"] = pd.to_datetime(df["timestamp"], unit=ts_unit, errors="coerce")

    now = pd.Timestamp.now()
    valid = (
        df["datetime"].notna() &
        (df["datetime"] <= now) &
        (df["datetime"] >= pd.Timestamp("1995-01-01"))
    )
    df = df[valid].copy()
    if len(df) == 0:
        return df

    # Filter by minimum year.
    df["year"] = df["datetime"].dt.year.astype(np.int16)
    df = df[df["year"] >= MIN_YEAR].copy()
    if len(df) == 0:
        return df

    df["unix_time"] = (df["datetime"].astype("int64") // 10**9).astype(np.int64)
    df["month"] = df["datetime"].dt.month.astype(np.int8)
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype(np.int8)

    # Binary image-availability pattern used by downstream features.
    df["missing_pattern"] = (1 - df["has_image"]).astype(np.int8)

    # 3-modality pattern (T(4)+X(2)+I(1))
    df["missing_pattern_tti"] = (
        df["has_title"].astype(np.int8) * 4
        + df["has_text"].astype(np.int8) * 2
        + df["has_image"].astype(np.int8)
    ).astype(np.int8)

    return df

def compute_time_bin(year: int, month: int) -> str:
    """Compute time bin string. 2023-04+ merged into one bin."""
    ym = f"{year}-{month:02d}"
    if ym > LAST_REGULAR_MONTH:
        return "2023-Q2+"
    return ym

def atomic_save_npy(path: str, arr: np.ndarray):
    tmp = path + ".tmp"
    np.save(tmp, arr)
    if not tmp.endswith(".npy"):
        tmp = tmp + ".npy"
    os.replace(tmp, path)

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

def clean_glob(pattern: str, base: str):
    for f in glob.glob(os.path.join(base, pattern)):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass


def _load_streaming_reviews_dataset(category: str):
    """
    Load Amazon-Reviews-2023 in streaming mode, using the configured dataset
    endpoint when available and direct category shards otherwise.
    """
    config_name = f"raw_review_{category}"

    def _load_script_based(use_trust_remote_code: bool):
        kwargs = {
            "split": "full",
            "streaming": True,
            "download_config": download_config,
        }
        if use_trust_remote_code:
            kwargs["trust_remote_code"] = True
        return load_dataset(DATASET_REPO_ID, config_name, **kwargs)

    import inspect
    supports_trust_remote_code = "trust_remote_code" in inspect.signature(load_dataset).parameters
    script_errors = []
    if supports_trust_remote_code:
        for use_trust_remote_code in (True, False):
            try:
                return _load_script_based(use_trust_remote_code=use_trust_remote_code)
            except Exception as e:
                script_errors.append(e)
                msg = str(e)
                if use_trust_remote_code and "trust_remote_code" in msg:
                    print("   [WARN] `datasets` does not accept `trust_remote_code`; retrying without it.")

        if len(script_errors) > 0:
            first = script_errors[0]
            print(
                f"   [WARN] script-based streaming unavailable ({type(first).__name__}); "
                "falling back to file-based streaming."
            )
    else:
        print("   [INFO] Current `datasets` version skips dataset scripts; using file-based streaming.")

    try:
        from huggingface_hub import list_repo_files
    except Exception as e:
        raise RuntimeError(
            "Need `huggingface_hub` for streaming fallback. "
            "Please install it in py311: `pip install -U huggingface_hub`."
        ) from e

    files = list_repo_files(DATASET_REPO_ID, repo_type="dataset")

    # Match the supported dataset layouts.
    candidates = []
    candidates.extend([p for p in files if p.startswith(f"{config_name}/")])
    candidates.extend([p for p in files if p.startswith(f"raw_review_{category}/")])
    candidates.extend([p for p in files if p.startswith(f"raw/review_categories/{category}.")])
    if len(candidates) == 0:
        candidates.extend([p for p in files if config_name in p])

    dedup = []
    seen = set()
    for p in sorted(candidates):
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    candidates = dedup

    if len(candidates) == 0:
        raise RuntimeError(
            f"Cannot find files for config `{config_name}` or category `{category}` in dataset `{DATASET_REPO_ID}`."
        )

    def _load_from_paths(builder_name: str, rel_paths: List[str]):
        hf_files = [f"hf://datasets/{DATASET_REPO_ID}/{p}" for p in rel_paths]
        try:
            return load_dataset(
                builder_name,
                data_files={"train": hf_files},
                split="train",
                streaming=True,
                download_config=download_config,
            )
        except Exception as e:
            msg = str(e)
            if "hf://datasets/" not in msg:
                raise
            https_files = [
                f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/{p}" for p in rel_paths
            ]
            print("   [WARN] `hf://` streaming failed; retrying with https URLs.")
            return load_dataset(
                builder_name,
                data_files={"train": https_files},
                split="train",
                streaming=True,
                download_config=download_config,
            )

    parquet_files = [p for p in candidates if p.lower().endswith(".parquet")]
    if len(parquet_files) > 0:
        print(f"   [INFO] Streaming from {len(parquet_files)} parquet shards")
        return _load_from_paths("parquet", parquet_files)

    json_ext = (".json", ".jsonl", ".json.gz", ".jsonl.gz", ".json.zst", ".jsonl.zst")
    json_files = [p for p in candidates if p.lower().endswith(json_ext)]
    if len(json_files) > 0:
        print(f"   [INFO] Streaming from {len(json_files)} json shards")
        return _load_from_paths("json", json_files)

    sample = ", ".join(candidates[:3])
    raise RuntimeError(
        "Found files for dataset config/category but none are parquet/json-compatible for streaming fallback. "
        f"Examples: {sample}"
    )

# Embeddings

def compute_text_embeddings_for_chunk(titles: List[str], texts: List[str],
                                     has_title: np.ndarray, has_text: np.ndarray
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    model = get_text_model()
    n = len(titles)
    title_embs = np.zeros((n, TITLE_EMB_DIM), dtype=np.float32)
    text_embs = np.zeros((n, TEXT_EMB_DIM), dtype=np.float32)

    if has_title.any():
        idxs = np.where(has_title)[0].tolist()
        batch_titles = [titles[i] for i in idxs]
        enc = model.encode(
            batch_titles,
            batch_size=TEXT_EMB_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        title_embs[idxs] = enc

    if has_text.any():
        idxs = np.where(has_text)[0].tolist()
        batch_texts = [texts[i] for i in idxs]
        enc = model.encode(
            batch_texts,
            batch_size=TEXT_EMB_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        text_embs[idxs] = enc

    return title_embs, text_embs

def compute_image_embeddings_for_chunk(
    image_urls: List[str],
    has_image: np.ndarray,
    image_cache: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    import torch
    from PIL import Image
    import requests
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import threading

    model, preprocess = get_clip_model()
    device = next(model.parameters()).device

    n = len(image_urls)
    out = np.zeros((n, IMAGE_EMB_DIM), dtype=np.float32)

    if not has_image.any():
        return out

    if image_cache is None:
        cache = {}
    else:
        cache = image_cache

    idxs = np.where(has_image)[0].tolist()
    urls = [image_urls[i] for i in idxs]

    to_download = []
    cache_hits = 0

    for idx, url in zip(idxs, urls):
        if url in cache:
            out[idx] = cache[url]
            cache_hits += 1
        else:
            to_download.append((idx, url))

    if cache_hits > 0:
        print(f"   Cache hits: {cache_hits}/{len(idxs)} ({cache_hits/len(idxs)*100:.1f}%)")

    if len(to_download) == 0:
        return out

    print(f"   Downloading {len(to_download):,} new images (32 workers)...")

    thread_local = threading.local()

    def get_session():
        sess = getattr(thread_local, "session", None)
        if sess is None:
            sess = requests.Session()
            retry = Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
            adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            thread_local.session = sess
        return sess

    def download_one(idx: int, url: str):
        if not url or not isinstance(url, str) or not url.strip():
            return idx, url, None
        sess = get_session()
        try:
            r = sess.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0", "Accept": "image/*"})
            r.raise_for_status()
            ctype = (r.headers.get("content-type", "") or "").lower()
            if "image" not in ctype:
                return idx, url, None
            img = Image.open(BytesIO(r.content)).convert("RGB")
            w, h = img.size
            if w < 50 or h < 50 or w > 10000 or h > 10000:
                return idx, url, None
            return idx, url, preprocess(img)
        except Exception:
            return idx, url, None

    failed = 0
    downloaded = 0

    def encode_batch(batch_data):
        if not batch_data:
            return
        batch_tensors = [x[2] for x in batch_data]
        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(x).float().cpu().numpy()
        for i, (idx, url, _) in enumerate(batch_data):
            emb = feats[i]
            out[idx] = emb
            cache[url] = emb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    encode_bs = 128 if device.type == "cuda" else 32
    cache_max = max(encode_bs * 4, 256)
    batch_data = []

    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(download_one, idx, url) for idx, url in to_download]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading images", leave=False):
            idx, url, tensor = fut.result()
            if tensor is None:
                failed += 1
                continue
            downloaded += 1
            batch_data.append((idx, url, tensor))
            if len(batch_data) >= cache_max:
                for s in range(0, len(batch_data), encode_bs):
                    encode_batch(batch_data[s:s+encode_bs])
                batch_data = []

    for s in range(0, len(batch_data), encode_bs):
        encode_batch(batch_data[s:s+encode_bs])

    print(f"   Downloaded {downloaded:,}/{len(to_download):,} images (failed {failed:,})")
    return out

# PASS 1: Stream -> chunk parquet + counts (with MIN_YEAR filter)

def pass1_stream_dump_and_count() -> Tuple[int, int]:
    print("\n" + "=" * 80)
    print(f"PASS 1: STREAM -> CHUNKS + COUNTS (filtering year >= {MIN_YEAR})")
    print("=" * 80)

    ds = _load_streaming_reviews_dataset(CATEGORY)

    clean_glob("pass1_chunk_*.parquet", TEMP_DIR)
    user_counts = Counter()
    item_counts = Counter()

    batch = []
    shard = 0
    total_kept = 0
    total_seen = 0
    total_filtered_by_year = 0

    for ex in tqdm(ds, desc="Streaming"):
        total_seen += 1
        rec = extract_review_record(ex)
        if rec is None:
            continue
        batch.append(rec)

        if len(batch) >= CHUNK_SIZE:
            df = pd.DataFrame(batch)
            len_before = len(df)
            df = process_timestamps(df)
            total_filtered_by_year += (len_before - len(df))

            if len(df) > 0:
                user_counts.update(df["user_id"].value_counts().to_dict())
                item_counts.update(df["item_id"].value_counts().to_dict())
                out_path = os.path.join(TEMP_DIR, f"pass1_chunk_{shard:04d}.parquet")
                df.to_parquet(out_path, index=False)
                total_kept += len(df)
                shard += 1

            batch = []
            gc.collect()

    if batch:
        df = pd.DataFrame(batch)
        len_before = len(df)
        df = process_timestamps(df)
        total_filtered_by_year += (len_before - len(df))

        if len(df) > 0:
            user_counts.update(df["user_id"].value_counts().to_dict())
            item_counts.update(df["item_id"].value_counts().to_dict())
            out_path = os.path.join(TEMP_DIR, f"pass1_chunk_{shard:04d}.parquet")
            df.to_parquet(out_path, index=False)
            total_kept += len(df)
            shard += 1

    with open(os.path.join(TEMP_DIR, "user_counts.pkl"), "wb") as f:
        pickle.dump(dict(user_counts), f)
    with open(os.path.join(TEMP_DIR, "item_counts.pkl"), "wb") as f:
        pickle.dump(dict(item_counts), f)

    print(f"\nPass 1 complete")
    print(f"   Seen: {total_seen:,}")
    print(f"   Filtered by year < {MIN_YEAR}: {total_filtered_by_year:,}")
    print(f"   Kept: {total_kept:,}")
    print(f"   Chunks: {shard}")
    print(f"   Users: {len(user_counts):,}  Items: {len(item_counts):,}")

    return total_kept, shard

# PASS 2: k-core filter + mapping

def _scan_counts(chunk_files: List[str], valid_users: Optional[set], valid_items: Optional[set]) -> Tuple[Counter, Counter]:
    ucnt = Counter()
    icnt = Counter()
    for fpath in chunk_files:
        df = pd.read_parquet(fpath, columns=["user_id", "item_id"])
        if valid_users is not None:
            df = df[df["user_id"].isin(valid_users)]
        if valid_items is not None:
            df = df[df["item_id"].isin(valid_items)]
        if len(df) == 0:
            continue
        ucnt.update(df["user_id"].value_counts().to_dict())
        icnt.update(df["item_id"].value_counts().to_dict())
        del df
    return ucnt, icnt

def pass2_kcore_and_write_filtered() -> Tuple[Dict, Dict, int]:
    print("\n" + "=" * 80)
    print(f"PASS 2: K-CORE FILTER ({MIN_USER_REVIEWS}-core) + MAPPING")
    print("=" * 80)

    with open(os.path.join(TEMP_DIR, "user_counts.pkl"), "rb") as f:
        user_counts = pickle.load(f)
    with open(os.path.join(TEMP_DIR, "item_counts.pkl"), "rb") as f:
        item_counts = pickle.load(f)

    valid_users = {u for u, c in user_counts.items() if c >= MIN_USER_REVIEWS}
    valid_items = {i for i, c in item_counts.items() if c >= MIN_ITEM_REVIEWS}

    chunk_files = sorted(glob.glob(os.path.join(TEMP_DIR, "pass1_chunk_*.parquet")))
    print(f"   Initial: {len(valid_users):,} users, {len(valid_items):,} items")

    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        ucnt, icnt = _scan_counts(chunk_files, valid_users, valid_items)
        new_users = {u for u, c in ucnt.items() if c >= MIN_USER_REVIEWS}
        new_items = {i for i, c in icnt.items() if c >= MIN_ITEM_REVIEWS}

        n_users_removed = len(valid_users) - len(new_users)
        n_items_removed = len(valid_items) - len(new_items)

        print(f"   Iter {iteration}: {len(new_users):,} users, {len(new_items):,} items "
              f"(removed {n_users_removed} users, {n_items_removed} items)")

        if len(new_users) == 0 or len(new_items) == 0:
            raise ValueError("K-core filtering resulted in empty dataset!")

        if n_users_removed == 0 and n_items_removed == 0:
            min_u = min(ucnt.values()) if ucnt else 0
            min_i = min(icnt.values()) if icnt else 0
            if min_u >= MIN_USER_REVIEWS and min_i >= MIN_ITEM_REVIEWS:
                print(f"   Converged after {iteration} iteration(s)")
                valid_users, valid_items = new_users, new_items
                break

        valid_users, valid_items = new_users, new_items

    # Verify the converged core before writing mappings.
    final_ucnt, final_icnt = _scan_counts(chunk_files, valid_users, valid_items)
    min_u = min(final_ucnt.values())
    min_i = min(final_icnt.values())
    print(f"   Final: min_user_reviews={min_u}, min_item_reviews={min_i}")

    if min_u < MIN_USER_REVIEWS or min_i < MIN_ITEM_REVIEWS:
        raise ValueError("K-core constraint not satisfied!")

    # +1 mapping (PAD=0)
    user2idx = {u: idx for idx, u in enumerate(sorted(valid_users), start=1)}
    item2idx = {i: idx for idx, i in enumerate(sorted(valid_items), start=1)}

    mappings = {
        "pad_idx": PAD_IDX,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": {v: k for k, v in user2idx.items()},
        "idx2item": {v: k for k, v in item2idx.items()},
        "num_users": int(len(user2idx) + 1),
        "num_items": int(len(item2idx) + 1),
    }
    with open(os.path.join(OUTPUT_DIR, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)

    # Write filtered chunks
    clean_glob("pass2_chunk_*.parquet", TEMP_DIR)
    total_core = 0
    global_idx = 0

    for i, fpath in enumerate(tqdm(chunk_files, desc="Writing filtered chunks")):
        df = pd.read_parquet(fpath)
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)].copy()
        if len(df) == 0:
            continue

        df["user_idx"] = df["user_id"].map(user2idx).astype(np.int32)
        df["item_idx"] = df["item_id"].map(item2idx).astype(np.int32)

        n = len(df)
        df["global_idx"] = np.arange(global_idx, global_idx + n, dtype=np.int64)
        global_idx += n
        total_core += n

        out_path = os.path.join(TEMP_DIR, f"pass2_chunk_{i:04d}.parquet")
        df.to_parquet(out_path, index=False)
        del df
        gc.collect()

    print(f"\nPass 2 complete: {total_core:,} interactions")
    print(f"   Users: {len(user2idx):,}  Items: {len(item2idx):,}")

    return user2idx, item2idx, total_core

# PASS 3a: Build histories

def pass3a_build_histories() -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    print("\n" + "=" * 80)
    print("PASS 3a: BUILD HISTORIES (time-sorted)")
    print("=" * 80)

    chunk_files = sorted(glob.glob(os.path.join(TEMP_DIR, "pass2_chunk_*.parquet")))

    from array import array
    user_g = defaultdict(lambda: array("q"))
    user_t = defaultdict(lambda: array("q"))
    item_g = defaultdict(lambda: array("q"))
    item_t = defaultdict(lambda: array("q"))

    cols = ["user_idx", "item_idx", "global_idx", "unix_time"]
    for fpath in tqdm(chunk_files, desc="Collect pairs"):
        df = pd.read_parquet(fpath, columns=cols)
        u = df["user_idx"].to_numpy(np.int32)
        it = df["item_idx"].to_numpy(np.int32)
        g = df["global_idx"].to_numpy(np.int64)
        t = df["unix_time"].to_numpy(np.int64)

        order = np.argsort(u, kind="mergesort")
        u_s, g_s, t_s = u[order], g[order], t[order]
        cuts = np.flatnonzero(np.diff(u_s)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(u_s)]))
        for s, e in zip(starts, ends):
            uid = int(u_s[s])
            user_g[uid].fromlist(g_s[s:e].astype(np.int64).tolist())
            user_t[uid].fromlist(t_s[s:e].astype(np.int64).tolist())

        order = np.argsort(it, kind="mergesort")
        it_s, g_s2, t_s2 = it[order], g[order], t[order]
        cuts = np.flatnonzero(np.diff(it_s)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(it_s)]))
        for s, e in zip(starts, ends):
            iid = int(it_s[s])
            item_g[iid].fromlist(g_s2[s:e].astype(np.int64).tolist())
            item_t[iid].fromlist(t_s2[s:e].astype(np.int64).tolist())

        del df
        gc.collect()

    user_histories = {}
    for uid in tqdm(list(user_g.keys()), desc="Sort user histories"):
        g_arr = np.frombuffer(user_g[uid], dtype=np.int64)
        t_arr = np.frombuffer(user_t[uid], dtype=np.int64)
        order = np.lexsort((g_arr, t_arr))
        user_histories[int(uid)] = g_arr[order].astype(np.int64).tolist()

    item_histories = {}
    for iid in tqdm(list(item_g.keys()), desc="Sort item histories"):
        g_arr = np.frombuffer(item_g[iid], dtype=np.int64)
        t_arr = np.frombuffer(item_t[iid], dtype=np.int64)
        order = np.lexsort((g_arr, t_arr))
        item_histories[int(iid)] = g_arr[order].astype(np.int64).tolist()

    with open(os.path.join(OUTPUT_DIR, "user_histories.pkl"), "wb") as f:
        pickle.dump(user_histories, f)
    with open(os.path.join(OUTPUT_DIR, "item_histories.pkl"), "wb") as f:
        pickle.dump(item_histories, f)

    print(f"\nHistories built")
    print(f"   Users: {len(user_histories):,}  Items: {len(item_histories):,}")

    return user_histories, item_histories

# PASS 3b: Split

def pass3b_split(user_histories: Dict[int, List[int]]) -> Tuple[List[int], List[int], List[int]]:
    print("\n" + "=" * 80)
    print("PASS 3b: TEMPORAL SPLIT (per-user last2)")
    print("=" * 80)

    train, val, test = [], [], []
    for uid, seq in tqdm(user_histories.items(), desc="Split users"):
        n = len(seq)
        if n < 3:
            train.extend(seq)
        else:
            train.extend(seq[:-2])
            val.append(seq[-2])
            test.append(seq[-1])

    split = {"train": train, "val": val, "test": test}
    with open(os.path.join(OUTPUT_DIR, "split_indices.pkl"), "wb") as f:
        pickle.dump(split, f)

    print(f"   Train={len(train):,}  Val={len(val):,}  Test={len(test):,}")
    return train, val, test

# PASS 3c: Train stats

def pass3c_train_global_stats(train_indices: List[int], total_records: int) -> Dict:
    print("\n" + "=" * 80)
    print("PASS 3c: TRAIN-ONLY GLOBAL STATS")
    print("=" * 80)

    train_mask = np.zeros(total_records, dtype=np.bool_)
    train_mask[np.array(train_indices, dtype=np.int64)] = True

    chunk_files = sorted(glob.glob(os.path.join(TEMP_DIR, "pass2_chunk_*.parquet")))
    sum_rating = 0.0
    sum_has_image = 0.0
    sum_title_len = 0.0
    sum_text_len = 0.0
    cnt = 0
    cnt_title = 0
    cnt_text = 0

    cols = ["global_idx", "rating", "has_image", "len_title_tokens", "len_text_tokens"]
    for fpath in tqdm(chunk_files, desc="Scan train stats"):
        df = pd.read_parquet(fpath, columns=cols)
        idxs = df["global_idx"].to_numpy(np.int64)
        m = train_mask[idxs]
        if not m.any():
            continue
        d = df[m]
        cnt += len(d)
        sum_rating += float(d["rating"].sum())
        sum_has_image += float(d["has_image"].sum())

        mt = d["len_title_tokens"] > 0
        if mt.any():
            sum_title_len += float(d.loc[mt, "len_title_tokens"].sum())
            cnt_title += int(mt.sum())

        mx = d["len_text_tokens"] > 0
        if mx.any():
            sum_text_len += float(d.loc[mx, "len_text_tokens"].sum())
            cnt_text += int(mx.sum())

        del df, d
        gc.collect()

    stats = {
        "mean_rating": float(sum_rating / cnt) if cnt > 0 else 3.0,
        "image_rate": float(sum_has_image / cnt) if cnt > 0 else 0.0,
        "mean_title_len": float(sum_title_len / cnt_title) if cnt_title > 0 else 5.0,
        "mean_text_len": float(sum_text_len / cnt_text) if cnt_text > 0 else 50.0,
    }

    with open(os.path.join(OUTPUT_DIR, "train_global_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    print(f"   mean_rating={stats['mean_rating']:.3f}  image_rate={stats['image_rate']*100:.2f}%")
    return stats

# PASS 4: Running stats + final chunks + embeddings + time_bin column

def pass4_running_stats_and_write(
    user_histories: Dict[int, List[int]],
    item_histories: Dict[int, List[int]],
    train_stats: Dict,
    total_records: int,
) -> Tuple[int, List[Dict]]:
    print("\n" + "=" * 80)
    print("PASS 4: RUNNING STATS + FINAL CHUNKS + EMBEDDINGS + TIME_BIN")
    print("=" * 80)

    chunk_files = sorted(glob.glob(os.path.join(TEMP_DIR, "pass2_chunk_*.parquet")))

    # Build lookup arrays
    ratings = np.zeros(total_records, dtype=np.float32)
    has_img = np.zeros(total_records, dtype=np.int8)
    has_title = np.zeros(total_records, dtype=np.int8)
    has_text = np.zeros(total_records, dtype=np.int8)
    len_title = np.zeros(total_records, dtype=np.int32)
    len_text = np.zeros(total_records, dtype=np.int32)
    ts = np.zeros(total_records, dtype=np.int64)

    for fpath in tqdm(chunk_files, desc="Build base arrays"):
        df = pd.read_parquet(
            fpath,
            columns=["global_idx", "rating", "has_image", "has_title", "has_text",
                     "len_title_tokens", "len_text_tokens", "unix_time"],
        )
        idxs = df["global_idx"].to_numpy(np.int64)
        ratings[idxs] = df["rating"].to_numpy(np.float32)
        has_img[idxs] = df["has_image"].to_numpy(np.int8)
        has_title[idxs] = df["has_title"].to_numpy(np.int8)
        has_text[idxs] = df["has_text"].to_numpy(np.int8)
        len_title[idxs] = df["len_title_tokens"].to_numpy(np.int32)
        len_text[idxs] = df["len_text_tokens"].to_numpy(np.int32)
        ts[idxs] = df["unix_time"].to_numpy(np.int64)
        del df
        gc.collect()

    # Allocate running-stat arrays
    u_num_prev = np.zeros(total_records, dtype=np.int32)
    u_prev_mean_rating = np.full(total_records, train_stats["mean_rating"], dtype=np.float32)
    u_prev_img_rate = np.full(total_records, train_stats["image_rate"], dtype=np.float32)
    u_prev_mean_title = np.full(total_records, train_stats["mean_title_len"], dtype=np.float32)
    u_prev_mean_text = np.full(total_records, train_stats["mean_text_len"], dtype=np.float32)
    u_is_first = np.zeros(total_records, dtype=np.int8)
    u_is_img_unusual = np.zeros(total_records, dtype=np.int8)
    u_is_text_unusual = np.zeros(total_records, dtype=np.int8)
    u_is_title_unusual = np.zeros(total_records, dtype=np.int8)
    u_rating_delta = np.zeros(total_records, dtype=np.float32)
    u_time_since_prev = np.zeros(total_records, dtype=np.float32)

    i_num_prev = np.zeros(total_records, dtype=np.int32)
    i_prev_mean_rating = np.full(total_records, train_stats["mean_rating"], dtype=np.float32)
    i_prev_std_rating = np.zeros(total_records, dtype=np.float32)
    i_prev_img_rate = np.full(total_records, train_stats["image_rate"], dtype=np.float32)
    i_prev_mean_text = np.full(total_records, train_stats["mean_text_len"], dtype=np.float32)
    i_is_first = np.zeros(total_records, dtype=np.int8)
    i_rating_delta = np.zeros(total_records, dtype=np.float32)
    i_time_since_prev = np.zeros(total_records, dtype=np.float32)

    # User running stats
    for uid, seq in tqdm(user_histories.items(), desc="User running stats"):
        idxs = np.array(seq, dtype=np.int64)
        n = len(idxs)
        if n == 0:
            continue

        pos = np.arange(n, dtype=np.int32)
        u_num_prev[idxs] = pos
        u_is_first[idxs[0]] = 1

        r = ratings[idxs]
        img = has_img[idxs].astype(np.float32)
        lt = len_title[idxs].astype(np.float32)
        lx = len_text[idxs].astype(np.float32)
        t = ts[idxs].astype(np.int64)

        csum_r = np.cumsum(r)
        prev_sum_r = np.concatenate(([0.0], csum_r[:-1]))
        prev_cnt = np.maximum(pos, 1).astype(np.float32)
        prev_mean_r = prev_sum_r / prev_cnt
        m = pos > 0
        u_prev_mean_rating[idxs[m]] = prev_mean_r[m]
        u_rating_delta[idxs[m]] = r[m] - prev_mean_r[m]

        csum_img = np.cumsum(img)
        prev_sum_img = np.concatenate(([0.0], csum_img[:-1]))
        prev_img_rate = prev_sum_img / prev_cnt
        u_prev_img_rate[idxs[m]] = prev_img_rate[m]
        u_is_img_unusual[idxs[m]] = ((img[m] > 0.5) & (prev_img_rate[m] < IMAGE_UNUSUAL_THRESHOLD)).astype(np.int8)

        title_present = (lt > 0).astype(np.float32)
        text_present = (lx > 0).astype(np.float32)

        csum_lt = np.cumsum(lt * title_present)
        csum_ct = np.cumsum(title_present)
        prev_sum_lt = np.concatenate(([0.0], csum_lt[:-1]))
        prev_cnt_lt = np.concatenate(([0.0], csum_ct[:-1]))
        prev_mean_lt = np.where(prev_cnt_lt > 0, prev_sum_lt / prev_cnt_lt, train_stats["mean_title_len"])
        u_prev_mean_title[idxs[m]] = prev_mean_lt[m]
        u_is_title_unusual[idxs[m]] = ((prev_cnt_lt[m] > 0) & (lt[m] > prev_mean_lt[m] * TEXT_LENGTH_UNUSUAL_ALPHA)).astype(np.int8)

        csum_lx = np.cumsum(lx * text_present)
        csum_cx = np.cumsum(text_present)
        prev_sum_lx = np.concatenate(([0.0], csum_lx[:-1]))
        prev_cnt_lx = np.concatenate(([0.0], csum_cx[:-1]))
        prev_mean_lx = np.where(prev_cnt_lx > 0, prev_sum_lx / prev_cnt_lx, train_stats["mean_text_len"])
        u_prev_mean_text[idxs[m]] = prev_mean_lx[m]
        u_is_text_unusual[idxs[m]] = ((prev_cnt_lx[m] > 0) & (lx[m] > prev_mean_lx[m] * TEXT_LENGTH_UNUSUAL_ALPHA)).astype(np.int8)

        dt = np.diff(t, prepend=t[0])
        dt = np.maximum(dt, 0)
        dt_days = dt.astype(np.float32) / 86400.0
        u_time_since_prev[idxs] = np.log1p(dt_days).astype(np.float32)
        u_time_since_prev[idxs[0]] = 0.0

    # Item running stats
    for iid, seq in tqdm(item_histories.items(), desc="Item running stats"):
        idxs = np.array(seq, dtype=np.int64)
        n = len(idxs)
        if n == 0:
            continue

        pos = np.arange(n, dtype=np.int32)
        i_num_prev[idxs] = pos
        i_is_first[idxs[0]] = 1

        r = ratings[idxs]
        img = has_img[idxs].astype(np.float32)
        lx = len_text[idxs].astype(np.float32)
        t = ts[idxs].astype(np.int64)

        csum_r = np.cumsum(r)
        csum_r2 = np.cumsum(r * r)
        prev_sum_r = np.concatenate(([0.0], csum_r[:-1]))
        prev_sum_r2 = np.concatenate(([0.0], csum_r2[:-1]))
        prev_cnt = np.maximum(pos, 1).astype(np.float32)
        prev_mean = prev_sum_r / prev_cnt
        var = np.maximum(prev_sum_r2 / prev_cnt - prev_mean * prev_mean, 0.0)
        prev_std = np.sqrt(var)

        m = pos > 0
        i_prev_mean_rating[idxs[m]] = prev_mean[m]
        i_prev_std_rating[idxs[m]] = prev_std[m]
        i_rating_delta[idxs[m]] = r[m] - prev_mean[m]

        csum_img = np.cumsum(img)
        prev_sum_img = np.concatenate(([0.0], csum_img[:-1]))
        prev_img_rate = prev_sum_img / prev_cnt
        i_prev_img_rate[idxs[m]] = prev_img_rate[m]

        text_present = (lx > 0).astype(np.float32)
        csum_lx = np.cumsum(lx * text_present)
        csum_cx = np.cumsum(text_present)
        prev_sum_lx = np.concatenate(([0.0], csum_lx[:-1]))
        prev_cnt_lx = np.concatenate(([0.0], csum_cx[:-1]))
        prev_mean_lx = np.where(prev_cnt_lx > 0, prev_sum_lx / prev_cnt_lx, train_stats["mean_text_len"])
        i_prev_mean_text[idxs[m]] = prev_mean_lx[m]

        dt = np.diff(t, prepend=t[0])
        dt = np.maximum(dt, 0)
        dt_days = dt.astype(np.float32) / 86400.0
        i_time_since_prev[idxs] = np.log1p(dt_days).astype(np.float32)
        i_time_since_prev[idxs[0]] = 0.0

    # Write final chunks + embeddings
    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
    clean_glob("chunk_*.parquet", os.path.join(OUTPUT_DIR, "data"))
    clean_glob("chunk_*_title_emb.npy", os.path.join(OUTPUT_DIR, "data"))
    clean_glob("chunk_*_text_emb.npy", os.path.join(OUTPUT_DIR, "data"))
    clean_glob("chunk_*_image_emb.npy", os.path.join(OUTPUT_DIR, "data"))

    if USE_TEXT_EMB:
        _ = get_text_model()
    if USE_IMAGE_EMB:
        _ = get_clip_model()

    # Load image cache
    image_cache = {}
    if USE_IMAGE_EMB and IMAGE_CACHE_ENABLED:
        if os.path.exists(IMAGE_CACHE_PATH):
            try:
                print(f"   Loading image embedding cache from {IMAGE_CACHE_PATH}...")
                with open(IMAGE_CACHE_PATH, 'rb') as f:
                    image_cache = pickle.load(f)
                print(f"   Loaded {len(image_cache):,} cached image embeddings")
            except Exception as e:
                print(f"   Cache load failed: {e}")
                image_cache = {}

    CACHE_SAVE_INTERVAL = 10
    chunk_info = []
    total_written = 0

    for cid, fpath in enumerate(tqdm(chunk_files, desc="Write final chunks")):
        df = pd.read_parquet(fpath)
        idxs = df["global_idx"].to_numpy(np.int64)

        # Add running stats columns
        df["u_num_prev_reviews"] = u_num_prev[idxs]
        df["u_prev_mean_rating"] = u_prev_mean_rating[idxs]
        df["u_prev_image_rate"] = u_prev_img_rate[idxs]
        df["u_prev_mean_title_len"] = u_prev_mean_title[idxs]
        df["u_prev_mean_text_len"] = u_prev_mean_text[idxs]
        df["u_is_first_review"] = u_is_first[idxs]
        df["u_is_image_unusual"] = u_is_img_unusual[idxs]
        df["u_is_text_long_unusual"] = u_is_text_unusual[idxs]
        df["u_is_title_long_unusual"] = u_is_title_unusual[idxs]
        df["u_rating_delta"] = u_rating_delta[idxs]
        df["u_time_since_prev"] = u_time_since_prev[idxs]

        df["i_num_prev_reviews"] = i_num_prev[idxs]
        df["i_prev_mean_rating"] = i_prev_mean_rating[idxs]
        df["i_prev_rating_std"] = i_prev_std_rating[idxs]
        df["i_prev_image_rate"] = i_prev_img_rate[idxs]
        df["i_prev_mean_text_len"] = i_prev_mean_text[idxs]
        df["i_is_first_review"] = i_is_first[idxs]
        df["i_rating_delta"] = i_rating_delta[idxs]
        df["i_time_since_prev"] = i_time_since_prev[idxs]

        # Add time_bin column (vectorized).
        ym = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
        df["time_bin"] = ym
        df.loc[ym > LAST_REGULAR_MONTH, "time_bin"] = "2023-Q2+"

        out_parquet = os.path.join(OUTPUT_DIR, "data", f"chunk_{cid:04d}.parquet")
        out_title = os.path.join(OUTPUT_DIR, "data", f"chunk_{cid:04d}_title_emb.npy")
        out_text = os.path.join(OUTPUT_DIR, "data", f"chunk_{cid:04d}_text_emb.npy")
        out_img = os.path.join(OUTPUT_DIR, "data", f"chunk_{cid:04d}_image_emb.npy")

        n = len(df)
        if USE_TEXT_EMB:
            titles = df["title"].fillna("").astype(str).tolist()
            texts = df["text"].fillna("").astype(str).tolist()
            ht = df["has_title"].to_numpy(np.int8).astype(bool)
            hx = df["has_text"].to_numpy(np.int8).astype(bool)
            title_embs, text_embs = compute_text_embeddings_for_chunk(titles, texts, ht, hx)
            atomic_save_npy(out_title, title_embs.astype(EMB_DTYPE))
            atomic_save_npy(out_text, text_embs.astype(EMB_DTYPE))

        if USE_IMAGE_EMB:
            hi = df["has_image"].to_numpy(np.int8).astype(bool)
            urls = df["image_url"].fillna("").astype(str).tolist()
            img_embs = compute_image_embeddings_for_chunk(urls, hi, image_cache=image_cache)
            atomic_save_npy(out_img, img_embs.astype(EMB_DTYPE))

            if IMAGE_CACHE_ENABLED and (cid + 1) % CACHE_SAVE_INTERVAL == 0:
                try:
                    with open(IMAGE_CACHE_PATH, 'wb') as f:
                        pickle.dump(image_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass

        # Drop raw text columns
        if not KEEP_RAW_TEXT:
            for c in ["title", "text"]:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)
        if not KEEP_IMAGE_URL and "image_url" in df.columns:
            df.drop(columns=["image_url"], inplace=True)
        if "datetime" in df.columns:
            if KEEP_DATETIME_STR:
                df["datetime_str"] = df["datetime"].astype(str)
            df.drop(columns=["datetime"], inplace=True)

        df.to_parquet(out_parquet, index=False)

        chunk_info.append({
            "chunk_id": cid,
            "file": f"data/chunk_{cid:04d}.parquet",
            "num_rows": int(n),
            "global_idx_min": int(idxs.min()) if n else 0,
            "global_idx_max": int(idxs.max()) if n else -1,
            "has_title_emb": bool(USE_TEXT_EMB),
            "has_text_emb": bool(USE_TEXT_EMB),
            "has_image_emb": bool(USE_IMAGE_EMB),
        })
        total_written += n
        del df
        gc.collect()

    with open(os.path.join(OUTPUT_DIR, "chunk_info.json"), "w") as f:
        json.dump(chunk_info, f, indent=2)

    # Final save image cache
    if USE_IMAGE_EMB and IMAGE_CACHE_ENABLED and len(image_cache) > 0:
        try:
            with open(IMAGE_CACHE_PATH, 'wb') as f:
                pickle.dump(image_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    print(f"\nPass 4 complete: wrote {total_written:,} rows in {len(chunk_info)} chunks")
    return total_written, chunk_info

# PASS 5: Final stats

def pass5_final_stats(user2idx: Dict, item2idx: Dict, total_records: int, chunk_info: List[Dict]) -> Dict:
    print("\n" + "=" * 80)
    print("PASS 5: FINAL STATS")
    print("=" * 80)

    with open(os.path.join(OUTPUT_DIR, "user_histories.pkl"), "rb") as f:
        user_histories = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "item_histories.pkl"), "rb") as f:
        item_histories = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "split_indices.pkl"), "rb") as f:
        split_indices = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "train_global_stats.pkl"), "rb") as f:
        train_stats = pickle.load(f)

    sum_rating = 0.0
    sum_rating_sq = 0.0
    sum_has_title = 0
    sum_has_text = 0
    sum_has_image = 0

    for meta in tqdm(chunk_info, desc="Scan chunks"):
        df = pd.read_parquet(
            os.path.join(OUTPUT_DIR, meta["file"]),
            columns=["rating", "has_title", "has_text", "has_image"],
        )
        r = df["rating"].to_numpy(np.float64)
        sum_rating += float(r.sum())
        sum_rating_sq += float((r * r).sum())
        sum_has_title += int(df["has_title"].sum())
        sum_has_text += int(df["has_text"].sum())
        sum_has_image += int(df["has_image"].sum())
        del df
        gc.collect()

    user_lens = [len(v) for v in user_histories.values()]
    item_lens = [len(v) for v in item_histories.values()]

    stats = {
        "category": CATEGORY,
        "preprocessing_date": datetime.now().isoformat(),
        "version": "did_ready",
        "min_year": MIN_YEAR,
        "total_reviews": int(total_records),
        "num_users": int(len(user2idx)),
        "num_items": int(len(item2idx)),
        "num_chunks": int(len(chunk_info)),
        "sparsity": float(1 - total_records / (max(len(user2idx),1) * max(len(item2idx),1))),
        "user_seq_length": {
            "mean": float(np.mean(user_lens)),
            "median": float(np.median(user_lens)),
            "min": int(np.min(user_lens)),
            "max": int(np.max(user_lens)),
        },
        "item_seq_length": {
            "mean": float(np.mean(item_lens)),
            "median": float(np.median(item_lens)),
            "min": int(np.min(item_lens)),
            "max": int(np.max(item_lens)),
        },
        "rating": {
            "mean": float(sum_rating / total_records) if total_records else 0.0,
            "std": float(np.sqrt(max(sum_rating_sq / total_records - (sum_rating / total_records) ** 2, 0.0))) if total_records else 0.0,
        },
        "modality": {
            "has_title_rate": float(sum_has_title / total_records) if total_records else 0.0,
            "has_text_rate": float(sum_has_text / total_records) if total_records else 0.0,
            "has_image_rate": float(sum_has_image / total_records) if total_records else 0.0,
        },
        "split": {
            "train": int(len(split_indices["train"])),
            "val": int(len(split_indices["val"])),
            "test": int(len(split_indices["test"])),
        },
        "config": {
            "min_user_reviews": MIN_USER_REVIEWS,
            "min_item_reviews": MIN_ITEM_REVIEWS,
            "chunk_size": CHUNK_SIZE,
            "pad_idx": PAD_IDX,
        },
        "embeddings": {
            "use_text_emb": bool(USE_TEXT_EMB),
            "use_image_emb": bool(USE_IMAGE_EMB),
            "title_emb_dim": int(TITLE_EMB_DIM if USE_TEXT_EMB else 0),
            "text_emb_dim": int(TEXT_EMB_DIM if USE_TEXT_EMB else 0),
            "image_emb_dim": int(IMAGE_EMB_DIM if USE_IMAGE_EMB else 0),
            "emb_dtype": str(EMB_DTYPE),
        },
        "train_global_stats": train_stats,
    }

    with open(os.path.join(OUTPUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    return stats

# PASS 5b: Build item reputation features table

def pass5b_build_item_rep_table(
    item_histories: Dict[int, List[int]],
    train_indices: List[int],
    total_records: int,
    num_items: int,
    train_stats: Dict,
    chunk_info: List[Dict],
) -> np.ndarray:
    print("\n" + "=" * 80)
    print("PASS 5b: BUILD ITEM REPUTATION FEATURES TABLE")
    print("=" * 80)

    cache_path = os.path.join(OUTPUT_DIR, "item_rep_features_train_last.npy")

    train_mask = np.zeros(total_records, dtype=np.bool_)
    train_mask[np.array(train_indices, dtype=np.int64)] = True

    print("   Selecting last train sample per item...")
    chosen = np.full((num_items,), -1, dtype=np.int64)

    for iid, seq in tqdm(item_histories.items(), desc="Select last train"):
        if iid <= 0 or iid >= num_items:
            continue
        if not seq:
            continue

        g_sel = -1
        for g in reversed(seq):
            if 0 <= g < total_records and train_mask[g]:
                g_sel = int(g)
                break

        # Keep -1 when no train event exists to avoid using val/test events.
        if g_sel >= 0:
            chosen[iid] = g_sel


    rep_cols = [
        "i_num_prev_reviews", "i_prev_mean_rating", "i_prev_rating_std",
        "i_prev_image_rate", "i_prev_mean_text_len", "i_is_first_review",
    ]

    print("   Building global_idx index...")
    g2chunk = np.full(total_records, -1, dtype=np.int32)
    g2row = np.full(total_records, -1, dtype=np.int32)

    for meta in tqdm(chunk_info, desc="Index global_idx"):
        cid = int(meta["chunk_id"])
        path = os.path.join(OUTPUT_DIR, meta["file"])
        dfg = pd.read_parquet(path, columns=["global_idx"])
        idxs = dfg["global_idx"].to_numpy(np.int64)
        g2chunk[idxs] = cid
        g2row[idxs] = np.arange(len(idxs), dtype=np.int32)
        del dfg

    rep_table = np.zeros((num_items, 6), dtype=np.float32)
    mean_rating = float(train_stats.get("mean_rating", 4.0))
    default_image_rate = float(train_stats.get("image_rate", 0.05))
    default_mean_text = float(train_stats.get("mean_text_len", 50.0))

    chunk2items = defaultdict(list)
    for iid in range(1, num_items):
        g = int(chosen[iid])
        # Use only train events; otherwise keep default filling.
        if 0 <= g < total_records and train_mask[g]:
            cid = int(g2chunk[g])
            row = int(g2row[g])
            if cid >= 0 and row >= 0:
                chunk2items[cid].append((iid, row))


    chunk_id_to_path = {meta["chunk_id"]: os.path.join(OUTPUT_DIR, meta["file"]) for meta in chunk_info}

    for cid, items in tqdm(chunk2items.items(), desc="Fetch features"):
        if cid not in chunk_id_to_path:
            continue
        chunk_path = chunk_id_to_path[cid]
        df = pd.read_parquet(chunk_path, columns=rep_cols)
        for iid, row in items:
            if row < len(df):
                rep_table[iid, 0] = float(df.iloc[row]["i_num_prev_reviews"]) / 100.0
                rep_table[iid, 1] = float(df.iloc[row]["i_prev_mean_rating"]) / 5.0
                rep_table[iid, 2] = float(df.iloc[row]["i_prev_rating_std"]) / 2.0
                rep_table[iid, 3] = float(df.iloc[row]["i_prev_image_rate"])
                rep_table[iid, 4] = float(df.iloc[row]["i_prev_mean_text_len"]) / 100.0
                rep_table[iid, 5] = float(df.iloc[row]["i_is_first_review"])
        del df

    # Fill missing with defaults (but keep PAD=0 as zeros)
    zero_mask = rep_table.sum(axis=1) == 0
    zero_mask[0] = False
    rep_table[0] = 0.0

    if zero_mask.any():
        default_rep = np.array([
            0.0, mean_rating / 5.0, 0.0, default_image_rate, default_mean_text / 100.0, 0.0
        ], dtype=np.float32)
        rep_table[zero_mask] = default_rep

    np.nan_to_num(rep_table, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(cache_path, rep_table)

    print(f"   Saved item rep table: {cache_path}, shape={rep_table.shape}")
    return rep_table

# PASS 6a: Build time_bin mapping and config

def pass6a_build_time_bin_config(chunk_info: List[Dict]) -> Dict[str, int]:
    """Build time_bin -> bin_id mapping and save config."""
    print("\n" + "=" * 80)
    print("PASS 6a: BUILD TIME BIN CONFIG")
    print("=" * 80)

    # Build month sequence deterministically (without data scanning).
    months = pd.period_range(start=f"{MIN_YEAR}-01", end=LAST_REGULAR_MONTH, freq="M").astype(str).tolist()
    months.append("2023-Q2+")

    time_bin_to_id = {b: i for i, b in enumerate(months)}
    id_to_time_bin = {i: b for b, i in time_bin_to_id.items()}

    config = {
        "min_year": MIN_YEAR,
        "last_regular_month": LAST_REGULAR_MONTH,
        "num_bins": len(months),
        "bin_list": months,
        "time_bin_to_id": time_bin_to_id,
        "id_to_time_bin": id_to_time_bin,
        "shrinkage_lambda": SHRINKAGE_LAMBDA,
        "n_groups": N_GROUPS,
        "group_names": GROUP_NAMES,
        "group_thresholds": GROUP_THRESHOLDS,
        "lag_rule": "For bin b, use O^{b-1}. For bin 0, use zero vector.",
    }

    with open(os.path.join(OUTPUT_DIR, "time_bin_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"   Total bins: {len(months)}")
    print(f"   First bin: {months[0]}")
    print(f"   Last bin: {months[-1]}")

    return time_bin_to_id
# PASS 6b: Compute user groups (train-only)

def pass6b_compute_user_groups(
    user_histories: Dict[int, List[int]],
    train_indices: List[int],
    total_records: int,
    chunk_info: List[Dict],
) -> Dict[int, int]:
    """
    Compute user group assignment based on train-only image_rate.
    Group 0: never (image_rate = 0)
    Group 1: rare (0 < image_rate <= 0.2)
    Group 2: frequent (image_rate > 0.2)
    """
    print("\n" + "=" * 80)
    print("PASS 6b: COMPUTE USER GROUPS (train-only)")
    print("=" * 80)

    # Train-event lookup.
    train_mask = np.zeros(total_records, dtype=np.bool_)
    train_mask[np.array(train_indices, dtype=np.int64)] = True

    # Build global_idx -> has_image lookup
    print("   Loading has_image for all events...")
    g2has_image = np.zeros(total_records, dtype=np.int8)

    for meta in tqdm(chunk_info, desc="Load has_image"):
        df = pd.read_parquet(os.path.join(OUTPUT_DIR, meta["file"]), columns=["global_idx", "has_image"])
        idxs = df["global_idx"].to_numpy(np.int64)
        g2has_image[idxs] = df["has_image"].to_numpy(np.int8)
        del df

    # Compute train-only image_rate per user
    print("   Computing train-only image_rate per user...")
    user_groups: Dict[int, int] = {}

    for uid, seq in tqdm(user_histories.items(), desc="Compute user groups"):
        idxs = np.array(seq, dtype=np.int64)
        m = train_mask[idxs]
        train_events = idxs[m]

        if len(train_events) == 0:
            user_groups[uid] = 0
            continue

        has_images = g2has_image[train_events]
        image_rate = float(has_images.mean())

        if image_rate == 0:
            user_groups[uid] = 0
        elif image_rate <= 0.2:
            user_groups[uid] = 1
        else:
            user_groups[uid] = 2

    with open(os.path.join(OUTPUT_DIR, "user_groups_train_only.pkl"), "wb") as f:
        pickle.dump(user_groups, f)

    group_counts = Counter(user_groups.values())
    print(f"\n   Group distribution:")
    for g in range(N_GROUPS):
        cnt = group_counts.get(g, 0)
        pct = cnt / len(user_groups) * 100 if user_groups else 0
        print(f"     Group {g} ({GROUP_NAMES[g]}): {cnt:,} ({pct:.1f}%)")

    return user_groups

# PASS 6c: Compute global and group O^t statistics (train-only)

def pass6c_compute_time_stats(
    train_indices: List[int],
    total_records: int,
    chunk_info: List[Dict],
    time_bin_to_id: Dict[str, int],
    user_groups: Dict[int, int],
) -> Tuple[Dict, Dict]:
    """
    Compute O^t statistics for each time bin, using train events only.
    Also generates causally lagged versions for training.

    Returns:
        global_time_stats: {bin_id: O^t dict}
        group_time_stats: {(group_id, bin_id): O^t dict}
    """
    print("\n" + "=" * 80)
    print("PASS 6c: COMPUTE TIME STATS (train-only, vectorized)")
    print("=" * 80)

    # Load train-level defaults.
    try:
        with open(os.path.join(OUTPUT_DIR, "train_global_stats.pkl"), "rb") as f:
            train_stats = pickle.load(f)
    except Exception:
        train_stats = {"mean_rating": 4.0, "image_rate": 0.08, "mean_text_len": 50.0}

    default_image_rate = float(train_stats.get("image_rate", 0.08))
    default_text_len = float(train_stats.get("mean_text_len", 50.0))

    # Train mask
    train_mask = np.zeros(total_records, dtype=np.bool_)
    train_mask[np.array(train_indices, dtype=np.int64)] = True

    num_bins = len(time_bin_to_id)

    # Accumulators
    # Global
    g_cnt = np.zeros(num_bins, dtype=np.int64)
    g_sum_rating = np.zeros(num_bins, dtype=np.float64)
    g_sum_rating_sq = np.zeros(num_bins, dtype=np.float64)
    g_sum_has_image = np.zeros(num_bins, dtype=np.float64)
    g_sum_text_len = np.zeros(num_bins, dtype=np.float64)
    g_sum_text_len_sq = np.zeros(num_bins, dtype=np.float64)

    # Per-group
    group_cnt = np.zeros((N_GROUPS, num_bins), dtype=np.int64)
    group_sum_rating = np.zeros((N_GROUPS, num_bins), dtype=np.float64)
    group_sum_rating_sq = np.zeros((N_GROUPS, num_bins), dtype=np.float64)
    group_sum_has_image = np.zeros((N_GROUPS, num_bins), dtype=np.float64)
    group_sum_text_len = np.zeros((N_GROUPS, num_bins), dtype=np.float64)
    group_sum_text_len_sq = np.zeros((N_GROUPS, num_bins), dtype=np.float64)

    # user_idx -> group lookup array
    max_user_idx = max(user_groups.keys()) + 1
    user_group_arr = np.zeros(max_user_idx, dtype=np.int32)
    for uid, grp in user_groups.items():
        if 0 <= uid < max_user_idx:
            user_group_arr[uid] = int(grp)

    cols = ["global_idx", "user_idx", "bin_id", "rating", "has_image", "len_text_tokens"]

    print("   Scanning train events (vectorized)...")
    for meta in tqdm(chunk_info, desc="Accumulate stats"):
        path = os.path.join(OUTPUT_DIR, meta["file"])
        df = pd.read_parquet(path, columns=cols)

        idxs = df["global_idx"].to_numpy(np.int64)
        m = train_mask[idxs]
        if not m.any():
            del df
            continue

        bin_ids = df["bin_id"].to_numpy(np.int32)[m]
        user_idxs = df["user_idx"].to_numpy(np.int32)[m]
        ratings = df["rating"].to_numpy(np.float64)[m]
        has_images = df["has_image"].to_numpy(np.float64)[m]
        text_lens = df["len_text_tokens"].to_numpy(np.float64)[m]

        del df

        # Map user indices to group ids.
        user_idxs_clip = np.clip(user_idxs, 0, max_user_idx - 1)
        group_ids = user_group_arr[user_idxs_clip]

        # Global
        np.add.at(g_cnt, bin_ids, 1)
        np.add.at(g_sum_rating, bin_ids, ratings)
        np.add.at(g_sum_rating_sq, bin_ids, ratings ** 2)
        np.add.at(g_sum_has_image, bin_ids, has_images)
        np.add.at(g_sum_text_len, bin_ids, text_lens)
        np.add.at(g_sum_text_len_sq, bin_ids, text_lens ** 2)

        # Per-group (loop over small N_GROUPS)
        for grp in range(N_GROUPS):
            gm = (group_ids == grp)
            if not gm.any():
                continue
            b = bin_ids[gm]
            r = ratings[gm]
            hi = has_images[gm]
            tl = text_lens[gm]

            np.add.at(group_cnt[grp], b, 1)
            np.add.at(group_sum_rating[grp], b, r)
            np.add.at(group_sum_rating_sq[grp], b, r ** 2)
            np.add.at(group_sum_has_image[grp], b, hi)
            np.add.at(group_sum_text_len[grp], b, tl)
            np.add.at(group_sum_text_len_sq[grp], b, tl ** 2)

        gc.collect()

    print("   Building stats dictionaries...")

    def build_ot(n, sum_r, sum_r2, sum_img, sum_tlen, sum_tlen2):
        if n > 0:
            rating_mean = sum_r / n
            rating_var = max(sum_r2 / n - rating_mean ** 2, 0.0)
            rating_std = np.sqrt(rating_var)

            image_rate = sum_img / n

            text_len_mean = sum_tlen / n
            text_len_var = max(sum_tlen2 / n - text_len_mean ** 2, 0.0)
            text_len_std = np.sqrt(text_len_var)
        else:
            rating_mean, rating_std = 4.0, 1.0
            image_rate = default_image_rate
            text_len_mean, text_len_std = default_text_len, 30.0

        return {
            "rating_mean": float(rating_mean),
            "rating_std": float(rating_std),
            "image_rate": float(image_rate),
            "text_len_mean": float(text_len_mean),
            "text_len_std": float(text_len_std),
            "n_events": int(n),
        }

    global_time_stats: Dict[int, Dict] = {}
    for b in range(num_bins):
        global_time_stats[b] = build_ot(
            int(g_cnt[b]),
            float(g_sum_rating[b]),
            float(g_sum_rating_sq[b]),
            float(g_sum_has_image[b]),
            float(g_sum_text_len[b]),
            float(g_sum_text_len_sq[b]),
        )

    group_time_stats: Dict[Tuple[int, int], Dict] = {}
    for grp in range(N_GROUPS):
        for b in range(num_bins):
            group_time_stats[(grp, b)] = build_ot(
                int(group_cnt[grp, b]),
                float(group_sum_rating[grp, b]),
                float(group_sum_rating_sq[grp, b]),
                float(group_sum_has_image[grp, b]),
                float(group_sum_text_len[grp, b]),
                float(group_sum_text_len_sq[grp, b]),
            )

    # Lag and shrinkage
    print("   Generating LAG versions...")

    default_ot = {
        "rating_mean": 4.0,
        "rating_std": 1.0,
        "image_rate": default_image_rate,
        "text_len_mean": default_text_len,
        "text_len_std": 30.0,
        "n_events": 0,
    }

    global_time_stats_lag: Dict[int, Dict] = {}
    for b in range(num_bins):
        global_time_stats_lag[b] = default_ot.copy() if b == 0 else global_time_stats[b - 1].copy()

    group_time_stats_lag: Dict[Tuple[int, int], Dict] = {}
    for grp in range(N_GROUPS):
        for b in range(num_bins):
            if b == 0:
                group_time_stats_lag[(grp, b)] = default_ot.copy()
            else:
                raw = group_time_stats[(grp, b - 1)]
                glob = global_time_stats[b - 1]
                n = raw["n_events"]
                alpha = n / (n + SHRINKAGE_LAMBDA)

                shrunk = {
                    "rating_mean": alpha * raw["rating_mean"] + (1 - alpha) * glob["rating_mean"],
                    "rating_std": alpha * raw["rating_std"] + (1 - alpha) * glob["rating_std"],
                    "image_rate": alpha * raw["image_rate"] + (1 - alpha) * glob["image_rate"],
                    "text_len_mean": alpha * raw["text_len_mean"] + (1 - alpha) * glob["text_len_mean"],
                    "text_len_std": alpha * raw["text_len_std"] + (1 - alpha) * glob["text_len_std"],
                    "n_events": raw["n_events"],
                    "shrinkage_alpha": float(alpha),
                }
                group_time_stats_lag[(grp, b)] = shrunk

    # Save
    with open(os.path.join(OUTPUT_DIR, "global_time_stats.pkl"), "wb") as f:
        pickle.dump(global_time_stats, f)
    with open(os.path.join(OUTPUT_DIR, "group_time_stats.pkl"), "wb") as f:
        pickle.dump(group_time_stats, f)

    with open(os.path.join(OUTPUT_DIR, "global_time_stats_lag.pkl"), "wb") as f:
        pickle.dump(global_time_stats_lag, f)
    with open(os.path.join(OUTPUT_DIR, "group_time_stats_lag.pkl"), "wb") as f:
        pickle.dump(group_time_stats_lag, f)

    return global_time_stats, group_time_stats

# PASS 6d: Add bin_id column to parquet files

def pass6d_add_bin_id_column(chunk_info: List[Dict], time_bin_to_id: Dict[str, int]):
    """Add bin_id column to all parquet files for efficient training lookup."""
    print("\n" + "=" * 80)
    print("PASS 6d: ADD bin_id COLUMN TO PARQUET FILES")
    print("=" * 80)

    for meta in tqdm(chunk_info, desc="Add bin_id"):
        path = os.path.join(OUTPUT_DIR, meta["file"])
        df = pd.read_parquet(path)
        df["bin_id"] = df["time_bin"].map(time_bin_to_id).astype(np.int16)
        df.to_parquet(path, index=False)
        del df

    print("   bin_id column added to all chunks")


# PASS 6e: Build item-side train-only resources

def pass6e_build_item_side_train_resources(
    train_indices: List[int],
    total_records: int,
    chunk_info: List[Dict],
    num_items: int,
    num_bins: int,
    n_item_groups: int = 8,
    cooc_window: int = 3,
    neighbor_k: int = 20,
):
    print("\n" + "=" * 80)
    print("PASS 6e: BUILD ITEM-SIDE RESOURCES (train-only)")
    print("=" * 80)

    train_mask = np.zeros(total_records, dtype=np.bool_)
    train_mask[np.array(train_indices, dtype=np.int64)] = True

    item_hist_tmp: Dict[int, List[int]] = defaultdict(list)
    item_time_tmp: Dict[int, List[float]] = defaultdict(list)
    user_event_tmp: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    item_counts = np.zeros(num_items, dtype=np.int64)
    train_bin_set = set()

    cols = ["global_idx", "user_idx", "item_idx", "unix_time", "bin_id"]
    for meta in tqdm(chunk_info, desc="Collect train item/user events"):
        path = os.path.join(OUTPUT_DIR, meta["file"])
        df = pd.read_parquet(path, columns=cols)
        g = df["global_idx"].to_numpy(np.int64)
        m = train_mask[g]
        if not m.any():
            del df
            continue

        users = df["user_idx"].to_numpy(np.int64)[m]
        items = df["item_idx"].to_numpy(np.int64)[m]
        gids = g[m]
        ts = df["unix_time"].to_numpy(np.int64)[m]
        bins = df["bin_id"].to_numpy(np.int64)[m]
        valid = (
            (users > 0)
            & (items > 0)
            & (items < num_items)
            & (bins >= 0)
            & (bins < num_bins)
        )
        users = users[valid]
        items = items[valid]
        gids = gids[valid]
        ts = ts[valid]
        bins = bins[valid]
        if items.size == 0:
            del df
            continue

        item_counts += np.bincount(items, minlength=num_items)
        train_bin_set.update(np.unique(bins).tolist())

        for u, it, gg, tt in zip(users.tolist(), items.tolist(), gids.tolist(), ts.tolist()):
            item_hist_tmp[int(it)].append(int(gg))
            item_time_tmp[int(it)].append(float(tt))
            user_event_tmp[int(u)].append((int(tt), int(it)))

        del df
        gc.collect()

    item_histories = {}
    item_hist_times = {}
    for it, seq in tqdm(item_hist_tmp.items(), desc="Sort item histories (train-only)"):
        gids = np.asarray(seq, dtype=np.int64)
        ts = np.asarray(item_time_tmp[it], dtype=np.float64)
        order = np.argsort(ts, kind="mergesort")
        item_histories[int(it)] = gids[order].astype(np.int64).tolist()
        item_hist_times[int(it)] = ts[order].astype(np.float64).tolist()

    with open(os.path.join(OUTPUT_DIR, "item_histories.pkl"), "wb") as f:
        pickle.dump(item_histories, f)
    with open(os.path.join(OUTPUT_DIR, "item_hist_times.pkl"), "wb") as f:
        pickle.dump(item_hist_times, f)
    with open(os.path.join(OUTPUT_DIR, "item_histories_meta.json"), "w") as f:
        json.dump(
            {
                "source_split": "train_only",
                "num_items_with_history": int(len(item_histories)),
                "built_at": int(datetime.utcnow().timestamp()),
            },
            f,
            indent=2,
        )

    order = np.argsort(item_counts)
    item_group_arr = np.zeros(num_items, dtype=np.int64)
    binsz = max(1, len(order) // n_item_groups)
    for grp in range(n_item_groups):
        s = grp * binsz
        e = (grp + 1) * binsz if grp < n_item_groups - 1 else len(order)
        item_group_arr[order[s:e]] = grp
    item_group_arr[0] = 0
    with open(os.path.join(OUTPUT_DIR, "item_groups.pkl"), "wb") as f:
        pickle.dump(item_group_arr.astype(np.int64), f)

    cooc: Dict[int, Counter] = defaultdict(Counter)
    for _, ev in tqdm(user_event_tmp.items(), desc="Build item neighbors (train-only)"):
        if len(ev) < 2:
            continue
        ev.sort(key=lambda x: x[0])
        items = np.asarray([x[1] for x in ev], dtype=np.int64)
        items = items[(items > 0) & (items < num_items)]
        if items.size < 2:
            continue
        L = items.size
        for i in range(L):
            a = int(items[i])
            end = min(L, i + cooc_window + 1)
            for j in range(i + 1, end):
                b = int(items[j])
                if a == b:
                    continue
                cooc[a][b] += 1
                cooc[b][a] += 1

    item_neighbors = {}
    for it in range(num_items):
        ctr = cooc.get(it, None)
        if ctr is None or len(ctr) == 0:
            item_neighbors[it] = np.zeros(0, dtype=np.int64)
        else:
            top = ctr.most_common(neighbor_k)
            item_neighbors[it] = np.asarray([x[0] for x in top], dtype=np.int64)

    with open(os.path.join(OUTPUT_DIR, "item_neighbors.pkl"), "wb") as f:
        pickle.dump(item_neighbors, f)
    with open(os.path.join(OUTPUT_DIR, "item_neighbors_meta.json"), "w") as f:
        json.dump(
            {
                "source_split": "train_only",
                "window": int(cooc_window),
                "topk": int(neighbor_k),
                "built_at": int(datetime.utcnow().timestamp()),
            },
            f,
            indent=2,
        )

    B = num_bins
    G = n_item_groups
    g_count = np.zeros(B, dtype=np.float64)
    g_sum_r = np.zeros(B, dtype=np.float64)
    g_sum_r2 = np.zeros(B, dtype=np.float64)
    g_sum_img = np.zeros(B, dtype=np.float64)
    g_sum_t = np.zeros(B, dtype=np.float64)
    g_sum_t2 = np.zeros(B, dtype=np.float64)
    gb_count = np.zeros(G * B, dtype=np.float64)
    gb_sum_r = np.zeros(G * B, dtype=np.float64)
    gb_sum_r2 = np.zeros(G * B, dtype=np.float64)
    gb_sum_img = np.zeros(G * B, dtype=np.float64)
    gb_sum_t = np.zeros(G * B, dtype=np.float64)
    gb_sum_t2 = np.zeros(G * B, dtype=np.float64)
    per_key = defaultdict(lambda: np.zeros(4, dtype=np.float64))

    cols = ["global_idx", "item_idx", "bin_id", "rating", "has_image", "len_text_tokens"]
    for meta in tqdm(chunk_info, desc="Build temporal/future targets (train-only)"):
        path = os.path.join(OUTPUT_DIR, meta["file"])
        df = pd.read_parquet(path, columns=cols)
        g = df["global_idx"].to_numpy(np.int64)
        m = train_mask[g]
        if not m.any():
            del df
            continue

        item = df["item_idx"].to_numpy(np.int64)[m]
        b = df["bin_id"].to_numpy(np.int64)[m]
        r = df["rating"].to_numpy(np.float64)[m]
        img = df["has_image"].to_numpy(np.float64)[m]
        tl = df["len_text_tokens"].to_numpy(np.float64)[m]
        valid = (item > 0) & (item < num_items) & (b >= 0) & (b < B)
        item = item[valid]
        b = b[valid]
        r = r[valid]
        img = img[valid]
        tl = tl[valid]
        if item.size == 0:
            del df
            continue

        g_count += np.bincount(b, minlength=B)
        g_sum_r += np.bincount(b, weights=r, minlength=B)
        g_sum_r2 += np.bincount(b, weights=r * r, minlength=B)
        g_sum_img += np.bincount(b, weights=img, minlength=B)
        g_sum_t += np.bincount(b, weights=tl, minlength=B)
        g_sum_t2 += np.bincount(b, weights=tl * tl, minlength=B)

        grp = item_group_arr[item]
        gb = grp * B + b
        gb_count += np.bincount(gb, minlength=G * B)
        gb_sum_r += np.bincount(gb, weights=r, minlength=G * B)
        gb_sum_r2 += np.bincount(gb, weights=r * r, minlength=G * B)
        gb_sum_img += np.bincount(gb, weights=img, minlength=G * B)
        gb_sum_t += np.bincount(gb, weights=tl, minlength=G * B)
        gb_sum_t2 += np.bincount(gb, weights=tl * tl, minlength=G * B)

        for it, bb, rr, ii, tt in zip(item.tolist(), b.tolist(), r.tolist(), img.tolist(), tl.tolist()):
            key = (int(it), int(bb))
            per_key[key][0] += 1.0
            per_key[key][1] += float(rr)
            per_key[key][2] += float(ii)
            per_key[key][3] += float(tt)

        del df
        gc.collect()

    item_global_ot = np.zeros((B, 5), dtype=np.float32)
    cnt = np.maximum(g_count, 1.0)
    mean_r = g_sum_r / cnt
    var_r = np.maximum(g_sum_r2 / cnt - mean_r * mean_r, 0.0)
    mean_t = g_sum_t / cnt
    var_t = np.maximum(g_sum_t2 / cnt - mean_t * mean_t, 0.0)
    item_global_ot[:, 0] = (mean_r / 5.0).astype(np.float32)
    item_global_ot[:, 1] = (np.sqrt(var_r) / 2.0).astype(np.float32)
    item_global_ot[:, 2] = (g_sum_img / cnt).astype(np.float32)
    item_global_ot[:, 3] = (mean_t / 100.0).astype(np.float32)
    item_global_ot[:, 4] = (np.sqrt(var_t) / 50.0).astype(np.float32)

    item_group_ot = np.zeros((G, B, 5), dtype=np.float32)
    gb_count_m = gb_count.reshape(G, B)
    gb_sum_r_m = gb_sum_r.reshape(G, B)
    gb_sum_r2_m = gb_sum_r2.reshape(G, B)
    gb_sum_img_m = gb_sum_img.reshape(G, B)
    gb_sum_t_m = gb_sum_t.reshape(G, B)
    gb_sum_t2_m = gb_sum_t2.reshape(G, B)
    cnt_gb = np.maximum(gb_count_m, 1.0)
    mean_r_gb = gb_sum_r_m / cnt_gb
    var_r_gb = np.maximum(gb_sum_r2_m / cnt_gb - mean_r_gb * mean_r_gb, 0.0)
    mean_t_gb = gb_sum_t_m / cnt_gb
    var_t_gb = np.maximum(gb_sum_t2_m / cnt_gb - mean_t_gb * mean_t_gb, 0.0)
    item_group_ot[:, :, 0] = (mean_r_gb / 5.0).astype(np.float32)
    item_group_ot[:, :, 1] = (np.sqrt(var_r_gb) / 2.0).astype(np.float32)
    item_group_ot[:, :, 2] = (gb_sum_img_m / cnt_gb).astype(np.float32)
    item_group_ot[:, :, 3] = (mean_t_gb / 100.0).astype(np.float32)
    item_group_ot[:, :, 4] = (np.sqrt(var_t_gb) / 50.0).astype(np.float32)

    item_global_ot_delta = np.zeros((B, 5), dtype=np.float32)
    item_global_ot_delta[1:] = item_global_ot[1:] - item_global_ot[:-1]
    item_group_ot_delta = np.zeros((G, B, 5), dtype=np.float32)
    item_group_ot_delta[:, 1:] = item_group_ot[:, 1:] - item_group_ot[:, :-1]

    with open(os.path.join(OUTPUT_DIR, "item_time_stats_lag.pkl"), "wb") as f:
        pickle.dump(
            {
                "item_global_ot": item_global_ot,
                "item_group_ot": item_group_ot,
                "item_global_ot_delta": item_global_ot_delta,
                "item_group_ot_delta": item_group_ot_delta,
                "meta": {
                    "source_split": "train_only",
                    "built_at": int(datetime.utcnow().timestamp()),
                },
            },
            f,
        )
    with open(os.path.join(OUTPUT_DIR, "item_time_stats_meta.json"), "w") as f:
        json.dump(
            {"source_split": "train_only", "built_at": int(datetime.utcnow().timestamp())},
            f,
            indent=2,
        )

    item_future_targets = {}
    for (it, bb), vec in per_key.items():
        if bb not in train_bin_set or (bb + 1) not in train_bin_set:
            item_future_targets[(it, bb)] = {"target": np.zeros(4, dtype=np.float32), "mask": False}
            continue
        nxt = per_key.get((it, bb + 1), None)
        if nxt is None or nxt[0] < 1:
            item_future_targets[(it, bb)] = {"target": np.zeros(4, dtype=np.float32), "mask": False}
            continue
        cnt = max(nxt[0], 1.0)
        target = np.zeros(4, dtype=np.float32)
        target[0] = float((nxt[1] / cnt) / 5.0)
        target[1] = float(nxt[2] / cnt)
        target[2] = float((nxt[3] / cnt) / 200.0)
        target[3] = float(np.log1p(cnt) / 5.0)
        item_future_targets[(it, bb)] = {"target": target, "mask": True}

    with open(os.path.join(OUTPUT_DIR, "item_future_targets.pkl"), "wb") as f:
        pickle.dump(
            {
                "data": item_future_targets,
                "meta": {
                    "source_split": "train_only",
                    "built_at": int(datetime.utcnow().timestamp()),
                },
            },
            f,
        )
    with open(os.path.join(OUTPUT_DIR, "item_future_targets_meta.json"), "w") as f:
        json.dump(
            {"source_split": "train_only", "built_at": int(datetime.utcnow().timestamp())},
            f,
            indent=2,
        )

    print(f"   item_histories (train-only): {len(item_histories):,} items")
    print(f"   item_neighbors (train-only): {len(item_neighbors):,} items")
    print(f"   train bins covered: {len(train_bin_set)}/{num_bins}")

# MAIN

def main():
    ensure_dirs()
    print("=" * 80)
    print("TS-SSM Preprocessing")
    print(f"Category: {CATEGORY}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Min Year: {MIN_YEAR}")
    print(f"K-core: {MIN_USER_REVIEWS}-core")
    print(f"Text emb: {USE_TEXT_EMB}  Image emb: {USE_IMAGE_EMB}")
    print("=" * 80)

    # Pass 1
    pass1_files = sorted(glob.glob(os.path.join(TEMP_DIR, "pass1_chunk_*.parquet")))
    counts_exist = (os.path.exists(os.path.join(TEMP_DIR, "user_counts.pkl")) and
                   os.path.exists(os.path.join(TEMP_DIR, "item_counts.pkl")))

    if pass1_files and counts_exist:
        print(f"\nSkip Pass 1 (found {len(pass1_files)} chunks + counts)")
    else:
        pass1_stream_dump_and_count()
    gc.collect()

    # Pass 2
    user2idx, item2idx, total_core = pass2_kcore_and_write_filtered()
    gc.collect()

    # Pass 3a
    user_histories, item_histories = pass3a_build_histories()
    gc.collect()

    # Pass 3b
    train_indices, val_indices, test_indices = pass3b_split(user_histories)
    gc.collect()

    # Pass 3c
    train_stats = pass3c_train_global_stats(train_indices, total_core)
    gc.collect()

    # Pass 4
    total_written, chunk_info = pass4_running_stats_and_write(
        user_histories, item_histories, train_stats, total_core
    )
    gc.collect()

    # Pass 5
    stats = pass5_final_stats(user2idx, item2idx, total_written, chunk_info)

    # Pass 5b
    pass5b_build_item_rep_table(
        item_histories=item_histories,
        train_indices=train_indices,
        total_records=total_written,
        num_items=len(item2idx) + 1,
        train_stats=train_stats,
        chunk_info=chunk_info,
    )
    gc.collect()

    # Pass 6: Time bin, groups, and O^t

    # Pass 6a: Time bin config
    time_bin_to_id = pass6a_build_time_bin_config(chunk_info)

    # Pass 6d: Write bin_id to parquet files before pass 6c.
    pass6d_add_bin_id_column(chunk_info, time_bin_to_id)

    # Pass 6b: User groups (train-only)
    user_groups = pass6b_compute_user_groups(
        user_histories=user_histories,
        train_indices=train_indices,
        total_records=total_written,
        chunk_info=chunk_info,
    )

    # Pass 6c: Global and group O^t stats (train-only)
    pass6c_compute_time_stats(
        train_indices=train_indices,
        total_records=total_written,
        chunk_info=chunk_info,
        time_bin_to_id=time_bin_to_id,
        user_groups=user_groups,
    )

    # Pass 6e: Item-side train-only resources (histories/neighbors/time stats/future targets)
    pass6e_build_item_side_train_resources(
        train_indices=train_indices,
        total_records=total_written,
        chunk_info=chunk_info,
        num_items=len(item2idx) + 1,
        num_bins=len(time_bin_to_id),
    )

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total reviews: {stats['total_reviews']:,}")
    print(f"Users: {stats['num_users']:,}  Items: {stats['num_items']:,}")
    print(f"Min year: {MIN_YEAR}")
    print(f"Time bins: {len(time_bin_to_id)}")
    print(f"User groups: {N_GROUPS}")
    print(f"Sparsity: {stats['sparsity']*100:.4f}%")
    print(f"Has image rate: {stats['modality']['has_image_rate']*100:.2f}%")

    if not KEEP_TEMP_DIR:
        print("\nCleaning temp dir...")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    return stats


if __name__ == "__main__":
    main()
