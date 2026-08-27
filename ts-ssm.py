#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS-SSM

Unified training and evaluation script for the TS-SSM model.
Includes user/item temporal state updates, MNAR-aware message passing,
and carryover-aware memory.
"""

import argparse
import json
import math
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


HOT_DATA_COLS = [
    "global_idx",
    "user_idx",
    "item_idx",
    "rating",
    "unix_time",
    "bin_id",
    "has_title",
    "has_text",
    "has_image",
    "len_title_tokens",
    "len_text_tokens",
    "u_num_prev_reviews",
    "u_prev_mean_rating",
    "u_prev_image_rate",
    "u_prev_mean_title_len",
    "u_prev_mean_text_len",
    "u_is_first_review",
    "u_time_since_prev",
    "u_is_image_unusual",
    "u_is_text_long_unusual",
    "u_is_title_long_unusual",
    "u_rating_delta",
    "i_num_prev_reviews",
    "i_prev_mean_rating",
    "i_prev_rating_std",
    "i_prev_image_rate",
    "i_prev_mean_text_len",
    "i_is_first_review",
    "i_rating_delta",
    "i_time_since_prev",
    "helpful_vote",
    "verified_purchase",
    "num_images",
    "day_of_week",
    "missing_pattern_tti",
    "missing_pattern",
]

HOT_INT_COLS = {
    "global_idx",
    "user_idx",
    "item_idx",
    "bin_id",
    "missing_pattern_tti",
    "missing_pattern",
}
HOT_FLOAT64_COLS = {"unix_time"}
HOT_EMBED_KINDS = ("title", "text", "image")


# Config


@dataclass
class Config:
    DATA_DIR: str = "./preprocessed_sequential"
    RESULTS_DIR: str = "./results_ts_ssm"
    BEST_CKPT_PATH: str = "./results_ts_ssm/best.pt"

    MAX_USER_SEQ_LEN: int = 64
    MAX_ITEM_SEQ_LEN: int = 40
    NUM_NEGATIVES_TRAIN: int = 48

    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    TITLE_EMB_DIM: int = 384
    TEXT_EMB_DIM: int = 384
    IMAGE_EMB_DIM: int = 512

    HIDDEN_DIM: int = 320
    DROPOUT: float = 0.1
    NUM_HEADS: int = 4
    NUM_LAYERS: int = 3
    ITEM_NUM_LAYERS: int = 2

    USER_NUM_FEAT_DIM: int = 18
    ITEM_NUM_FEAT_DIM: int = 12
    O_T_DIM: int = 5
    DEVIATION_DIM: int = 16
    ITEM_DEVIATION_DIM: int = 16
    SHOCK_FEAT_DIM: int = 8
    RELIABILITY_FEAT_DIM: int = 3
    CURRENT_EVENT_NUM_FEAT_DIM: int = 12  # rating + 3 availability flags + shock features

    N_GROUPS: int = 3
    N_ITEM_GROUPS: int = 8
    NUM_BINS: int = 112
    PAD_IDX: int = 0

    NUM_MISSING_PATTERNS: int = 8
    MISSING_PATTERN_DIM: int = 16

    # Bounded update schedule
    MAX_UPDATE_RATIO: float = 0.10
    MAX_UPDATE_RATIO_FINAL: float = 0.15
    UPDATE_RATIO_WARMUP_EPOCHS: int = 15
    ITEM_MAX_UPDATE_RATIO: float = 0.10
    ITEM_MAX_UPDATE_RATIO_FINAL: float = 0.15

    # Sampling and ranking
    USE_MIXED_NEG_SAMPLING: bool = True
    UNIFORM_NEG_RATIO: float = 0.6
    POPULARITY_ALPHA: float = 0.75
    USE_INBATCH_NEGATIVES: bool = False
    USE_ITEM_BIAS: bool = True
    USE_DYNAMIC_ITEM_BIAS: bool = True

    # BPR
    BPR_TEMPERATURE: float = 1.0
    USE_HARD_NEG_FOCUS: bool = False
    HARD_NEG_TEMPERATURE: float = 1.0
    BPR_TEMP_INIT: float = 1.0
    BPR_TEMP_FINAL: float = 0.5
    HARD_NEG_TEMP_INIT: float = 1.0
    HARD_NEG_TEMP_FINAL: float = 0.5
    TEMP_SCHEDULE_WARMUP_EPOCHS: int = 8

    # Message passing / carryover
    ITEM_NEIGHBOR_K: int = 20
    COOC_WINDOW: int = 3
    MESSAGE_DROPOUT: float = 0.1
    CARRYOVER_LAMBDA_POS_INIT: float = 0.25
    CARRYOVER_LAMBDA_NEG_INIT: float = 0.15

    EPOCHS: int = 50
    LR: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    AMP: bool = True
    AMP_DTYPE: str = "auto"
    SEED: int = 42

    USE_SCHEDULER: bool = True
    SCHEDULER_T_MAX: int = 50
    SCHEDULER_ETA_MIN: float = 1e-6
    WARMUP_EPOCHS: int = 8

    USE_EMA: bool = True
    EMA_DECAY: float = 0.999

    # Loss weights
    W_RANK: float = 1.0
    W_DRIFT: float = 0.08
    W_CARRY: float = 0.05
    W_RELIABILITY: float = 0.03

    AUX_WARMUP_EPOCHS: int = 5
    AUX_RAMPUP_EPOCHS: int = 5
    EARLY_STOP_PATIENCE: int = 10

    VAL_EVERY: int = 1
    FAST_VAL_EARLY_EPOCHS: int = 10
    FAST_VAL_EARLY_INTERVAL: int = 2
    VAL_MAX_BATCHES: int = 50
    SAMPLED_EVAL_NEGATIVES: int = 200
    K_VALUES: Tuple[int, ...] = (10, 20, 50)
    FULL_SORT_CHUNK_SIZE: int = 4096
    # <= 0 means the complete validation split.  Full validation is required
    # when checkpoint selection is based on small differences between metrics.
    FULL_SORT_VAL_BATCHES: int = 0

    MAX_SEEN_LEN: int = 5000
    SEEN_BLOCK_SIZE: int = 128

    PRELOAD_HOT_DATA: bool = True
    CACHE_SAMPLE_FEATURES: bool = True
    SAMPLE_CACHE_VERSION: int = 2

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Shared Resources


class SharedResources:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_dir = cfg.DATA_DIR

        print("[Shared] Loading mappings...")
        with open(os.path.join(self.data_dir, "mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
        self.num_items = int(mappings["num_items"])
        self.pad_idx = int(mappings.get("pad_idx", 0))

        print("[Shared] Loading time config...")
        with open(os.path.join(self.data_dir, "time_bin_config.json"), "r") as f:
            time_cfg = json.load(f)
        self.num_bins = int(time_cfg["num_bins"])
        self.n_groups = int(time_cfg["n_groups"])

        print("[Shared] Loading user histories and splits...")
        with open(os.path.join(self.data_dir, "user_histories.pkl"), "rb") as f:
            self.user_histories = pickle.load(f)
        with open(os.path.join(self.data_dir, "split_indices.pkl"), "rb") as f:
            self.splits = pickle.load(f)

        print("[Shared] Loading train stats...")
        with open(os.path.join(self.data_dir, "train_global_stats.pkl"), "rb") as f:
            self.train_stats = pickle.load(f)

        print("[Shared] Building global index mapper...")
        self._build_index_mapper()
        print("[Shared] Building train-only lookup...")
        self._build_train_lookup()

        print("[Shared] Loading user/group temporal stats...")
        self._load_user_time_stats_with_delta()

        print("[Shared] Loading user seen caches...")
        self._load_user_seen()
        self._load_user_item_seq()

        print("[Shared] Building target->(user,pos) map...")
        self._build_user_for_target()

        print("[Shared] Building item popularity counts...")
        self._build_item_counts()

        print("[Shared] Loading/building item-side resources...")
        self._load_or_build_item_groups()
        self._load_or_build_item_histories()
        self._load_or_build_item_neighbors()
        self._load_or_build_item_temporal_and_future_targets()
        self.has_hot_data = False
        if self.cfg.PRELOAD_HOT_DATA:
            print("[Shared] Preloading hot columns and embeddings...")
            self._load_hot_data_into_memory()

        print(
            f"[Shared] Ready: items={self.num_items}, bins={self.num_bins}, "
            f"user_groups={self.n_groups}, item_groups={self.n_item_groups}"
        )

    def _build_index_mapper(self):
        with open(os.path.join(self.data_dir, "chunk_info.json"), "r") as f:
            chunk_info = json.load(f)

        self.chunk_info = chunk_info
        self.chunk_paths = {m["chunk_id"]: os.path.join(self.data_dir, m["file"]) for m in chunk_info}
        self.max_global_idx = max(m["global_idx_max"] for m in chunk_info) + 1

        self.g2cid = np.full(self.max_global_idx, -1, dtype=np.int32)
        self.g2row = np.full(self.max_global_idx, -1, dtype=np.int32)

        for meta in tqdm(chunk_info, desc="Indexing chunks", leave=False):
            cid = int(meta["chunk_id"])
            path = os.path.join(self.data_dir, meta["file"])
            df = pd.read_parquet(path, columns=["global_idx"])
            idxs = df["global_idx"].to_numpy(np.int64)
            self.g2cid[idxs] = cid
            self.g2row[idxs] = np.arange(len(idxs), dtype=np.int32)
            del df

    def _build_train_lookup(self):
        train = np.asarray(self.splits.get("train", []), dtype=np.int64)
        train = train[(train >= 0) & (train < self.max_global_idx)]
        self.train_indices = train
        self.train_mask = np.zeros(self.max_global_idx, dtype=np.bool_)
        if train.size > 0:
            self.train_mask[train] = True

        self.user_train_histories: Dict[int, np.ndarray] = {}
        for u, seq in self.user_histories.items():
            g = np.asarray(seq, dtype=np.int64)
            if g.size == 0:
                continue
            valid = (g >= 0) & (g < self.max_global_idx)
            g = g[valid]
            if g.size == 0:
                continue
            g = g[self.train_mask[g]]
            if g.size > 0:
                self.user_train_histories[int(u)] = g

        self.train_bin_set = set()
        if train.size > 0:
            cids = self.g2cid[train]
            rows = self.g2row[train]
            valid = cids != -1
            cids = cids[valid]
            rows = rows[valid]
            for cid in np.unique(cids):
                path = self.chunk_paths.get(int(cid))
                if path is None or not os.path.exists(path):
                    continue
                try:
                    df = pd.read_parquet(path, columns=["bin_id"])
                except Exception:
                    continue
                bins = df["bin_id"].to_numpy(np.int64, copy=False)
                r = rows[cids == cid]
                r = r[(r >= 0) & (r < bins.shape[0])]
                if r.size > 0:
                    b = bins[r]
                    b = b[(b >= 0) & (b < self.num_bins)]
                    self.train_bin_set.update(np.unique(b).tolist())
                del df

        if len(self.train_bin_set) > 0:
            self.train_bin_min = int(min(self.train_bin_set))
            self.train_bin_max = int(max(self.train_bin_set))
        else:
            self.train_bin_min = 0
            self.train_bin_max = max(0, self.num_bins - 1)

    @staticmethod
    def _load_meta(meta_path: str) -> Dict[str, Any]:
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _save_meta(meta_path: str, meta: Dict[str, Any]):
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def _validate_item_histories_train_only(
        self,
        item_histories: Dict[int, np.ndarray],
        item_hist_times: Dict[int, np.ndarray],
    ) -> bool:
        for it, seq in item_histories.items():
            gids = np.asarray(seq, dtype=np.int64)
            times = np.asarray(item_hist_times.get(int(it), np.zeros(0, dtype=np.float64)), dtype=np.float64)
            if gids.size != times.size:
                return False
            if gids.size == 0:
                continue
            valid = (gids >= 0) & (gids < self.max_global_idx)
            if not valid.all():
                return False
            if not self.train_mask[gids].all():
                return False
            if np.any(np.diff(times) < 0):
                return False
        return True

    def _validate_item_neighbors(self, item_neighbors: Dict[int, np.ndarray]) -> bool:
        for it, seq in item_neighbors.items():
            if int(it) < 0 or int(it) >= self.num_items:
                return False
            arr = np.asarray(seq, dtype=np.int64)
            if arr.size == 0:
                continue
            if ((arr < 0) | (arr >= self.num_items) | (arr == self.pad_idx)).any():
                return False
        return True

    def _load_user_time_stats_with_delta(self):
        with open(os.path.join(self.data_dir, "global_time_stats_lag.pkl"), "rb") as f:
            global_lag = pickle.load(f)
        with open(os.path.join(self.data_dir, "group_time_stats_lag.pkl"), "rb") as f:
            group_lag = pickle.load(f)
        with open(os.path.join(self.data_dir, "user_groups_train_only.pkl"), "rb") as f:
            self.user_groups = pickle.load(f)

        self.global_ot = np.zeros((self.num_bins, 5), dtype=np.float32)
        for b in range(self.num_bins):
            ot = global_lag.get(b, {})
            self.global_ot[b, 0] = ot.get("rating_mean", 4.0) / 5.0
            self.global_ot[b, 1] = ot.get("rating_std", 1.0) / 2.0
            self.global_ot[b, 2] = ot.get("image_rate", 0.08)
            self.global_ot[b, 3] = ot.get("text_len_mean", 50.0) / 100.0
            self.global_ot[b, 4] = ot.get("text_len_std", 30.0) / 50.0

        self.global_ot_delta = np.zeros((self.num_bins, 5), dtype=np.float32)
        for b in range(1, self.num_bins):
            self.global_ot_delta[b] = self.global_ot[b] - self.global_ot[b - 1]

        self.group_ot = np.zeros((self.n_groups, self.num_bins, 5), dtype=np.float32)
        for g in range(self.n_groups):
            for b in range(self.num_bins):
                ot = group_lag.get((g, b), {})
                self.group_ot[g, b, 0] = ot.get("rating_mean", 4.0) / 5.0
                self.group_ot[g, b, 1] = ot.get("rating_std", 1.0) / 2.0
                self.group_ot[g, b, 2] = ot.get("image_rate", 0.08)
                self.group_ot[g, b, 3] = ot.get("text_len_mean", 50.0) / 100.0
                self.group_ot[g, b, 4] = ot.get("text_len_std", 30.0) / 50.0

        self.group_ot_delta = np.zeros((self.n_groups, self.num_bins, 5), dtype=np.float32)
        for g in range(self.n_groups):
            for b in range(1, self.num_bins):
                self.group_ot_delta[g, b] = self.group_ot[g, b] - self.group_ot[g, b - 1]

    def _load_user_seen(self):
        seen_path = os.path.join(self.data_dir, "user_seen_items.pkl")
        if os.path.exists(seen_path):
            with open(seen_path, "rb") as f:
                user_seen_dict = pickle.load(f)
            self.user_seen_np: Dict[int, np.ndarray] = {}
            for u, items in user_seen_dict.items():
                arr = np.asarray(items, dtype=np.int64)
                if len(arr) > self.cfg.MAX_SEEN_LEN:
                    arr = arr[-self.cfg.MAX_SEEN_LEN :]
                self.user_seen_np[int(u)] = arr
        else:
            self.user_seen_np = {}

    def _load_user_item_seq(self):
        item_seq_path = os.path.join(self.data_dir, "user_item_seq.pkl")
        if os.path.exists(item_seq_path):
            with open(item_seq_path, "rb") as f:
                self.user_item_seq = pickle.load(f)
            self._use_fast_seen = True
        else:
            self.user_item_seq = {}
            self._use_fast_seen = False

    def _build_user_for_target(self):
        self.user_for_target: Dict[int, Tuple[int, int]] = {}
        for u, seq in self.user_histories.items():
            for pos, g in enumerate(seq):
                self.user_for_target[int(g)] = (int(u), int(pos))

    def _build_item_counts(self):
        train_indices = self.splits.get("train", [])
        item_counts = np.zeros(self.num_items, dtype=np.float32)
        if len(train_indices) == 0:
            pop = np.ones(self.num_items, dtype=np.float32)
            pop[self.pad_idx] = 0.0
            self.item_counts = item_counts
            self.item_pop_probs = pop / (pop.sum() + 1e-8)
            return

        g = np.asarray(train_indices, dtype=np.int64)
        g = g[(g >= 0) & (g < self.max_global_idx)]
        cids = self.g2cid[g]
        rows = self.g2row[g]
        valid = (cids != -1)
        cids = cids[valid]
        rows = rows[valid]

        for cid in tqdm(np.unique(cids), desc="Counting item popularity", leave=False):
            path = self.chunk_paths.get(int(cid))
            if path is None or not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path, columns=["item_idx"])
            except Exception:
                continue
            items = df["item_idx"].to_numpy(dtype=np.int64, copy=False)
            r = rows[cids == cid]
            r = r[(r >= 0) & (r < items.shape[0])]
            if r.size == 0:
                continue
            sel = items[r]
            sel = sel[(sel >= 0) & (sel < self.num_items) & (sel != self.pad_idx)]
            if sel.size == 0:
                continue
            item_counts += np.bincount(sel, minlength=self.num_items).astype(np.float32, copy=False)

        alpha = float(self.cfg.POPULARITY_ALPHA)
        smoothed = np.power(item_counts + 1.0, alpha).astype(np.float32, copy=False)
        smoothed[self.pad_idx] = 0.0
        pop = smoothed / (smoothed.sum() + 1e-8)
        self.item_counts = item_counts
        self.item_pop_probs = pop

    def _load_or_build_item_groups(self):
        path = os.path.join(self.data_dir, "item_groups.pkl")
        n_item_groups = int(self.cfg.N_ITEM_GROUPS)
        if os.path.exists(path):
            with open(path, "rb") as f:
                groups_obj = pickle.load(f)
            if isinstance(groups_obj, dict):
                arr = np.zeros(self.num_items, dtype=np.int64)
                for i, g in groups_obj.items():
                    ii = int(i)
                    if 0 <= ii < self.num_items:
                        arr[ii] = int(g)
                self.item_group_arr = np.clip(arr, 0, n_item_groups - 1)
            else:
                arr = np.asarray(groups_obj, dtype=np.int64)
                if arr.shape[0] < self.num_items:
                    tmp = np.zeros(self.num_items, dtype=np.int64)
                    tmp[: arr.shape[0]] = arr
                    arr = tmp
                self.item_group_arr = np.clip(arr[: self.num_items], 0, n_item_groups - 1)
        else:
            order = np.argsort(self.item_counts)
            arr = np.zeros(self.num_items, dtype=np.int64)
            binsz = max(1, len(order) // n_item_groups)
            for g in range(n_item_groups):
                s = g * binsz
                e = (g + 1) * binsz if g < n_item_groups - 1 else len(order)
                arr[order[s:e]] = g
            arr[self.pad_idx] = 0
            self.item_group_arr = arr
        self.n_item_groups = n_item_groups

    def _build_global_item_array(self):
        arr = np.full(self.max_global_idx, self.pad_idx, dtype=np.int64)
        for meta in tqdm(self.chunk_info, desc="Building global->item map", leave=False):
            path = os.path.join(self.data_dir, meta["file"])
            try:
                df = pd.read_parquet(path, columns=["global_idx", "item_idx"])
            except Exception:
                continue
            g = df["global_idx"].to_numpy(np.int64, copy=False)
            it = df["item_idx"].to_numpy(np.int64, copy=False)
            mask = (g >= 0) & (g < self.max_global_idx) & (it >= 0) & (it < self.num_items)
            arr[g[mask]] = it[mask]
            del df
        self.global_item_arr = arr

    def _load_or_build_item_histories(self):
        hist_path = os.path.join(self.data_dir, "item_histories.pkl")
        times_path = os.path.join(self.data_dir, "item_hist_times.pkl")
        meta_path = os.path.join(self.data_dir, "item_histories_meta.json")

        if os.path.exists(hist_path) and os.path.exists(times_path):
            with open(hist_path, "rb") as f:
                obj_hist = pickle.load(f)
            with open(times_path, "rb") as f:
                obj_times = pickle.load(f)
            self.item_histories = {int(k): np.asarray(v, dtype=np.int64) for k, v in obj_hist.items()}
            self.item_hist_times = {int(k): np.asarray(v, dtype=np.float64) for k, v in obj_times.items()}

            meta = self._load_meta(meta_path)
            meta_ok = meta.get("source_split", "") == "train_only"
            valid = self._validate_item_histories_train_only(self.item_histories, self.item_hist_times)
            if meta_ok and valid:
                return

            print("[Shared][WARN] item_histories cache is not train-only or failed validation; rebuilding.")

        hist_tmp: Dict[int, List[int]] = defaultdict(list)
        time_tmp: Dict[int, List[float]] = defaultdict(list)
        for m in tqdm(self.chunk_info, desc="Building item histories (train-only)", leave=False):
            path = os.path.join(self.data_dir, m["file"])
            try:
                df = pd.read_parquet(path, columns=["item_idx", "global_idx", "unix_time"])
            except Exception:
                continue
            items = df["item_idx"].to_numpy(np.int64, copy=False)
            gids = df["global_idx"].to_numpy(np.int64, copy=False)
            ts = df["unix_time"].to_numpy(np.float64, copy=False)
            g_valid = (gids >= 0) & (gids < self.max_global_idx)
            in_train = np.zeros_like(g_valid)
            in_train[g_valid] = self.train_mask[gids[g_valid]]
            mask = (items >= 0) & (items < self.num_items) & (items != self.pad_idx)
            mask &= g_valid
            mask &= in_train
            items = items[mask]
            gids = gids[mask]
            ts = ts[mask]
            for it, g, t in zip(items.tolist(), gids.tolist(), ts.tolist()):
                hist_tmp[it].append(int(g))
                time_tmp[it].append(float(t))
            del df

        self.item_histories: Dict[int, np.ndarray] = {}
        self.item_hist_times: Dict[int, np.ndarray] = {}
        for it, seq in tqdm(hist_tmp.items(), desc="Sorting item histories", leave=False):
            times = np.asarray(time_tmp[it], dtype=np.float64)
            gids = np.asarray(seq, dtype=np.int64)
            order = np.argsort(times, kind="mergesort")
            self.item_histories[int(it)] = gids[order]
            self.item_hist_times[int(it)] = times[order]

        with open(hist_path, "wb") as f:
            pickle.dump(self.item_histories, f)
        with open(times_path, "wb") as f:
            pickle.dump(self.item_hist_times, f)
        self._save_meta(
            meta_path,
            {
                "source_split": "train_only",
                "num_items_with_history": len(self.item_histories),
                "built_at": int(time.time()),
            },
        )

    def _load_or_build_item_neighbors(self):
        path = os.path.join(self.data_dir, "item_neighbors.pkl")
        meta_path = os.path.join(self.data_dir, "item_neighbors_meta.json")
        if os.path.exists(path):
            with open(path, "rb") as f:
                neighbors_obj = pickle.load(f)
            if isinstance(neighbors_obj, dict):
                self.item_neighbors = {}
                for k, v in neighbors_obj.items():
                    arr = np.asarray(v, dtype=np.int64)
                    arr = arr[(arr >= 0) & (arr < self.num_items) & (arr != self.pad_idx)]
                    self.item_neighbors[int(k)] = arr
            else:
                self.item_neighbors = {}
                for i, v in enumerate(neighbors_obj):
                    arr = np.asarray(v, dtype=np.int64)
                    arr = arr[(arr >= 0) & (arr < self.num_items) & (arr != self.pad_idx)]
                    self.item_neighbors[int(i)] = arr
            meta = self._load_meta(meta_path)
            meta_ok = meta.get("source_split", "") == "train_only"
            valid = self._validate_item_neighbors(self.item_neighbors)
            if meta_ok and valid:
                return
            print("[Shared][WARN] item_neighbors cache is not train-only or failed validation; rebuilding.")

        if not hasattr(self, "global_item_arr"):
            self._build_global_item_array()

        window = int(self.cfg.COOC_WINDOW)
        k = int(self.cfg.ITEM_NEIGHBOR_K)
        cooc: Dict[int, Counter] = defaultdict(Counter)

        for _, seq in tqdm(self.user_train_histories.items(), desc="Building item co-occurrence graph (train-only)", leave=False):
            if len(seq) < 2:
                continue
            g = np.asarray(seq, dtype=np.int64)
            valid = (g >= 0) & (g < self.max_global_idx)
            g = g[valid]
            if g.size < 2:
                continue
            items = self.global_item_arr[g]
            items = items[(items >= 0) & (items < self.num_items) & (items != self.pad_idx)]
            if items.size < 2:
                continue
            L = items.size
            for i in range(L):
                a = int(items[i])
                end = min(L, i + window + 1)
                for j in range(i + 1, end):
                    b = int(items[j])
                    if a == b:
                        continue
                    cooc[a][b] += 1
                    cooc[b][a] += 1

        self.item_neighbors: Dict[int, np.ndarray] = {}
        for it in range(self.num_items):
            ctr = cooc.get(it, None)
            if ctr is None or len(ctr) == 0:
                self.item_neighbors[it] = np.zeros(0, dtype=np.int64)
                continue
            top = ctr.most_common(k)
            self.item_neighbors[it] = np.asarray([x[0] for x in top], dtype=np.int64)

        with open(path, "wb") as f:
            pickle.dump(self.item_neighbors, f)
        self._save_meta(
            meta_path,
            {
                "source_split": "train_only",
                "window": int(window),
                "topk": int(k),
                "built_at": int(time.time()),
            },
        )

    def _load_or_build_item_temporal_and_future_targets(self):
        stats_path = os.path.join(self.data_dir, "item_time_stats_lag.pkl")
        fut_path = os.path.join(self.data_dir, "item_future_targets.pkl")
        stats_meta_path = os.path.join(self.data_dir, "item_time_stats_meta.json")
        fut_meta_path = os.path.join(self.data_dir, "item_future_targets_meta.json")

        stats_ok = False
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict) and "item_global_ot" in obj:
                    self.item_global_ot = np.asarray(obj["item_global_ot"], dtype=np.float32)
                    self.item_group_ot = np.asarray(obj["item_group_ot"], dtype=np.float32)
                    self.item_global_ot_delta = np.asarray(obj["item_global_ot_delta"], dtype=np.float32)
                    self.item_group_ot_delta = np.asarray(obj["item_group_ot_delta"], dtype=np.float32)
                    meta = obj.get("meta", {})
                    if not isinstance(meta, dict) or meta.get("source_split", "") != "train_only":
                        meta = self._load_meta(stats_meta_path)
                    stats_ok = isinstance(meta, dict) and meta.get("source_split", "") == "train_only"
            except Exception:
                stats_ok = False
        if not stats_ok:
            print("[Shared][WARN] item_time_stats cache is not train-only; rebuilding.")
            self._build_item_temporal_stats_from_chunks()
            with open(stats_path, "wb") as f:
                pickle.dump(
                    {
                        "item_global_ot": self.item_global_ot,
                        "item_group_ot": self.item_group_ot,
                        "item_global_ot_delta": self.item_global_ot_delta,
                        "item_group_ot_delta": self.item_group_ot_delta,
                        "meta": {
                            "source_split": "train_only",
                            "built_at": int(time.time()),
                        },
                    },
                    f,
                )
            self._save_meta(stats_meta_path, {"source_split": "train_only", "built_at": int(time.time())})

        fut_ok = False
        if os.path.exists(fut_path):
            try:
                with open(fut_path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict) and "data" in obj and "meta" in obj:
                    self.item_future_targets = obj["data"]
                    meta = obj["meta"] if isinstance(obj["meta"], dict) else {}
                    fut_ok = meta.get("source_split", "") == "train_only"
                elif isinstance(obj, dict):
                    self.item_future_targets = obj
                    meta = self._load_meta(fut_meta_path)
                    fut_ok = meta.get("source_split", "") == "train_only"
            except Exception:
                fut_ok = False
        if not fut_ok:
            print("[Shared][WARN] item_future_targets cache is not train-only; rebuilding.")
            self._build_item_future_targets_from_chunks()
            with open(fut_path, "wb") as f:
                pickle.dump(
                    {
                        "data": self.item_future_targets,
                        "meta": {
                            "source_split": "train_only",
                            "built_at": int(time.time()),
                        },
                    },
                    f,
                )
            self._save_meta(fut_meta_path, {"source_split": "train_only", "built_at": int(time.time())})

    def _build_item_temporal_stats_from_chunks(self):
        B = self.num_bins
        G = self.n_item_groups

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

        for meta in tqdm(self.chunk_info, desc="Building item temporal stats", leave=False):
            path = os.path.join(self.data_dir, meta["file"])
            try:
                df = pd.read_parquet(
                    path,
                    columns=["global_idx", "item_idx", "bin_id", "rating", "has_image", "len_text_tokens"],
                )
            except Exception:
                continue

            gids = df["global_idx"].to_numpy(np.int64, copy=False)
            item = df["item_idx"].to_numpy(np.int64, copy=False)
            b = df["bin_id"].to_numpy(np.int64, copy=False)
            r = df["rating"].to_numpy(np.float64, copy=False)
            img = df["has_image"].to_numpy(np.float64, copy=False)
            tl = df["len_text_tokens"].to_numpy(np.float64, copy=False)
            g_valid = (gids >= 0) & (gids < self.max_global_idx)
            in_train = np.zeros_like(g_valid)
            in_train[g_valid] = self.train_mask[gids[g_valid]]

            mask = (
                (item >= 0)
                & (item < self.num_items)
                & (item != self.pad_idx)
                & (b >= 0)
                & (b < B)
            )
            mask &= g_valid & in_train
            item = item[mask]
            b = b[mask]
            r = r[mask]
            img = img[mask]
            tl = tl[mask]
            if b.size == 0:
                continue

            g_count += np.bincount(b, minlength=B)
            g_sum_r += np.bincount(b, weights=r, minlength=B)
            g_sum_r2 += np.bincount(b, weights=r * r, minlength=B)
            g_sum_img += np.bincount(b, weights=img, minlength=B)
            g_sum_t += np.bincount(b, weights=tl, minlength=B)
            g_sum_t2 += np.bincount(b, weights=tl * tl, minlength=B)

            grp = self.item_group_arr[item]
            gb = grp * B + b
            gb_count += np.bincount(gb, minlength=G * B)
            gb_sum_r += np.bincount(gb, weights=r, minlength=G * B)
            gb_sum_r2 += np.bincount(gb, weights=r * r, minlength=G * B)
            gb_sum_img += np.bincount(gb, weights=img, minlength=G * B)
            gb_sum_t += np.bincount(gb, weights=tl, minlength=G * B)
            gb_sum_t2 += np.bincount(gb, weights=tl * tl, minlength=G * B)
            del df

        self.item_global_ot = np.zeros((B, 5), dtype=np.float32)
        cnt = np.maximum(g_count, 1.0)
        mean_r = g_sum_r / cnt
        var_r = np.maximum(g_sum_r2 / cnt - mean_r * mean_r, 0.0)
        mean_t = g_sum_t / cnt
        var_t = np.maximum(g_sum_t2 / cnt - mean_t * mean_t, 0.0)
        self.item_global_ot[:, 0] = (mean_r / 5.0).astype(np.float32)
        self.item_global_ot[:, 1] = (np.sqrt(var_r) / 2.0).astype(np.float32)
        self.item_global_ot[:, 2] = (g_sum_img / cnt).astype(np.float32)
        self.item_global_ot[:, 3] = (mean_t / 100.0).astype(np.float32)
        self.item_global_ot[:, 4] = (np.sqrt(var_t) / 50.0).astype(np.float32)

        self.item_group_ot = np.zeros((G, B, 5), dtype=np.float32)
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
        self.item_group_ot[:, :, 0] = (mean_r_gb / 5.0).astype(np.float32)
        self.item_group_ot[:, :, 1] = (np.sqrt(var_r_gb) / 2.0).astype(np.float32)
        self.item_group_ot[:, :, 2] = (gb_sum_img_m / cnt_gb).astype(np.float32)
        self.item_group_ot[:, :, 3] = (mean_t_gb / 100.0).astype(np.float32)
        self.item_group_ot[:, :, 4] = (np.sqrt(var_t_gb) / 50.0).astype(np.float32)

        self.item_global_ot_delta = np.zeros((B, 5), dtype=np.float32)
        self.item_global_ot_delta[1:] = self.item_global_ot[1:] - self.item_global_ot[:-1]
        self.item_group_ot_delta = np.zeros((G, B, 5), dtype=np.float32)
        self.item_group_ot_delta[:, 1:] = self.item_group_ot[:, 1:] - self.item_group_ot[:, :-1]

    def _build_item_future_targets_from_chunks(self):
        per_key = defaultdict(lambda: np.zeros(4, dtype=np.float64))
        # [count, sum_rating, sum_img, sum_text]
        for meta in tqdm(self.chunk_info, desc="Building item future targets", leave=False):
            path = os.path.join(self.data_dir, meta["file"])
            try:
                df = pd.read_parquet(
                    path,
                    columns=["global_idx", "item_idx", "bin_id", "rating", "has_image", "len_text_tokens"],
                )
            except Exception:
                continue
            gids = df["global_idx"].to_numpy(np.int64, copy=False)
            item = df["item_idx"].to_numpy(np.int64, copy=False)
            b = df["bin_id"].to_numpy(np.int64, copy=False)
            r = df["rating"].to_numpy(np.float64, copy=False)
            img = df["has_image"].to_numpy(np.float64, copy=False)
            tl = df["len_text_tokens"].to_numpy(np.float64, copy=False)
            g_valid = (gids >= 0) & (gids < self.max_global_idx)
            in_train = np.zeros_like(g_valid)
            in_train[g_valid] = self.train_mask[gids[g_valid]]
            mask = (
                (item >= 0)
                & (item < self.num_items)
                & (item != self.pad_idx)
                & (b >= 0)
                & (b < self.num_bins)
            )
            mask &= g_valid & in_train
            item = item[mask]
            b = b[mask]
            r = r[mask]
            img = img[mask]
            tl = tl[mask]
            for it, bb, rr, ii, tt in zip(item.tolist(), b.tolist(), r.tolist(), img.tolist(), tl.tolist()):
                key = (int(it), int(bb))
                per_key[key][0] += 1.0
                per_key[key][1] += float(rr)
                per_key[key][2] += float(ii)
                per_key[key][3] += float(tt)
            del df

        self.item_future_targets: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for (it, bb), vec in per_key.items():
            if bb not in self.train_bin_set or (bb + 1) not in self.train_bin_set:
                self.item_future_targets[(it, bb)] = {
                    "target": np.zeros(4, dtype=np.float32),
                    "mask": False,
                }
                continue
            nxt = per_key.get((it, bb + 1), None)
            if nxt is None or nxt[0] < 1:
                self.item_future_targets[(it, bb)] = {
                    "target": np.zeros(4, dtype=np.float32),
                    "mask": False,
                }
                continue
            cnt = max(nxt[0], 1.0)
            target = np.zeros(4, dtype=np.float32)
            target[0] = float((nxt[1] / cnt) / 5.0)  # next mean rating
            target[1] = float(nxt[2] / cnt)  # next image rate
            target[2] = float((nxt[3] / cnt) / 200.0)  # next mean text len
            target[3] = float(np.log1p(cnt) / 5.0)  # next review volume proxy
            self.item_future_targets[(it, bb)] = {"target": target, "mask": True}

    def _load_hot_data_into_memory(self):
        hot_arrays: Dict[str, np.ndarray] = {}
        for col in HOT_DATA_COLS:
            if col in HOT_INT_COLS:
                hot_arrays[col] = np.zeros(self.max_global_idx, dtype=np.int64)
            elif col in HOT_FLOAT64_COLS:
                hot_arrays[col] = np.zeros(self.max_global_idx, dtype=np.float64)
            else:
                hot_arrays[col] = np.zeros(self.max_global_idx, dtype=np.float32)

        title_all = np.zeros((self.max_global_idx, self.cfg.TITLE_EMB_DIM), dtype=np.float32)
        text_all = np.zeros((self.max_global_idx, self.cfg.TEXT_EMB_DIM), dtype=np.float32)
        image_all = np.zeros((self.max_global_idx, self.cfg.IMAGE_EMB_DIM), dtype=np.float32)
        emb_targets = {
            "title": title_all,
            "text": text_all,
            "image": image_all,
        }

        for meta in tqdm(self.chunk_info, desc="Preloading hot data", leave=False):
            cid = int(meta["chunk_id"])
            path = os.path.join(self.data_dir, meta["file"])
            try:
                df = pd.read_parquet(path, columns=HOT_DATA_COLS)
            except Exception:
                continue

            gids = df["global_idx"].to_numpy(np.int64, copy=False)
            valid = (gids >= 0) & (gids < self.max_global_idx)
            if not valid.any():
                del df
                continue
            gids_valid = gids[valid]

            for col in HOT_DATA_COLS:
                if col not in df.columns:
                    continue
                hot_arrays[col][gids_valid] = df[col].to_numpy(dtype=hot_arrays[col].dtype, copy=False)[valid]

            for kind, dst in emb_targets.items():
                emb_path = os.path.join(self.data_dir, "data", f"chunk_{cid:04d}_{kind}_emb.npy")
                if not os.path.exists(emb_path):
                    continue
                arr = np.load(emb_path, mmap_mode="r")
                dst[gids_valid] = np.asarray(arr[valid], dtype=np.float32)

            del df

        self.hot_arrays = hot_arrays
        self.title_emb_all = title_all
        self.text_emb_all = text_all
        self.image_emb_all = image_all
        self.has_hot_data = True

    def get_hot_values(self, global_idxs: np.ndarray, cols: List[str]) -> Dict[str, np.ndarray]:
        g = np.asarray(global_idxs, dtype=np.int64)
        if g.size == 0:
            out: Dict[str, np.ndarray] = {}
            for c in cols:
                if c in HOT_INT_COLS:
                    out[c] = np.zeros(0, dtype=np.int64)
                elif c in HOT_FLOAT64_COLS:
                    out[c] = np.zeros(0, dtype=np.float64)
                else:
                    out[c] = np.zeros(0, dtype=np.float32)
            return out

        out = {}
        valid = (g >= 0) & (g < self.max_global_idx)
        clipped = np.clip(g, 0, max(0, self.max_global_idx - 1))
        for c in cols:
            src = self.hot_arrays[c]
            arr = src[clipped].copy()
            if not valid.all():
                arr[~valid] = 0
            out[c] = arr
        return out

    def get_hot_embeddings(self, global_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = np.asarray(global_idxs, dtype=np.int64)
        if g.size == 0:
            return (
                np.zeros((0, self.cfg.TITLE_EMB_DIM), dtype=np.float32),
                np.zeros((0, self.cfg.TEXT_EMB_DIM), dtype=np.float32),
                np.zeros((0, self.cfg.IMAGE_EMB_DIM), dtype=np.float32),
            )
        valid = (g >= 0) & (g < self.max_global_idx)
        clipped = np.clip(g, 0, max(0, self.max_global_idx - 1))
        title = self.title_emb_all[clipped].copy()
        text = self.text_emb_all[clipped].copy()
        image = self.image_emb_all[clipped].copy()
        if not valid.all():
            title[~valid] = 0.0
            text[~valid] = 0.0
            image[~valid] = 0.0
        return title, text, image

    def is_valid_global_idx(self, g: int) -> bool:
        return 0 <= g < self.max_global_idx and self.g2cid[g] != -1

    def get_loc(self, global_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = global_idxs.astype(np.int64, copy=False)
        return self.g2cid[g], self.g2row[g]

    def get_user_group(self, user_idx: int) -> int:
        return int(self.user_groups.get(int(user_idx), 0))

    def get_global_ot(self, bin_id: int) -> np.ndarray:
        return self.global_ot[min(max(int(bin_id), 0), self.num_bins - 1)]

    def get_group_ot(self, group_id: int, bin_id: int) -> np.ndarray:
        g = min(max(int(group_id), 0), self.n_groups - 1)
        b = min(max(int(bin_id), 0), self.num_bins - 1)
        return self.group_ot[g, b]

    def get_global_ot_delta(self, bin_id: int) -> np.ndarray:
        return self.global_ot_delta[min(max(int(bin_id), 0), self.num_bins - 1)]

    def get_group_ot_delta(self, group_id: int, bin_id: int) -> np.ndarray:
        g = min(max(int(group_id), 0), self.n_groups - 1)
        b = min(max(int(bin_id), 0), self.num_bins - 1)
        return self.group_ot_delta[g, b]

    def get_item_group(self, item_idx: int) -> int:
        if 0 <= int(item_idx) < self.num_items:
            return int(self.item_group_arr[int(item_idx)])
        return 0

    def get_item_global_ot(self, bin_id: int) -> np.ndarray:
        return self.item_global_ot[min(max(int(bin_id), 0), self.num_bins - 1)]

    def get_item_group_ot(self, item_group: int, bin_id: int) -> np.ndarray:
        g = min(max(int(item_group), 0), self.n_item_groups - 1)
        b = min(max(int(bin_id), 0), self.num_bins - 1)
        return self.item_group_ot[g, b]

    def get_item_global_ot_delta(self, bin_id: int) -> np.ndarray:
        return self.item_global_ot_delta[min(max(int(bin_id), 0), self.num_bins - 1)]

    def get_item_group_ot_delta(self, item_group: int, bin_id: int) -> np.ndarray:
        g = min(max(int(item_group), 0), self.n_item_groups - 1)
        b = min(max(int(bin_id), 0), self.num_bins - 1)
        return self.item_group_ot_delta[g, b]

    def get_item_history_before(
        self,
        item_idx: int,
        target_unix: float,
        target_global_idx: int,
        max_len: int,
    ) -> np.ndarray:
        item = int(item_idx)
        if item not in self.item_histories:
            return np.zeros(0, dtype=np.int64)
        gids = self.item_histories[item]
        ts = self.item_hist_times[item]
        pos = int(np.searchsorted(ts, float(target_unix), side="left"))
        if pos <= 0:
            return np.zeros(0, dtype=np.int64)
        hist = gids[max(0, pos - max_len) : pos]
        if hist.size == 0:
            return hist
        if target_global_idx >= 0:
            hist = hist[hist != int(target_global_idx)]
        if hist.size == 0:
            return hist
        valid = np.array([self.is_valid_global_idx(int(g)) for g in hist], dtype=np.bool_)
        valid &= self.train_mask[hist]
        return hist[valid]

    def get_item_neighbors(self, item_idx: int, k: int) -> np.ndarray:
        arr = self.item_neighbors.get(int(item_idx), np.zeros(0, dtype=np.int64))
        if arr.size >= k:
            return arr[:k].astype(np.int64, copy=False)
        if arr.size == 0:
            return np.zeros(k, dtype=np.int64)
        out = np.zeros(k, dtype=np.int64)
        out[: arr.size] = arr
        return out

    def get_item_future_target(self, item_idx: int, bin_id: int) -> Tuple[np.ndarray, bool]:
        d = self.item_future_targets.get((int(item_idx), int(bin_id)))
        if d is None:
            return np.zeros(4, dtype=np.float32), False
        return np.asarray(d["target"], dtype=np.float32), bool(d["mask"])

    def get_user_seen_np(self, user_idx: int) -> np.ndarray:
        return self.user_seen_np.get(int(user_idx), np.array([], dtype=np.int64))

    def get_user_seen_items_fast(self, user_idx: int, pos_in_user: int) -> np.ndarray:
        if self._use_fast_seen and user_idx in self.user_item_seq and pos_in_user > 0:
            seq = self.user_item_seq[user_idx]
            return np.asarray(seq[:pos_in_user], dtype=np.int64)
        return np.array([], dtype=np.int64)


# Chunk cache and embedding store


class ChunkCache:
    NEEDED_COLS = HOT_DATA_COLS

    def __init__(self, shared: SharedResources, max_chunks: int = 32):
        self.shared = shared
        self.max_chunks = max_chunks
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._lru: List[int] = []

    def _ensure(self, cid: int):
        if cid in self._cache:
            if cid in self._lru:
                self._lru.remove(cid)
                self._lru.append(cid)
            return
        path = self.shared.chunk_paths.get(cid)
        if path is None or not os.path.exists(path):
            return
        df = pd.read_parquet(path, columns=self.NEEDED_COLS)
        self._cache[cid] = {c: df[c].to_numpy() for c in self.NEEDED_COLS if c in df.columns}
        del df
        self._lru.append(cid)
        if len(self._lru) > self.max_chunks:
            old = self._lru.pop(0)
            self._cache.pop(old, None)

    def get(self, cids: np.ndarray, rows: np.ndarray, cols: List[str]) -> Dict[str, np.ndarray]:
        n = len(rows)
        out: Dict[str, np.ndarray] = {}
        for c in cols:
            if c in HOT_FLOAT64_COLS:
                out[c] = np.zeros(n, dtype=np.float64)
            elif c in HOT_INT_COLS:
                out[c] = np.zeros(n, dtype=np.int64)
            else:
                out[c] = np.zeros(n, dtype=np.float32)

        for cid in np.unique(cids):
            if cid < 0:
                continue
            cid = int(cid)
            self._ensure(cid)
            if cid not in self._cache:
                continue
            idxs = np.where(cids == cid)[0]
            table = self._cache[cid]
            r = rows[idxs].astype(np.int64)
            for c in cols:
                if c in table:
                    out[c][idxs] = table[c][r].astype(out[c].dtype, copy=False)
        return out


class EmbeddingStore:
    def __init__(self, data_dir: str, max_cache: int = 64):
        self.data_dir = data_dir
        self.max_cache = max_cache
        self._cache: Dict[Tuple[int, str], np.ndarray] = {}
        self._lru: List[Tuple[int, str]] = []

    def get(self, cids: np.ndarray, rows: np.ndarray, kind: str, dim: int) -> np.ndarray:
        n = len(rows)
        out = np.zeros((n, dim), dtype=np.float32)
        for cid in np.unique(cids):
            if cid < 0:
                continue
            cid = int(cid)
            idxs = np.where(cids == cid)[0]
            key = (cid, kind)
            if key not in self._cache:
                path = os.path.join(self.data_dir, "data", f"chunk_{cid:04d}_{kind}_emb.npy")
                if not os.path.exists(path):
                    continue
                self._cache[key] = np.load(path, mmap_mode="r")
                self._lru.append(key)
                if len(self._lru) > self.max_cache:
                    old = self._lru.pop(0)
                    self._cache.pop(old, None)
            else:
                if key in self._lru:
                    self._lru.remove(key)
                self._lru.append(key)
            arr = self._cache[key]
            r = rows[idxs].astype(np.int64)
            out[idxs] = np.asarray(arr[r], dtype=np.float32)
        return out


# Dataset


class TSSSMDataset(Dataset):
    def __init__(
        self,
        shared: SharedResources,
        split: str,
        max_user_len: int,
        max_item_len: int,
        num_negatives: int,
        return_user_seen: bool,
        cfg: Config,
    ):
        self.shared = shared
        self.split = split
        self.max_user_len = max_user_len
        self.max_item_len = max_item_len
        self.num_negatives = num_negatives
        self.return_user_seen = return_user_seen
        self.cfg = cfg
        self.chunk_cache = None if shared.has_hot_data else ChunkCache(shared, max_chunks=32)
        self.emb_store = None if shared.has_hot_data else EmbeddingStore(shared.data_dir)

        valid = []
        for t in shared.splits.get(split, []):
            g = int(t)
            if g not in shared.user_for_target:
                continue
            _, pos = shared.user_for_target[g]
            if pos <= 0:
                continue
            if not shared.is_valid_global_idx(g):
                continue
            valid.append(g)
        self.target_indices = np.asarray(valid, dtype=np.int64)

        self.global_text_mean = float(shared.train_stats.get("mean_text_len", 50.0))
        self.global_image_rate = float(shared.train_stats.get("image_rate", 0.08))
        self.sample_cache = self._load_or_build_sample_cache() if cfg.CACHE_SAMPLE_FEATURES else None

    def __len__(self) -> int:
        return int(self.target_indices.shape[0])

    def _get_values(self, global_idxs: List[int], cols: List[str]) -> Dict[str, np.ndarray]:
        if self.shared.has_hot_data:
            return self.shared.get_hot_values(np.asarray(global_idxs, dtype=np.int64), cols)
        if len(global_idxs) == 0:
            out = {}
            for c in cols:
                if c in HOT_INT_COLS:
                    out[c] = np.zeros(0, dtype=np.int64)
                elif c in HOT_FLOAT64_COLS:
                    out[c] = np.zeros(0, dtype=np.float64)
                else:
                    out[c] = np.zeros(0, dtype=np.float32)
            return out
        g = np.asarray(global_idxs, dtype=np.int64)
        cids, rows = self.shared.get_loc(g)
        return self.chunk_cache.get(cids, rows, cols)

    def _get_embeddings(self, global_idxs: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.shared.has_hot_data:
            return self.shared.get_hot_embeddings(np.asarray(global_idxs, dtype=np.int64))
        if len(global_idxs) == 0:
            return (
                np.zeros((0, self.cfg.TITLE_EMB_DIM), dtype=np.float32),
                np.zeros((0, self.cfg.TEXT_EMB_DIM), dtype=np.float32),
                np.zeros((0, self.cfg.IMAGE_EMB_DIM), dtype=np.float32),
            )
        g = np.asarray(global_idxs, dtype=np.int64)
        cids, rows = self.shared.get_loc(g)
        return (
            self.emb_store.get(cids, rows, "title", self.cfg.TITLE_EMB_DIM),
            self.emb_store.get(cids, rows, "text", self.cfg.TEXT_EMB_DIM),
            self.emb_store.get(cids, rows, "image", self.cfg.IMAGE_EMB_DIM),
        )

    def _get_missing_pattern(self, vals: Dict[str, np.ndarray], idx: int) -> int:
        if "missing_pattern_tti" in vals and len(vals["missing_pattern_tti"]) > idx:
            return int(vals["missing_pattern_tti"][idx])
        if "missing_pattern" in vals and len(vals["missing_pattern"]) > idx:
            return int(vals["missing_pattern"][idx])
        return 0

    def _sample_cache_path(self) -> str:
        cache_name = (
            f"sample_cache_v{self.cfg.SAMPLE_CACHE_VERSION}_{self.split}"
            f"_u{self.max_user_len}_i{self.max_item_len}"
            f"_k{self.cfg.ITEM_NEIGHBOR_K}_seen{int(self.return_user_seen)}.pkl"
        )
        return os.path.join(self.shared.data_dir, cache_name)

    def _load_or_build_sample_cache(self) -> Dict[str, Any]:
        cache_path = self._sample_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                cached_targets = np.asarray(cache.get("target_indices", []), dtype=np.int64)
                if np.array_equal(cached_targets, self.target_indices):
                    print(f"[Dataset:{self.split}] Loaded sample cache: {cache_path}")
                    return cache
            except Exception:
                pass

        print(f"[Dataset:{self.split}] Building sample cache: {cache_path}")
        cache = self._build_sample_cache()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        return cache

    def _build_sample_cache(self) -> Dict[str, Any]:
        n = int(self.target_indices.shape[0])
        cfg = self.cfg

        user_idx_arr = np.zeros(n, dtype=np.int64)
        user_group_arr = np.zeros(n, dtype=np.int64)
        target_item_arr = np.zeros(n, dtype=np.int64)
        target_bin_arr = np.zeros(n, dtype=np.int64)
        target_rating_arr = np.zeros(n, dtype=np.float32)
        target_has_title_arr = np.zeros(n, dtype=np.float32)
        target_has_text_arr = np.zeros(n, dtype=np.float32)
        target_has_image_arr = np.zeros(n, dtype=np.float32)
        target_missing_pattern_arr = np.zeros(n, dtype=np.int64)
        item_group_arr = np.zeros(n, dtype=np.int64)

        global_ot_arr = np.zeros((n, cfg.O_T_DIM), dtype=np.float32)
        group_ot_arr = np.zeros((n, cfg.O_T_DIM), dtype=np.float32)
        global_ot_delta_arr = np.zeros((n, cfg.O_T_DIM), dtype=np.float32)
        group_ot_delta_arr = np.zeros((n, cfg.O_T_DIM), dtype=np.float32)
        item_global_ot_arr = np.zeros((n, cfg.O_T_DIM), dtype=np.float32)
        item_group_ot_arr = np.zeros((n, cfg.O_T_DIM), dtype=np.float32)
        item_global_ot_delta_arr = np.zeros((n, cfg.O_T_DIM), dtype=np.float32)
        item_group_ot_delta_arr = np.zeros((n, cfg.O_T_DIM), dtype=np.float32)

        deviation_arr = np.zeros((n, cfg.DEVIATION_DIM), dtype=np.float32)
        item_deviation_arr = np.zeros((n, cfg.ITEM_DEVIATION_DIM), dtype=np.float32)
        shock_arr = np.zeros((n, cfg.SHOCK_FEAT_DIM), dtype=np.float32)
        reliability_arr = np.zeros((n, cfg.RELIABILITY_FEAT_DIM), dtype=np.float32)
        future_targets_arr = np.zeros((n, 4), dtype=np.float32)
        future_mask_arr = np.zeros(n, dtype=np.bool_)
        carry_targets_arr = np.zeros((n, 2), dtype=np.float32)
        carry_mask_arr = np.zeros(n, dtype=np.bool_)
        anchor_item_arr = np.zeros(n, dtype=np.int64)
        anchor_neighbors_arr = np.zeros((n, cfg.ITEM_NEIGHBOR_K), dtype=np.int64)

        user_hist_offsets = np.zeros(n + 1, dtype=np.int64)
        item_hist_offsets = np.zeros(n + 1, dtype=np.int64)
        user_seen_offsets = np.zeros(n + 1, dtype=np.int64)
        user_hist_parts: List[np.ndarray] = []
        item_hist_parts: List[np.ndarray] = []
        user_seen_parts: List[np.ndarray] = []

        for idx, target_g in enumerate(tqdm(self.target_indices, desc=f"Building sample cache [{self.split}]", leave=False)):
            target_g = int(target_g)
            user_idx, pos_in_user = self.shared.user_for_target[target_g]
            user_idx_arr[idx] = int(user_idx)

            target_vals = self._get_values([target_g], HOT_DATA_COLS)
            tdict = {k: float(v[0]) if len(v) > 0 else 0.0 for k, v in target_vals.items()}
            target_item = int(tdict.get("item_idx", 0))
            target_bin = int(tdict.get("bin_id", 0))
            target_rating = float(tdict.get("rating", 4.0))
            target_unix = float(tdict.get("unix_time", 0.0))

            user_seq = self.shared.user_histories.get(user_idx, [])
            start = max(0, pos_in_user - self.max_user_len)
            user_hist_g = np.asarray(
                [int(g) for g in user_seq[start:pos_in_user] if self.shared.is_valid_global_idx(int(g))],
                dtype=np.int64,
            )
            item_hist_g = self.shared.get_item_history_before(target_item, target_unix, target_g, self.max_item_len).astype(np.int64, copy=False)

            user_hist_offsets[idx + 1] = user_hist_offsets[idx] + user_hist_g.size
            item_hist_offsets[idx + 1] = item_hist_offsets[idx] + item_hist_g.size
            user_hist_parts.append(user_hist_g)
            item_hist_parts.append(item_hist_g)

            user_group = self.shared.get_user_group(user_idx)
            item_group = self.shared.get_item_group(target_item)

            user_group_arr[idx] = int(user_group)
            target_item_arr[idx] = int(target_item)
            target_bin_arr[idx] = int(target_bin)
            target_rating_arr[idx] = np.float32(target_rating)
            target_has_title_arr[idx] = np.float32(tdict.get("has_title", 1.0))
            target_has_text_arr[idx] = np.float32(tdict.get("has_text", 1.0))
            target_has_image_arr[idx] = np.float32(tdict.get("has_image", 0.0))
            target_missing_pattern_arr[idx] = np.int64(self._get_missing_pattern(target_vals, 0))
            item_group_arr[idx] = np.int64(item_group)

            global_ot_arr[idx] = self.shared.get_global_ot(target_bin).astype(np.float32)
            group_ot_arr[idx] = self.shared.get_group_ot(user_group, target_bin).astype(np.float32)
            global_ot_delta_arr[idx] = self.shared.get_global_ot_delta(target_bin).astype(np.float32)
            group_ot_delta_arr[idx] = self.shared.get_group_ot_delta(user_group, target_bin).astype(np.float32)
            item_global_ot_arr[idx] = self.shared.get_item_global_ot(target_bin).astype(np.float32)
            item_group_ot_arr[idx] = self.shared.get_item_group_ot(item_group, target_bin).astype(np.float32)
            item_global_ot_delta_arr[idx] = self.shared.get_item_global_ot_delta(target_bin).astype(np.float32)
            item_group_ot_delta_arr[idx] = self.shared.get_item_group_ot_delta(item_group, target_bin).astype(np.float32)

            if user_hist_g.size > 0:
                hvals = self._get_values(user_hist_g.tolist(), HOT_DATA_COLS)
                u_items = hvals["item_idx"].astype(np.int64)
                u_dev = self._build_user_deviation(hvals, tdict)
            else:
                u_items = np.zeros(0, dtype=np.int64)
                u_dev = np.zeros(self.cfg.DEVIATION_DIM, dtype=np.float32)

            if item_hist_g.size > 0:
                ivals = self._get_values(item_hist_g.tolist(), HOT_DATA_COLS)
                item_dev = self._build_item_deviation(ivals, tdict)
            else:
                item_dev = self._build_item_deviation({}, tdict)

            deviation_arr[idx] = u_dev.astype(np.float32)
            item_deviation_arr[idx] = item_dev.astype(np.float32)

            target_missing_pattern = int(target_missing_pattern_arr[idx])
            shock_feat = np.zeros(self.cfg.SHOCK_FEAT_DIM, dtype=np.float32)
            shock_feat[0] = np.clip((target_rating - float(tdict.get("i_prev_mean_rating", 4.0))) / 2.0, -2.0, 2.0)
            shock_feat[1] = float(tdict.get("has_image", 0.0))
            shock_feat[2] = float(tdict.get("has_text", 1.0))
            shock_feat[3] = np.clip(float(tdict.get("len_text_tokens", 50.0)) / 200.0, 0.0, 4.0)
            shock_feat[4] = np.log1p(float(tdict.get("helpful_vote", 0.0))) / 5.0
            shock_feat[5] = np.log1p(float(tdict.get("num_images", 0.0))) / 2.0
            shock_feat[6] = np.clip(float(tdict.get("u_rating_delta", 0.0)) / 2.0, -2.0, 2.0)
            shock_feat[7] = np.clip(float(target_missing_pattern) / max(1, self.cfg.NUM_MISSING_PATTERNS - 1), 0.0, 1.0)
            shock_arr[idx] = shock_feat

            obs_tgt = float((tdict.get("has_title", 1.0) + tdict.get("has_text", 1.0) + tdict.get("has_image", 0.0)) / 3.0)
            rel_tgt = float(np.clip(0.5 * u_dev[4] + 0.5 * item_dev[4], 0.0, 1.0))
            conflict_tgt = 1.0 if (u_dev[3] * item_dev[0] < 0) else 0.0
            reliability_arr[idx] = np.asarray([obs_tgt, rel_tgt, conflict_tgt], dtype=np.float32)

            future_target, future_mask = self.shared.get_item_future_target(target_item, target_bin)
            future_targets_arr[idx] = future_target.astype(np.float32)
            future_mask_arr[idx] = np.bool_(future_mask)
            carry_mask_arr[idx] = np.bool_(future_mask)
            if future_mask:
                carry_targets_arr[idx, 0] = future_target[0] - float(tdict.get("i_prev_mean_rating", 4.0)) / 5.0
                carry_targets_arr[idx, 1] = future_target[3] - np.log1p(float(tdict.get("i_num_prev_reviews", 0.0))) / 5.0

            anchor_item = int(u_items[-1]) if u_items.size > 0 else int(target_item)
            anchor_item_arr[idx] = np.int64(anchor_item)
            anchor_neighbors_arr[idx] = self.shared.get_item_neighbors(anchor_item, self.cfg.ITEM_NEIGHBOR_K).astype(np.int64)

            if self.return_user_seen:
                user_seen = self.shared.get_user_seen_items_fast(user_idx, pos_in_user)
                if user_seen.size == 0 and not self.shared._use_fast_seen:
                    past = np.asarray([int(g) for g in user_seq[:pos_in_user] if self.shared.is_valid_global_idx(int(g))], dtype=np.int64)
                    if past.size > 0:
                        user_seen = self._get_values(past.tolist(), ["item_idx"])["item_idx"].astype(np.int64)
                    else:
                        user_seen = np.zeros(0, dtype=np.int64)
                user_seen_offsets[idx + 1] = user_seen_offsets[idx] + user_seen.size
                user_seen_parts.append(user_seen.astype(np.int64, copy=False))
            else:
                user_seen_offsets[idx + 1] = user_seen_offsets[idx]

        return {
            "target_indices": self.target_indices.copy(),
            "user_idx": user_idx_arr,
            "user_group": user_group_arr,
            "target_item": target_item_arr,
            "target_bin": target_bin_arr,
            "target_rating": target_rating_arr,
            "target_has_title": target_has_title_arr,
            "target_has_text": target_has_text_arr,
            "target_has_image": target_has_image_arr,
            "target_missing_pattern": target_missing_pattern_arr,
            "item_group": item_group_arr,
            "global_ot": global_ot_arr,
            "group_ot": group_ot_arr,
            "global_ot_delta": global_ot_delta_arr,
            "group_ot_delta": group_ot_delta_arr,
            "item_global_ot": item_global_ot_arr,
            "item_group_ot": item_group_ot_arr,
            "item_global_ot_delta": item_global_ot_delta_arr,
            "item_group_ot_delta": item_group_ot_delta_arr,
            "deviation": deviation_arr,
            "item_deviation": item_deviation_arr,
            "shock_features": shock_arr,
            "reliability_targets": reliability_arr,
            "item_future_targets": future_targets_arr,
            "item_future_mask": future_mask_arr,
            "carry_targets": carry_targets_arr,
            "carry_mask": carry_mask_arr,
            "anchor_item": anchor_item_arr,
            "anchor_neighbors": anchor_neighbors_arr,
            "user_hist_offsets": user_hist_offsets,
            "user_hist_flat": np.concatenate(user_hist_parts) if user_hist_parts else np.zeros(0, dtype=np.int64),
            "item_hist_offsets": item_hist_offsets,
            "item_hist_flat": np.concatenate(item_hist_parts) if item_hist_parts else np.zeros(0, dtype=np.int64),
            "user_seen_offsets": user_seen_offsets,
            "user_seen_flat": np.concatenate(user_seen_parts) if user_seen_parts else np.zeros(0, dtype=np.int64),
        }

    def _build_user_numeric_features(self, vals: Dict[str, np.ndarray], target_unix: float) -> Tuple[np.ndarray, np.ndarray]:
        L = len(next(iter(vals.values()))) if vals else 0
        x = np.zeros((L, self.cfg.USER_NUM_FEAT_DIM), dtype=np.float32)
        ts = self.shared.train_stats

        def get(name: str, default: float):
            return vals.get(name, np.full(L, default, dtype=np.float32))

        x[:, 0] = get("rating", 4.0) / 5.0
        x[:, 1] = get("has_title", 1.0)
        x[:, 2] = get("has_text", 1.0)
        x[:, 3] = get("has_image", 0.0)
        x[:, 4] = get("len_title_tokens", 5.0) / 50.0
        x[:, 5] = get("len_text_tokens", 50.0) / 200.0
        x[:, 6] = get("u_num_prev_reviews", 0.0) / 50.0
        x[:, 7] = get("u_prev_mean_rating", ts.get("mean_rating", 4.0)) / 5.0
        x[:, 8] = get("u_prev_image_rate", ts.get("image_rate", 0.08))
        x[:, 9] = get("u_prev_mean_text_len", ts.get("mean_text_len", 50.0)) / 100.0
        x[:, 10] = get("u_is_first_review", 0.0)
        x[:, 11] = get("u_time_since_prev", 0.0) / 5.0
        x[:, 12] = get("i_num_prev_reviews", 0.0) / 50.0
        x[:, 13] = get("i_prev_mean_rating", ts.get("mean_rating", 4.0)) / 5.0
        x[:, 14] = get("i_prev_rating_std", 1.0) / 2.0
        x[:, 15] = get("i_prev_image_rate", ts.get("image_rate", 0.08))
        x[:, 16] = get("i_prev_mean_text_len", ts.get("mean_text_len", 50.0)) / 100.0
        x[:, 17] = get("i_is_first_review", 0.0)
        np.nan_to_num(x, copy=False, nan=0.0)

        hist_unix = vals.get("unix_time", np.full(L, target_unix, dtype=np.float64))
        age_seconds = np.maximum(target_unix - hist_unix, 0.0)
        age_days = age_seconds / 86400.0
        time_age = np.log1p(age_days).astype(np.float32)
        return x, time_age

    def _build_item_numeric_features(self, vals: Dict[str, np.ndarray], target_unix: float) -> Tuple[np.ndarray, np.ndarray]:
        L = len(next(iter(vals.values()))) if vals else 0
        x = np.zeros((L, self.cfg.ITEM_NUM_FEAT_DIM), dtype=np.float32)

        def get(name: str, default: float):
            return vals.get(name, np.full(L, default, dtype=np.float32))

        x[:, 0] = get("rating", 4.0) / 5.0
        x[:, 1] = get("has_title", 1.0)
        x[:, 2] = get("has_text", 1.0)
        x[:, 3] = get("has_image", 0.0)
        x[:, 4] = get("len_title_tokens", 5.0) / 50.0
        x[:, 5] = get("len_text_tokens", 50.0) / 200.0
        x[:, 6] = np.log1p(get("helpful_vote", 0.0)) / 5.0
        x[:, 7] = np.log1p(get("num_images", 0.0)) / 2.0
        x[:, 8] = get("verified_purchase", 0.0)
        x[:, 9] = np.clip(get("u_rating_delta", 0.0) / 2.0, -1.5, 1.5)
        x[:, 10] = np.clip(get("i_rating_delta", 0.0) / 2.0, -1.5, 1.5)
        x[:, 11] = np.clip(np.log1p(get("i_time_since_prev", 0.0)) / 10.0, 0.0, 1.0)
        np.nan_to_num(x, copy=False, nan=0.0)

        hist_unix = vals.get("unix_time", np.full(L, target_unix, dtype=np.float64))
        age_seconds = np.maximum(target_unix - hist_unix, 0.0)
        age_days = age_seconds / 86400.0
        time_age = np.log1p(age_days).astype(np.float32)
        return x, time_age

    def _build_user_deviation(self, hist_vals: Dict[str, np.ndarray], target_vals: Dict[str, float]) -> np.ndarray:
        dev = np.zeros(self.cfg.DEVIATION_DIM, dtype=np.float32)
        L = len(hist_vals.get("has_image", []))
        if L < 2:
            dev[6] = float(target_vals.get("u_is_image_unusual", 0.0))
            dev[7] = float(target_vals.get("u_is_text_long_unusual", 0.0))
            dev[8] = float(target_vals.get("u_is_title_long_unusual", 0.0))
            dev[9] = np.clip(float(target_vals.get("u_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
            dev[10] = np.clip(float(target_vals.get("i_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
            dev[11] = np.log1p(float(target_vals.get("num_images", 0.0))) / 2.0
            dev[12] = np.log1p(float(target_vals.get("helpful_vote", 0.0))) / 5.0
            return dev

        last_has_img = float(hist_vals["has_image"][-1])
        last_tlen = float(hist_vals["len_text_tokens"][-1])
        last_hlen = float(hist_vals["len_title_tokens"][-1])
        last_rating = float(hist_vals["rating"][-1])
        prev_img_rate = float(hist_vals["has_image"][:-1].mean())
        prev_tmean = float(hist_vals["len_text_tokens"][:-1].mean())
        prev_hmean = float(hist_vals["len_title_tokens"][:-1].mean())
        prev_rmean = float(hist_vals["rating"][:-1].mean())

        dev[0] = last_has_img - prev_img_rate
        dev[1] = np.clip((last_tlen - prev_tmean) / max(prev_tmean * 0.5, 15.0), -3.0, 3.0)
        dev[2] = np.clip((last_hlen - prev_hmean) / max(prev_hmean * 0.5, 2.0), -3.0, 3.0)
        dev[3] = (last_rating - prev_rmean) / 2.0
        dev[4] = min(L / 20.0, 1.0)
        dev[5] = abs(dev[0])
        dev[6] = float(target_vals.get("u_is_image_unusual", 0.0))
        dev[7] = float(target_vals.get("u_is_text_long_unusual", 0.0))
        dev[8] = float(target_vals.get("u_is_title_long_unusual", 0.0))
        dev[9] = np.clip(float(target_vals.get("u_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
        dev[10] = np.clip(float(target_vals.get("i_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
        dev[11] = np.log1p(float(target_vals.get("num_images", 0.0))) / 2.0
        dev[12] = np.log1p(float(target_vals.get("helpful_vote", 0.0))) / 5.0
        dev[13] = 1.0 if (prev_img_rate < 0.05 and last_has_img > 0.5) else 0.0
        dev[14] = np.clip(np.log1p(float(target_vals.get("u_time_since_prev", 0.0))) / 10.0, 0.0, 1.0)
        dev[15] = np.clip(np.log1p(float(target_vals.get("i_time_since_prev", 0.0))) / 10.0, 0.0, 1.0)
        return dev

    def _build_item_deviation(self, item_hist_vals: Dict[str, np.ndarray], target_vals: Dict[str, float]) -> np.ndarray:
        dev = np.zeros(self.cfg.ITEM_DEVIATION_DIM, dtype=np.float32)
        L = len(item_hist_vals.get("rating", []))
        if L < 2:
            dev[4] = min(L / 20.0, 1.0)
            dev[6] = np.clip(float(target_vals.get("i_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
            dev[7] = np.clip(float(target_vals.get("u_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
            dev[8] = np.log1p(float(target_vals.get("num_images", 0.0))) / 2.0
            dev[9] = np.log1p(float(target_vals.get("helpful_vote", 0.0))) / 5.0
            dev[10] = float(target_vals.get("has_image", 0.0))
            dev[11] = float(target_vals.get("has_text", 1.0))
            dev[12] = float(target_vals.get("has_title", 1.0))
            return dev

        hist_r = item_hist_vals["rating"].astype(np.float32)
        hist_img = item_hist_vals["has_image"].astype(np.float32)
        hist_tlen = item_hist_vals["len_text_tokens"].astype(np.float32)
        last_r = float(hist_r[-1])
        prev_mean = float(hist_r[:-1].mean())
        prev_std = float(hist_r[:-1].std() + 1e-6)
        dev[0] = (last_r - prev_mean) / 2.0
        dev[1] = np.clip(prev_std / 2.0, 0.0, 2.0)
        dev[2] = float(hist_img[-1] - hist_img[:-1].mean())
        dev[3] = np.clip((float(hist_tlen[-1]) - float(hist_tlen[:-1].mean())) / max(float(hist_tlen[:-1].mean()) * 0.5, 15.0), -3.0, 3.0)
        dev[4] = min(L / 30.0, 1.0)
        dev[5] = abs(dev[0])
        dev[6] = np.clip(float(target_vals.get("i_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
        dev[7] = np.clip(float(target_vals.get("u_rating_delta", 0.0)) / 2.0, -1.5, 1.5)
        dev[8] = np.log1p(float(target_vals.get("num_images", 0.0))) / 2.0
        dev[9] = np.log1p(float(target_vals.get("helpful_vote", 0.0))) / 5.0
        dev[10] = float(target_vals.get("has_image", 0.0))
        dev[11] = float(target_vals.get("has_text", 1.0))
        dev[12] = float(target_vals.get("has_title", 1.0))
        dev[13] = 1.0 if (hist_img[:-1].mean() < 0.05 and hist_img[-1] > 0.5) else 0.0
        dev[14] = np.clip(np.log1p(float(target_vals.get("u_time_since_prev", 0.0))) / 10.0, 0.0, 1.0)
        dev[15] = np.clip(np.log1p(float(target_vals.get("i_time_since_prev", 0.0))) / 10.0, 0.0, 1.0)
        return dev

    def _sample_negatives(self, user_idx: int, target_item: int) -> Tuple[np.ndarray, np.ndarray]:
        num_neg = self.num_negatives
        negatives = np.full(num_neg, self.shared.pad_idx, dtype=np.int64)
        valid = np.zeros(num_neg, dtype=np.bool_)
        if num_neg <= 0:
            return negatives, valid

        used = set(self.shared.get_user_seen_np(user_idx).tolist())
        used.add(int(target_item))
        used.add(int(self.shared.pad_idx))

        num_items = self.shared.num_items
        sample_min = 1 if self.shared.pad_idx == 0 else 0

        if self.cfg.USE_MIXED_NEG_SAMPLING:
            n_uni = int(round(num_neg * self.cfg.UNIFORM_NEG_RATIO))
            n_uni = max(0, min(num_neg, n_uni))
            n_pop = num_neg - n_uni
        else:
            n_uni = num_neg
            n_pop = 0

        c = 0
        for r in range(4):
            need = n_uni - c
            if need <= 0:
                break
            cand = np.random.randint(sample_min, num_items, size=need * (30 if r == 0 else 60))
            for it in np.unique(cand).tolist():
                if it in used:
                    continue
                negatives[c] = it
                valid[c] = True
                used.add(it)
                c += 1
                if c >= n_uni:
                    break

        if n_pop > 0:
            probs = self.shared.item_pop_probs.copy()
            if len(used) > 0:
                idx = np.fromiter(used, dtype=np.int64, count=len(used))
                idx = idx[(idx >= 0) & (idx < num_items)]
                probs[idx] = 0.0
            s = float(probs.sum())
            if s > 1e-8:
                probs = probs / s
                cand = np.random.choice(num_items, size=n_pop * 12, replace=True, p=probs)
                target = n_uni + n_pop
                for it in cand.tolist():
                    if it in used:
                        continue
                    negatives[c] = it
                    valid[c] = True
                    used.add(it)
                    c += 1
                    if c >= target:
                        break
        return negatives, valid

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache = self.sample_cache
        if cache is None:
            raise RuntimeError("Sample cache must be enabled for optimized dataset path")

        target_g = int(cache["target_indices"][idx])
        user_idx = int(cache["user_idx"][idx])
        target_item = int(cache["target_item"][idx])
        target_bin = int(cache["target_bin"][idx])
        target_rating = float(cache["target_rating"][idx])

        if target_item == self.shared.pad_idx:
            raise RuntimeError(f"target_item equals pad_idx at global_idx={target_g}")

        u_start = int(cache["user_hist_offsets"][idx])
        u_end = int(cache["user_hist_offsets"][idx + 1])
        user_hist_g = cache["user_hist_flat"][u_start:u_end]

        i_start = int(cache["item_hist_offsets"][idx])
        i_end = int(cache["item_hist_offsets"][idx + 1])
        item_hist_g = cache["item_hist_flat"][i_start:i_end]

        target_vals = self._get_values([target_g], HOT_DATA_COLS)
        tdict = {k: float(v[0]) if len(v) > 0 else 0.0 for k, v in target_vals.items()}
        target_unix = float(tdict.get("unix_time", 0.0))

        if user_hist_g.size > 0:
            hvals = self._get_values(user_hist_g.tolist(), HOT_DATA_COLS)
            u_items = hvals["item_idx"].astype(np.int64)
            u_num, u_age = self._build_user_numeric_features(hvals, target_unix)
            u_title, u_text, u_image = self._get_embeddings(user_hist_g.tolist())
            if "missing_pattern_tti" in hvals:
                u_mp = hvals["missing_pattern_tti"].astype(np.int64)
            elif "missing_pattern" in hvals:
                u_mp = hvals["missing_pattern"].astype(np.int64)
            else:
                u_mp = np.zeros(user_hist_g.size, dtype=np.int64)
            u_bins = hvals.get("bin_id", np.zeros(user_hist_g.size, dtype=np.int64)).astype(np.int64)
        else:
            u_items = np.zeros(0, dtype=np.int64)
            u_num = np.zeros((0, self.cfg.USER_NUM_FEAT_DIM), dtype=np.float32)
            u_age = np.zeros(0, dtype=np.float32)
            u_title = np.zeros((0, self.cfg.TITLE_EMB_DIM), dtype=np.float32)
            u_text = np.zeros((0, self.cfg.TEXT_EMB_DIM), dtype=np.float32)
            u_image = np.zeros((0, self.cfg.IMAGE_EMB_DIM), dtype=np.float32)
            u_mp = np.zeros(0, dtype=np.int64)
            u_bins = np.zeros(0, dtype=np.int64)

        if item_hist_g.size > 0:
            ivals = self._get_values(item_hist_g.tolist(), HOT_DATA_COLS)
            i_items = ivals["item_idx"].astype(np.int64)
            i_num, i_age = self._build_item_numeric_features(ivals, target_unix)
            i_title, i_text, i_image = self._get_embeddings(item_hist_g.tolist())
            if "missing_pattern_tti" in ivals:
                i_mp = ivals["missing_pattern_tti"].astype(np.int64)
            elif "missing_pattern" in ivals:
                i_mp = ivals["missing_pattern"].astype(np.int64)
            else:
                i_mp = np.zeros(item_hist_g.size, dtype=np.int64)
            i_bins = ivals.get("bin_id", np.zeros(item_hist_g.size, dtype=np.int64)).astype(np.int64)
        else:
            i_items = np.zeros(0, dtype=np.int64)
            i_num = np.zeros((0, self.cfg.ITEM_NUM_FEAT_DIM), dtype=np.float32)
            i_age = np.zeros(0, dtype=np.float32)
            i_title = np.zeros((0, self.cfg.TITLE_EMB_DIM), dtype=np.float32)
            i_text = np.zeros((0, self.cfg.TEXT_EMB_DIM), dtype=np.float32)
            i_image = np.zeros((0, self.cfg.IMAGE_EMB_DIM), dtype=np.float32)
            i_mp = np.zeros(0, dtype=np.int64)
            i_bins = np.zeros(0, dtype=np.int64)

        tt, tx, ti = self._get_embeddings([target_g])
        target_title = tt[0] if len(tt) > 0 else np.zeros(self.cfg.TITLE_EMB_DIM, dtype=np.float32)
        target_text = tx[0] if len(tx) > 0 else np.zeros(self.cfg.TEXT_EMB_DIM, dtype=np.float32)
        target_image = ti[0] if len(ti) > 0 else np.zeros(self.cfg.IMAGE_EMB_DIM, dtype=np.float32)

        if self.split == "train" and self.num_negatives > 0:
            negatives, neg_valid = self._sample_negatives(user_idx, target_item)
        else:
            negatives = np.zeros(0, dtype=np.int64)
            neg_valid = np.zeros(0, dtype=np.bool_)

        s_start = int(cache["user_seen_offsets"][idx])
        s_end = int(cache["user_seen_offsets"][idx + 1])
        user_seen = cache["user_seen_flat"][s_start:s_end] if self.return_user_seen else np.zeros(0, dtype=np.int64)

        return {
            "pad_idx": self.shared.pad_idx,
            "user_idx": user_idx,
            "user_group": int(cache["user_group"][idx]),
            "target_item": target_item,
            "target_bin": target_bin,
            "target_rating": np.float32(target_rating),
            "target_has_title": np.float32(cache["target_has_title"][idx]),
            "target_has_text": np.float32(cache["target_has_text"][idx]),
            "target_has_image": np.float32(cache["target_has_image"][idx]),
            "target_title_emb": target_title.astype(np.float32),
            "target_text_emb": target_text.astype(np.float32),
            "target_image_emb": target_image.astype(np.float32),
            "target_missing_pattern": np.int64(cache["target_missing_pattern"][idx]),
            "global_ot": cache["global_ot"][idx].astype(np.float32),
            "group_ot": cache["group_ot"][idx].astype(np.float32),
            "global_ot_delta": cache["global_ot_delta"][idx].astype(np.float32),
            "group_ot_delta": cache["group_ot_delta"][idx].astype(np.float32),
            "deviation": cache["deviation"][idx].astype(np.float32),
            "user_hist_len": int(user_hist_g.size),
            "user_hist_items": u_items,
            "user_hist_num": u_num,
            "user_hist_time_age": u_age,
            "user_hist_title_emb": u_title,
            "user_hist_text_emb": u_text,
            "user_hist_image_emb": u_image,
            "user_hist_missing_pattern": u_mp,
            "user_hist_bin_ids": u_bins,
            "item_group": np.int64(cache["item_group"][idx]),
            "item_global_ot": cache["item_global_ot"][idx].astype(np.float32),
            "item_group_ot": cache["item_group_ot"][idx].astype(np.float32),
            "item_global_ot_delta": cache["item_global_ot_delta"][idx].astype(np.float32),
            "item_group_ot_delta": cache["item_group_ot_delta"][idx].astype(np.float32),
            "item_deviation": cache["item_deviation"][idx].astype(np.float32),
            "item_hist_len": int(item_hist_g.size),
            "item_hist_items": i_items,
            "item_hist_num": i_num,
            "item_hist_time_age": i_age,
            "item_hist_title_emb": i_title,
            "item_hist_text_emb": i_text,
            "item_hist_image_emb": i_image,
            "item_hist_missing_pattern": i_mp,
            "item_hist_bin_ids": i_bins,
            "shock_features": cache["shock_features"][idx].astype(np.float32),
            "reliability_targets": cache["reliability_targets"][idx].astype(np.float32),
            "item_future_targets": cache["item_future_targets"][idx].astype(np.float32),
            "item_future_mask": np.bool_(cache["item_future_mask"][idx]),
            "carry_targets": cache["carry_targets"][idx].astype(np.float32),
            "carry_mask": np.bool_(cache["carry_mask"][idx]),
            "anchor_item": np.int64(cache["anchor_item"][idx]),
            "anchor_neighbors": cache["anchor_neighbors"][idx].astype(np.int64),
            "negative_items": negatives,
            "neg_valid": neg_valid,
            "user_seen_items": user_seen.astype(np.int64),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch = [b for b in batch if b is not None]
    B = len(batch)
    if B == 0:
        raise ValueError("Empty batch")

    pad_idx = int(batch[0].get("pad_idx", 0))
    title_dim = int(batch[0]["target_title_emb"].shape[0])
    text_dim = int(batch[0]["target_text_emb"].shape[0])
    image_dim = int(batch[0]["target_image_emb"].shape[0])
    user_num_dim = int(batch[0]["user_hist_num"].shape[1]) if batch[0]["user_hist_num"].ndim == 2 and batch[0]["user_hist_num"].shape[0] > 0 else 18
    item_num_dim = int(batch[0]["item_hist_num"].shape[1]) if batch[0]["item_hist_num"].ndim == 2 and batch[0]["item_hist_num"].shape[0] > 0 else 12

    out: Dict[str, torch.Tensor] = {}
    keys_long = [
        "user_idx",
        "user_group",
        "target_item",
        "target_bin",
        "target_missing_pattern",
        "item_group",
        "anchor_item",
    ]
    for k in keys_long:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)

    keys_float = [
        "target_rating",
        "target_has_title",
        "target_has_text",
        "target_has_image",
    ]
    for k in keys_float:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)

    out["target_title_emb"] = torch.from_numpy(np.stack([b["target_title_emb"] for b in batch]))
    out["target_text_emb"] = torch.from_numpy(np.stack([b["target_text_emb"] for b in batch]))
    out["target_image_emb"] = torch.from_numpy(np.stack([b["target_image_emb"] for b in batch]))

    for k in ["global_ot", "group_ot", "global_ot_delta", "group_ot_delta", "item_global_ot", "item_group_ot", "item_global_ot_delta", "item_group_ot_delta"]:
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))

    out["deviation"] = torch.from_numpy(np.stack([b["deviation"] for b in batch]))
    out["item_deviation"] = torch.from_numpy(np.stack([b["item_deviation"] for b in batch]))
    out["shock_features"] = torch.from_numpy(np.stack([b["shock_features"] for b in batch]))
    out["reliability_targets"] = torch.from_numpy(np.stack([b["reliability_targets"] for b in batch]))
    out["item_future_targets"] = torch.from_numpy(np.stack([b["item_future_targets"] for b in batch]))
    out["item_future_mask"] = torch.tensor([bool(b["item_future_mask"]) for b in batch], dtype=torch.bool)
    out["carry_targets"] = torch.from_numpy(np.stack([b["carry_targets"] for b in batch]))
    out["carry_mask"] = torch.tensor([bool(b["carry_mask"]) for b in batch], dtype=torch.bool)
    out["anchor_neighbors"] = torch.from_numpy(np.stack([b["anchor_neighbors"] for b in batch]))

    # User history padding
    user_lens = [int(b["user_hist_len"]) for b in batch]
    Lu = max(1, max(user_lens))
    out["user_hist_len"] = torch.tensor(user_lens, dtype=torch.long)
    out["user_hist_items"] = torch.full((B, Lu), pad_idx, dtype=torch.long)
    out["user_hist_num"] = torch.zeros(B, Lu, user_num_dim, dtype=torch.float32)
    out["user_hist_time_age"] = torch.zeros(B, Lu, dtype=torch.float32)
    out["user_hist_mask"] = torch.zeros(B, Lu, dtype=torch.bool)
    out["user_hist_title_emb"] = torch.zeros(B, Lu, title_dim, dtype=torch.float32)
    out["user_hist_text_emb"] = torch.zeros(B, Lu, text_dim, dtype=torch.float32)
    out["user_hist_image_emb"] = torch.zeros(B, Lu, image_dim, dtype=torch.float32)
    out["user_hist_missing_pattern"] = torch.zeros(B, Lu, dtype=torch.long)
    out["user_hist_bin_ids"] = torch.zeros(B, Lu, dtype=torch.long)

    for i, b in enumerate(batch):
        l = int(b["user_hist_len"])
        if l <= 0:
            continue
        out["user_hist_items"][i, :l] = torch.from_numpy(b["user_hist_items"][:l])
        out["user_hist_num"][i, :l] = torch.from_numpy(b["user_hist_num"][:l])
        out["user_hist_time_age"][i, :l] = torch.from_numpy(b["user_hist_time_age"][:l])
        out["user_hist_mask"][i, :l] = True
        out["user_hist_title_emb"][i, :l] = torch.from_numpy(b["user_hist_title_emb"][:l])
        out["user_hist_text_emb"][i, :l] = torch.from_numpy(b["user_hist_text_emb"][:l])
        out["user_hist_image_emb"][i, :l] = torch.from_numpy(b["user_hist_image_emb"][:l])
        if len(b["user_hist_missing_pattern"]) > 0:
            out["user_hist_missing_pattern"][i, :l] = torch.from_numpy(b["user_hist_missing_pattern"][:l])
        if len(b["user_hist_bin_ids"]) > 0:
            out["user_hist_bin_ids"][i, :l] = torch.from_numpy(b["user_hist_bin_ids"][:l])

    # Item history padding
    item_lens = [int(b["item_hist_len"]) for b in batch]
    Li = max(1, max(item_lens))
    out["item_hist_len"] = torch.tensor(item_lens, dtype=torch.long)
    out["item_hist_items"] = torch.full((B, Li), pad_idx, dtype=torch.long)
    out["item_hist_num"] = torch.zeros(B, Li, item_num_dim, dtype=torch.float32)
    out["item_hist_time_age"] = torch.zeros(B, Li, dtype=torch.float32)
    out["item_hist_mask"] = torch.zeros(B, Li, dtype=torch.bool)
    out["item_hist_title_emb"] = torch.zeros(B, Li, title_dim, dtype=torch.float32)
    out["item_hist_text_emb"] = torch.zeros(B, Li, text_dim, dtype=torch.float32)
    out["item_hist_image_emb"] = torch.zeros(B, Li, image_dim, dtype=torch.float32)
    out["item_hist_missing_pattern"] = torch.zeros(B, Li, dtype=torch.long)
    out["item_hist_bin_ids"] = torch.zeros(B, Li, dtype=torch.long)

    for i, b in enumerate(batch):
        l = int(b["item_hist_len"])
        if l <= 0:
            continue
        out["item_hist_items"][i, :l] = torch.from_numpy(b["item_hist_items"][:l])
        out["item_hist_num"][i, :l] = torch.from_numpy(b["item_hist_num"][:l])
        out["item_hist_time_age"][i, :l] = torch.from_numpy(b["item_hist_time_age"][:l])
        out["item_hist_mask"][i, :l] = True
        out["item_hist_title_emb"][i, :l] = torch.from_numpy(b["item_hist_title_emb"][:l])
        out["item_hist_text_emb"][i, :l] = torch.from_numpy(b["item_hist_text_emb"][:l])
        out["item_hist_image_emb"][i, :l] = torch.from_numpy(b["item_hist_image_emb"][:l])
        if len(b["item_hist_missing_pattern"]) > 0:
            out["item_hist_missing_pattern"][i, :l] = torch.from_numpy(b["item_hist_missing_pattern"][:l])
        if len(b["item_hist_bin_ids"]) > 0:
            out["item_hist_bin_ids"][i, :l] = torch.from_numpy(b["item_hist_bin_ids"][:l])

    # Negatives
    max_negs = max((b["negative_items"].shape[0] for b in batch), default=0)
    if max_negs > 0:
        out["negative_items"] = torch.full((B, max_negs), pad_idx, dtype=torch.long)
        out["neg_valid_mask"] = torch.zeros(B, max_negs, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = int(b["negative_items"].shape[0])
            if n <= 0:
                continue
            out["negative_items"][i, :n] = torch.from_numpy(b["negative_items"][:n])
            out["neg_valid_mask"][i, :n] = torch.from_numpy(b["neg_valid"][:n])
    else:
        out["negative_items"] = torch.empty(B, 0, dtype=torch.long)
        out["neg_valid_mask"] = torch.empty(B, 0, dtype=torch.bool)

    # User seen
    max_seen = min(5000, max((b["user_seen_items"].shape[0] for b in batch), default=0))
    out["user_seen_items"] = torch.full((B, max(1, max_seen)), pad_idx, dtype=torch.long)
    out["user_seen_mask"] = torch.zeros(B, max(1, max_seen), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = min(int(b["user_seen_items"].shape[0]), 5000)
        if n <= 0:
            continue
        out["user_seen_items"][i, :n] = torch.from_numpy(b["user_seen_items"][:n])
        out["user_seen_mask"][i, :n] = True

    return out


# Model blocks


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class GatedModalityFusion(nn.Module):
    def __init__(self, title_dim: int, text_dim: int, image_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.title_proj = nn.Sequential(nn.Linear(title_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.image_proj = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Linear(hidden_dim * 3, 3)

    def forward(self, title_emb, text_emb, image_emb, has_title, has_text, has_image):
        t_title = self.title_proj(title_emb)
        t_text = self.text_proj(text_emb)
        t_image = self.image_proj(image_emb)
        logits = self.gate(torch.cat([t_title, t_text, t_image], dim=-1))

        if has_title.dim() < logits.dim():
            has_title = has_title.unsqueeze(-1)
            has_text = has_text.unsqueeze(-1)
            has_image = has_image.unsqueeze(-1)
        avail = torch.stack([has_title > 0.5, has_text > 0.5, has_image > 0.5], dim=-1).squeeze(-2)
        logits = logits.masked_fill(~avail, -1e4)
        w = F.softmax(logits, dim=-1)
        w = w.masked_fill(~avail.any(dim=-1, keepdim=True), 0.0)
        return w[..., 0:1] * t_title + w[..., 1:2] * t_text + w[..., 2:3] * t_image


class EventEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_feat_dim: int, item_emb: nn.Embedding, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.item_emb = item_emb
        self.num_mlp = nn.Sequential(nn.Linear(num_feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(cfg.DROPOUT))
        self.modality_fusion = GatedModalityFusion(cfg.TITLE_EMB_DIM, cfg.TEXT_EMB_DIM, cfg.IMAGE_EMB_DIM, hidden_dim, cfg.DROPOUT)
        self.recency_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.missing_emb = nn.Embedding(cfg.NUM_MISSING_PATTERNS, cfg.MISSING_PATTERN_DIM)
        self.missing_proj = nn.Linear(cfg.MISSING_PATTERN_DIM, hidden_dim)
        self.bin_emb = nn.Embedding(cfg.NUM_BINS + 1, hidden_dim // 4)
        self.bin_proj = nn.Linear(hidden_dim // 4, hidden_dim)
        self.out_ln = nn.LayerNorm(hidden_dim)

    def forward(self, item_ids, num_feat, time_age, title_emb, text_emb, image_emb, missing_pattern=None, bin_ids=None):
        item_term = self.item_emb(item_ids)
        num_term = self.num_mlp(num_feat)
        rec_term = self.recency_mlp(time_age.unsqueeze(-1))
        mod_term = self.modality_fusion(title_emb, text_emb, image_emb, num_feat[..., 1], num_feat[..., 2], num_feat[..., 3])
        out = item_term + num_term + rec_term + mod_term
        if missing_pattern is not None:
            pat = missing_pattern.clamp(0, self.cfg.NUM_MISSING_PATTERNS - 1)
            out = out + self.missing_proj(self.missing_emb(pat))
        if bin_ids is not None:
            b = bin_ids.clamp(0, self.cfg.NUM_BINS)
            out = out + self.bin_proj(self.bin_emb(b))
        return self.out_ln(out)


class CausalTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, num_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.pos = PositionalEncoding(hidden_dim, max_len, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def _get_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        key = (L, device)
        if key not in self._mask_cache:
            mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
            self._mask_cache[key] = mask.masked_fill(mask == 1, float("-inf"))
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_sequence: bool = False):
        if mask.dim() != 2:
            raise ValueError("mask must have shape [B,L]")
        safe_mask = mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True

        x = self.pos(x)
        B, L, _ = x.shape
        causal = self._get_causal_mask(L, x.device)
        y = self.enc(x, mask=causal, src_key_padding_mask=~safe_mask)
        lengths = safe_mask.sum(dim=1).clamp(min=1) - 1
        last = y[torch.arange(B, device=x.device), lengths]
        if return_sequence:
            return last, y
        return last


class UserTimeContextEncoder(nn.Module):
    def __init__(self, n_groups: int, num_bins: int, o_t_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.group_emb = nn.Embedding(n_groups, hidden_dim)
        self.bin_emb = nn.Embedding(num_bins, hidden_dim // 2)
        self.global_encoder = nn.Sequential(nn.Linear(o_t_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.group_encoder = nn.Sequential(nn.Linear(o_t_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.global_delta_encoder = nn.Sequential(nn.Linear(o_t_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.group_delta_encoder = nn.Sequential(nn.Linear(o_t_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 3 + hidden_dim // 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())

    def forward(self, group_ids, bin_ids, global_ot, group_ot, global_ot_delta, group_ot_delta):
        group_ids = group_ids.clamp(0, self.group_emb.num_embeddings - 1)
        bin_ids = bin_ids.clamp(0, self.bin_emb.num_embeddings - 1)
        g = self.group_emb(group_ids)
        b = self.bin_emb(bin_ids)
        go = self.global_encoder(global_ot)
        gro = self.group_encoder(group_ot)
        gd = self.global_delta_encoder(global_ot_delta)
        grd = self.group_delta_encoder(group_ot_delta)
        return self.fuse(torch.cat([g + gd + grd, go, gro, b], dim=-1))


class TSSSMUserUpdater(nn.Module):
    """
    User state update:
      delta = eta2 * systematic_correction + gate * eta3 * individual_correction
    """

    def __init__(self, hidden_dim: int, deviation_dim: int, dropout: float, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.deviation_encoder = nn.Sequential(
            nn.Linear(deviation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + deviation_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.individual_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.eta2_raw = nn.Parameter(torch.tensor(0.0))
        self.eta3_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_prev: torch.Tensor, systematic_correction: torch.Tensor, deviation: torch.Tensor) -> Dict[str, torch.Tensor]:
        dev_emb = self.deviation_encoder(deviation)
        gate = self.gate_net(torch.cat([z_prev, deviation], dim=-1))
        individual = self.individual_net(torch.cat([z_prev, dev_emb], dim=-1))

        eta2 = F.softplus(self.eta2_raw)
        eta3 = F.softplus(self.eta3_raw)
        raw_update = eta2 * systematic_correction + gate * eta3 * individual

        z_norm = z_prev.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        u_norm = raw_update.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        max_u = self.cfg.MAX_UPDATE_RATIO * z_norm
        scale = (max_u / u_norm).clamp(max=1.0)
        bounded = scale * raw_update
        z_new = z_prev + bounded

        return {
            "z_new": z_new,
            "gate": gate,
            "individual_correction": individual,
            "systematic_norm": systematic_correction.norm(dim=-1).mean(),
            "individual_norm": individual.norm(dim=-1).mean(),
            "actual_ratio": (bounded.norm(dim=-1) / z_norm.squeeze(-1)).mean(),
            "eta2": eta2,
            "eta3": eta3,
        }


class ItemTimeContextEncoder(nn.Module):
    def __init__(self, n_item_groups: int, num_bins: int, o_t_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.group_emb = nn.Embedding(n_item_groups, hidden_dim)
        self.bin_emb = nn.Embedding(num_bins, hidden_dim // 2)
        self.global_encoder = nn.Sequential(nn.Linear(o_t_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.group_encoder = nn.Sequential(nn.Linear(o_t_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.global_delta_encoder = nn.Sequential(nn.Linear(o_t_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.group_delta_encoder = nn.Sequential(nn.Linear(o_t_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 3 + hidden_dim // 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())

    def forward(self, item_group_ids, bin_ids, global_ot, group_ot, global_ot_delta, group_ot_delta):
        item_group_ids = item_group_ids.clamp(0, self.group_emb.num_embeddings - 1)
        bin_ids = bin_ids.clamp(0, self.bin_emb.num_embeddings - 1)
        g = self.group_emb(item_group_ids)
        b = self.bin_emb(bin_ids)
        go = self.global_encoder(global_ot)
        gro = self.group_encoder(group_ot)
        gd = self.global_delta_encoder(global_ot_delta)
        grd = self.group_delta_encoder(group_ot_delta)
        return self.fuse(torch.cat([g + gd + grd, go, gro, b], dim=-1))


class TSSSMItemUpdater(nn.Module):
    """
    Item-side dynamic decomposition:
      v_new = v_prev + bounded(beta2 * item_systematic + rho * beta3 * item_individual + beta4 * carry_memory)
    """

    def __init__(self, hidden_dim: int, deviation_dim: int, dropout: float, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.dev_encoder = nn.Sequential(nn.Linear(deviation_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + deviation_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.individual_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.beta2_raw = nn.Parameter(torch.tensor(0.0))
        self.beta3_raw = nn.Parameter(torch.tensor(0.0))
        self.beta4_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, v_prev: torch.Tensor, item_systematic: torch.Tensor, item_deviation: torch.Tensor, carry_memory: torch.Tensor) -> Dict[str, torch.Tensor]:
        dev_emb = self.dev_encoder(item_deviation)
        rho = self.gate_net(torch.cat([v_prev, item_deviation], dim=-1))
        individual = self.individual_net(torch.cat([v_prev, dev_emb], dim=-1))
        beta2 = F.softplus(self.beta2_raw)
        beta3 = F.softplus(self.beta3_raw)
        beta4 = F.softplus(self.beta4_raw)
        raw_update = beta2 * item_systematic + rho * beta3 * individual + beta4 * carry_memory

        v_norm = v_prev.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        u_norm = raw_update.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        max_u = self.cfg.ITEM_MAX_UPDATE_RATIO * v_norm
        scale = (max_u / u_norm).clamp(max=1.0)
        bounded = scale * raw_update
        v_new = v_prev + bounded

        return {
            "v_new": v_new,
            "rho": rho,
            "individual_correction": individual,
            "item_systematic_norm": item_systematic.norm(dim=-1).mean(),
            "item_individual_norm": individual.norm(dim=-1).mean(),
            "carry_norm": carry_memory.norm(dim=-1).mean(),
            "item_actual_ratio": (bounded.norm(dim=-1) / v_norm.squeeze(-1)).mean(),
            "beta2": beta2,
            "beta3": beta3,
            "beta4": beta4,
        }


class ShockEncoder(nn.Module):
    def __init__(self, shock_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shock_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.shock_head = nn.Linear(hidden_dim, 1)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, shock_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(shock_features)
        shock = torch.tanh(self.shock_head(h).squeeze(-1))
        msg = self.msg_proj(h)
        return {"shock": shock, "message": msg}


class CarryoverMemory(nn.Module):
    """
    Builds carryover memory with asymmetric attenuation:
      exp(-lambda_pos * delta_bin) for positive shock
      exp(-lambda_neg * delta_bin) for negative shock
    """

    def __init__(self, hidden_dim: int, item_num_feat_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.missing_emb = nn.Embedding(cfg.NUM_MISSING_PATTERNS, cfg.MISSING_PATTERN_DIM)
        self.obs_net = nn.Sequential(
            nn.Linear(cfg.MISSING_PATTERN_DIM + 3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.rel_net = nn.Sequential(
            nn.Linear(item_num_feat_dim + cfg.DEVIATION_DIM + cfg.ITEM_DEVIATION_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.shock_net = nn.Sequential(
            nn.Linear(item_num_feat_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lambda_pos_raw = nn.Parameter(torch.tensor(math.log(math.exp(cfg.CARRYOVER_LAMBDA_POS_INIT) - 1.0)))
        self.lambda_neg_raw = nn.Parameter(torch.tensor(math.log(math.exp(cfg.CARRYOVER_LAMBDA_NEG_INIT) - 1.0)))

    def forward(
        self,
        item_hist_tokens: torch.Tensor,
        item_hist_num: torch.Tensor,
        item_hist_bin_ids: torch.Tensor,
        target_bins: torch.Tensor,
        item_hist_missing_pattern: torch.Tensor,
        item_hist_mask: torch.Tensor,
        user_deviation: torch.Tensor,
        item_deviation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, L, H = item_hist_tokens.shape
        mp = item_hist_missing_pattern.clamp(min=0, max=self.cfg.NUM_MISSING_PATTERNS - 1)
        mp_emb = self.missing_emb(mp)

        obs_feat = torch.cat([mp_emb, item_hist_num[..., 1:4]], dim=-1)
        obs_w = torch.sigmoid(self.obs_net(obs_feat).squeeze(-1))

        u_dev_expand = user_deviation.unsqueeze(1).expand(-1, L, -1)
        i_dev_expand = item_deviation.unsqueeze(1).expand(-1, L, -1)
        rel_feat = torch.cat([item_hist_num, u_dev_expand, i_dev_expand], dim=-1)
        rel_w = torch.sigmoid(self.rel_net(rel_feat).squeeze(-1))

        shock_raw = self.shock_net(item_hist_num).squeeze(-1)
        shock = torch.tanh(shock_raw + item_hist_num[..., 10] + 0.3 * item_hist_num[..., 9])
        sign_w = torch.tanh(shock)

        # Carryover uses the train-aligned time-bin scale.
        # Other recency modules continue to use exact elapsed days.
        age_bins = (target_bins.unsqueeze(1) - item_hist_bin_ids).clamp(min=0).to(item_hist_tokens.dtype)
        lambda_pos = F.softplus(self.lambda_pos_raw)
        lambda_neg = F.softplus(self.lambda_neg_raw)
        decay_pos = torch.exp(-lambda_pos * age_bins)
        decay_neg = torch.exp(-lambda_neg * age_bins)
        time_w = torch.where(shock >= 0.0, decay_pos, decay_neg)

        mask = item_hist_mask.float()
        omega = obs_w * rel_w * (1.0 + shock.abs()) * time_w * mask
        msg = self.msg_proj(item_hist_tokens)
        signed_msg = sign_w.unsqueeze(-1) * msg
        weighted = omega.unsqueeze(-1) * signed_msg
        denom = omega.sum(dim=1, keepdim=True).clamp(min=1e-6)
        memory = weighted.sum(dim=1) / denom

        return {
            "memory": memory,
            "omega_mean": (omega.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).mean(),
            "shock_mean": (shock * mask).sum() / mask.sum().clamp(min=1.0),
            "lambda_pos": lambda_pos,
            "lambda_neg": lambda_neg,
        }


class TypedMessagePassing(nn.Module):
    """
    Sparse typed propagation:
      user <- history items (MNAR-weighted)
      anchor item <- user
      anchor item <- item-item neighbors
      user <- updated anchor item
    """

    def __init__(self, hidden_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.obs_net = nn.Sequential(
            nn.Linear(cfg.MISSING_PATTERN_DIM + 3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.rel_net = nn.Sequential(
            nn.Linear(cfg.USER_NUM_FEAT_DIM + cfg.DEVIATION_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.W_ui = nn.Linear(hidden_dim, hidden_dim)
        self.W_iu = nn.Linear(hidden_dim, hidden_dim)
        self.W_ii = nn.Linear(hidden_dim, hidden_dim)
        self.W_fb = nn.Linear(hidden_dim, hidden_dim)
        self.gate_ui = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.gate_fb = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.missing_emb = nn.Embedding(cfg.NUM_MISSING_PATTERNS, cfg.MISSING_PATTERN_DIM)
        self.dropout = nn.Dropout(cfg.MESSAGE_DROPOUT)
        self.time_lambda_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.2) - 1.0)))

    def forward(
        self,
        z_user: torch.Tensor,
        user_hist_tokens: torch.Tensor,
        user_hist_item_states: torch.Tensor,
        user_hist_num: torch.Tensor,
        user_hist_time_age: torch.Tensor,
        user_hist_missing_pattern: torch.Tensor,
        user_hist_mask: torch.Tensor,
        deviation: torch.Tensor,
        anchor_item_state: torch.Tensor,
        anchor_neighbor_states: torch.Tensor,
        anchor_neighbor_mask: torch.Tensor,
        current_shock_msg: torch.Tensor,
        current_shock: torch.Tensor,
        current_rel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, L, H = user_hist_tokens.shape
        mp = user_hist_missing_pattern.clamp(min=0, max=self.cfg.NUM_MISSING_PATTERNS - 1)
        mp_emb = self.missing_emb(mp)

        obs_feat = torch.cat([mp_emb, user_hist_num[..., 1:4]], dim=-1)
        obs_w = torch.sigmoid(self.obs_net(obs_feat).squeeze(-1))

        dev_expand = deviation.unsqueeze(1).expand(-1, L, -1)
        rel_feat = torch.cat([user_hist_num, dev_expand], dim=-1)
        rel_w = torch.sigmoid(self.rel_net(rel_feat).squeeze(-1))

        rating_residual = user_hist_num[..., 0] - user_hist_num[..., 13]
        sign_w = torch.tanh(rating_residual + 0.2 * (user_hist_num[..., 5] - user_hist_num[..., 9]))

        age_days = torch.expm1(user_hist_time_age).clamp(min=0.0)
        lam = F.softplus(self.time_lambda_raw)
        time_w = torch.exp(-lam * age_days)

        mask = user_hist_mask.float()
        omega = obs_w * rel_w * (1.0 + sign_w.abs()) * time_w * mask

        ui_msg = self.W_ui(user_hist_item_states + user_hist_tokens)
        user_agg = (omega.unsqueeze(-1) * ui_msg).sum(dim=1) / omega.sum(dim=1, keepdim=True).clamp(min=1e-6)

        current_w = torch.sigmoid(current_rel) * (1.0 + current_shock.abs())
        user_from_current = current_w.unsqueeze(-1) * self.W_ui(current_shock_msg)
        z_mid = z_user + self.dropout(user_agg + user_from_current)

        anchor_state = anchor_item_state
        g_ui = self.gate_ui(torch.cat([z_mid, anchor_state], dim=-1))
        anchor_state = anchor_state + g_ui * self.W_iu(z_mid)

        neigh = anchor_neighbor_states
        logits = (anchor_state.unsqueeze(1) * neigh).sum(dim=-1) / math.sqrt(float(H))
        logits = logits.masked_fill(~anchor_neighbor_mask, -1e9)
        alpha = torch.softmax(logits, dim=1)
        alpha = alpha.masked_fill(~anchor_neighbor_mask, 0.0)
        neigh_msg = (alpha.unsqueeze(-1) * self.W_ii(neigh)).sum(dim=1)
        anchor_state = anchor_state + self.dropout(neigh_msg)

        g_fb = self.gate_fb(torch.cat([z_mid, anchor_state], dim=-1))
        z_final = z_mid + g_fb * self.W_fb(anchor_state)

        return {
            "z_final": z_final,
            "anchor_state": anchor_state,
            "omega_mean": (omega.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).mean(),
            "msg_gate_ui": g_ui.mean(),
            "msg_gate_fb": g_fb.mean(),
            "msg_time_lambda": lam,
            "msg_current_weight": current_w.mean(),
        }


class ReliabilityHead(nn.Module):
    def __init__(self, shock_dim: int, dev_dim: int, item_dev_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        in_dim = shock_dim + dev_dim + item_dev_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.obs_head = nn.Linear(hidden_dim // 2, 1)
        self.rel_head = nn.Linear(hidden_dim // 2, 1)
        self.conflict_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, shock_feat: torch.Tensor, deviation: torch.Tensor, item_deviation: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([shock_feat, deviation, item_deviation], dim=-1)
        h = self.net(x)
        obs = self.obs_head(h).squeeze(-1)
        rel = self.rel_head(h).squeeze(-1)
        conflict = self.conflict_head(h).squeeze(-1)
        return {"obs": obs, "rel": rel, "conflict": conflict}


class DriftHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CarryHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, item_state: torch.Tensor, carry_memory: torch.Tensor, shock_msg: torch.Tensor):
        x = torch.cat([item_state, carry_memory, shock_msg], dim=-1)
        return self.net(x)


class ItemDynamicBias(nn.Module):
    """
    Time-aware item bias based on (item_group, query_bin).
    """

    def __init__(self, n_item_groups: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.group_emb = nn.Embedding(n_item_groups, hidden_dim // 4)
        self.bin_emb = nn.Embedding(num_bins, hidden_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, item_group_ids: torch.Tensor, query_bins: torch.Tensor) -> torch.Tensor:
        g = self.group_emb(item_group_ids)
        b = self.bin_emb(query_bins)
        return self.mlp(torch.cat([g, b], dim=-1)).squeeze(-1)

    def forward_batched(self, item_group_ids: torch.Tensor, query_bins: torch.Tensor) -> torch.Tensor:
        # item_group_ids: [N], query_bins: [B] -> [B, N]
        g = self.group_emb(item_group_ids).unsqueeze(0)  # [1,N,D]
        b = self.bin_emb(query_bins).unsqueeze(1)  # [B,1,D]
        x = torch.cat([g.expand(query_bins.size(0), -1, -1), b.expand(-1, item_group_ids.size(0), -1)], dim=-1)
        return self.mlp(x).squeeze(-1)


# Model


class TSSSMModel(nn.Module):
    def __init__(self, num_items: int, pad_idx: int, cfg: Config, item_group_arr: np.ndarray):
        super().__init__()
        self.num_items = int(num_items)
        self.pad_idx = int(pad_idx)
        self.cfg = cfg
        H = cfg.HIDDEN_DIM

        self.item_emb = nn.Embedding(num_items, H, padding_idx=pad_idx)
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=pad_idx)
        nn.init.zeros_(self.item_bias.weight)

        # Query-conditioned dynamic item state backbone (used by negatives/full-sort/message passing).
        self.item_state_group_emb = nn.Embedding(cfg.N_ITEM_GROUPS, H)
        self.item_state_bin_emb = nn.Embedding(cfg.NUM_BINS, H)
        self.item_state_mlp = nn.Sequential(
            nn.Linear(H * 3, H),
            nn.LayerNorm(H),
            nn.Tanh(),
            nn.Dropout(cfg.DROPOUT),
        )
        self.item_state_gate = nn.Sequential(
            nn.Linear(H * 2, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
            nn.Sigmoid(),
        )

        # The observed target review is a distinct event, not part of either
        # pre-query history. Encode its title/text/image content together with
        # its numeric and availability signals so it can condition both sides.
        self.current_event_encoder = EventEncoder(H, cfg.CURRENT_EVENT_NUM_FEAT_DIM, self.item_emb, cfg)
        self.current_to_user_deviation = nn.Linear(H, cfg.DEVIATION_DIM, bias=False)
        self.current_to_item_deviation = nn.Linear(H, cfg.ITEM_DEVIATION_DIM, bias=False)
        self.current_message_proj = nn.Linear(H, H, bias=False)

        if item_group_arr.shape[0] < num_items:
            arr = np.zeros(num_items, dtype=np.int64)
            arr[: item_group_arr.shape[0]] = item_group_arr
            item_group_arr = arr
        self.register_buffer("item_group_ids", torch.from_numpy(item_group_arr[:num_items].astype(np.int64)))

        # User side
        self.user_event_encoder = EventEncoder(H, cfg.USER_NUM_FEAT_DIM, self.item_emb, cfg)
        self.user_seq_encoder = CausalTransformerEncoder(H, cfg.NUM_HEADS, cfg.NUM_LAYERS, cfg.DROPOUT, cfg.MAX_USER_SEQ_LEN)
        self.user_time_context = UserTimeContextEncoder(cfg.N_GROUPS, cfg.NUM_BINS, cfg.O_T_DIM, H, cfg.DROPOUT)
        self.user_updater = TSSSMUserUpdater(H, cfg.DEVIATION_DIM, cfg.DROPOUT, cfg)

        # Item side
        self.item_event_encoder = EventEncoder(H, cfg.ITEM_NUM_FEAT_DIM, self.item_emb, cfg)
        self.item_seq_encoder = CausalTransformerEncoder(H, cfg.NUM_HEADS, cfg.ITEM_NUM_LAYERS, cfg.DROPOUT, cfg.MAX_ITEM_SEQ_LEN)
        self.item_time_context = ItemTimeContextEncoder(cfg.N_ITEM_GROUPS, cfg.NUM_BINS, cfg.O_T_DIM, H, cfg.DROPOUT)
        self.item_updater = TSSSMItemUpdater(H, cfg.ITEM_DEVIATION_DIM, cfg.DROPOUT, cfg)

        self.shock_encoder = ShockEncoder(cfg.SHOCK_FEAT_DIM, H, cfg.DROPOUT)
        self.carryover_memory = CarryoverMemory(H, cfg.ITEM_NUM_FEAT_DIM, cfg)
        self.message_passing = TypedMessagePassing(H, cfg)
        self.reliability_head = ReliabilityHead(cfg.SHOCK_FEAT_DIM, cfg.DEVIATION_DIM, cfg.ITEM_DEVIATION_DIM, H, cfg.DROPOUT)

        self.drift_head = DriftHead(H, out_dim=4, dropout=cfg.DROPOUT)
        self.carry_head = CarryHead(H, out_dim=2, dropout=cfg.DROPOUT)

        if cfg.USE_DYNAMIC_ITEM_BIAS:
            self.item_dynamic_bias = ItemDynamicBias(cfg.N_ITEM_GROUPS, cfg.NUM_BINS, H)
        else:
            self.item_dynamic_bias = None

        # Explicit no-history states to avoid pseudo-token bias on empty histories.
        self.user_empty_state = nn.Parameter(torch.zeros(H))
        self.item_empty_state = nn.Parameter(torch.zeros(H))

        self._init_weights()
        # Near-neutral residual initialization keeps current-event conditioning
        # stable at startup while allowing ranking gradients to reach the adapters.
        nn.init.normal_(self.current_to_user_deviation.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.current_to_item_deviation.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.current_message_proj.weight, mean=0.0, std=1e-3)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_emb(item_ids)

    def get_item_bias(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_bias(item_ids).squeeze(-1)

    def encode_current_event(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        current_num = torch.cat(
            [
                (batch["target_rating"] / 5.0).unsqueeze(-1),
                batch["target_has_title"].unsqueeze(-1),
                batch["target_has_text"].unsqueeze(-1),
                batch["target_has_image"].unsqueeze(-1),
                batch["shock_features"],
            ],
            dim=-1,
        )
        current_age = torch.zeros_like(batch["target_rating"])
        # Information-boundary control for event-conditioned retrieval: the
        # observed review content may condition the query, but its ground-truth
        # catalog id must not be copied into the query used to rank that id.
        # The target id remains available to the item-side transition and
        # auxiliary heads through the rest of the batch.
        current_event_item_ids = torch.full_like(batch["target_item"], self.pad_idx)
        return self.current_event_encoder(
            current_event_item_ids,
            current_num,
            current_age,
            batch["target_title_emb"],
            batch["target_text_emb"],
            batch["target_image_emb"],
            missing_pattern=batch["target_missing_pattern"],
            bin_ids=batch["target_bin"],
        )

    def encode_current_conditioning(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        current_event = self.encode_current_event(batch)
        user_deviation = batch["deviation"] + self.current_to_user_deviation(current_event)
        item_deviation = batch["item_deviation"] + self.current_to_item_deviation(current_event)
        shock_out = self.shock_encoder(batch["shock_features"])
        shock_out["message"] = shock_out["message"] + self.current_message_proj(current_event)
        rel_out = self.reliability_head(batch["shock_features"], user_deviation, item_deviation)
        return {
            "current_event": current_event,
            "user_deviation": user_deviation,
            "item_deviation": item_deviation,
            "shock_out": shock_out,
            "rel_out": rel_out,
        }

    @torch.no_grad()
    def get_carryover_stats(self) -> Dict[str, float]:
        lambda_pos = float(F.softplus(self.carryover_memory.lambda_pos_raw).detach().cpu().item())
        lambda_neg = float(F.softplus(self.carryover_memory.lambda_neg_raw).detach().cpu().item())
        return {
            "lambda_pos_per_bin": lambda_pos,
            "lambda_neg_per_bin": lambda_neg,
            "half_life_pos_bins": math.log(2.0) / max(lambda_pos, 1e-12),
            "half_life_neg_bins": math.log(2.0) / max(lambda_neg, 1e-12),
        }

    def _dynamic_item_state_for_bin(self, item_ids_1d: torch.Tensor, query_bin: int) -> torch.Tensor:
        ids = item_ids_1d.clamp(0, self.num_items - 1)
        base = self.item_emb(ids)
        groups = self.item_group_ids[ids]
        g = self.item_state_group_emb(groups)
        b = self.item_state_bin_emb(
            torch.full_like(ids, int(max(0, min(query_bin, self.cfg.NUM_BINS - 1))), dtype=torch.long)
        )
        delta = self.item_state_mlp(torch.cat([base, g, b], dim=-1))
        gate = self.item_state_gate(torch.cat([base, delta], dim=-1))
        state = base + gate * delta
        valid = (item_ids_1d != self.pad_idx).unsqueeze(-1)
        return state * valid

    def _dynamic_item_state_point(self, item_ids: torch.Tensor, query_bins: torch.Tensor) -> torch.Tensor:
        ids = item_ids.clamp(0, self.num_items - 1)
        base = self.item_emb(ids)
        groups = self.item_group_ids[ids]
        g = self.item_state_group_emb(groups)
        b = self.item_state_bin_emb(query_bins.clamp(0, self.cfg.NUM_BINS - 1))
        delta = self.item_state_mlp(torch.cat([base, g, b], dim=-1))
        gate = self.item_state_gate(torch.cat([base, delta], dim=-1))
        state = base + gate * delta
        valid = (item_ids != self.pad_idx).unsqueeze(-1)
        return state * valid

    def _dynamic_item_state_pair(self, item_ids: torch.Tensor, query_bins: torch.Tensor) -> torch.Tensor:
        ids = item_ids.clamp(0, self.num_items - 1)
        base = self.item_emb(ids)
        groups = self.item_group_ids[ids]
        g = self.item_state_group_emb(groups)
        b = self.item_state_bin_emb(query_bins.clamp(0, self.cfg.NUM_BINS - 1)).unsqueeze(1).expand_as(base)
        delta = self.item_state_mlp(torch.cat([base, g, b], dim=-1))
        gate = self.item_state_gate(torch.cat([base, delta], dim=-1))
        state = base + gate * delta
        valid = (item_ids != self.pad_idx).unsqueeze(-1)
        return state * valid

    def _dynamic_bias_pointwise(self, item_ids: torch.Tensor, query_bins: torch.Tensor) -> torch.Tensor:
        if self.item_dynamic_bias is None:
            return torch.zeros_like(query_bins, dtype=torch.float32, device=query_bins.device)
        groups = self.item_group_ids[item_ids.clamp(0, self.num_items - 1)]
        return self.item_dynamic_bias(groups, query_bins.clamp(0, self.cfg.NUM_BINS - 1))

    def _dynamic_bias_pairwise(self, item_ids: torch.Tensor, query_bins: torch.Tensor) -> torch.Tensor:
        if self.item_dynamic_bias is None:
            return torch.zeros(item_ids.size(0), item_ids.size(1), device=item_ids.device, dtype=torch.float32)
        groups = self.item_group_ids[item_ids.clamp(0, self.num_items - 1)]
        b = query_bins.clamp(0, self.cfg.NUM_BINS - 1).unsqueeze(1).expand_as(item_ids)
        return self.item_dynamic_bias(groups, b)

    def _dynamic_bias_chunk(self, item_ids_chunk: torch.Tensor, query_bins: torch.Tensor) -> torch.Tensor:
        if self.item_dynamic_bias is None:
            return torch.zeros(query_bins.size(0), item_ids_chunk.size(0), device=query_bins.device, dtype=torch.float32)
        groups = self.item_group_ids[item_ids_chunk.clamp(0, self.num_items - 1)]
        qb = query_bins.clamp(0, self.cfg.NUM_BINS - 1)
        return self.item_dynamic_bias.forward_batched(groups, qb)

    def _build_forward_item_state_cache(self, batch: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
        cache: Dict[int, Dict[str, torch.Tensor]] = {}
        query_bins = batch["target_bin"]
        pos_items = batch["target_item"]
        neg_items = batch.get("negative_items")

        for b in torch.unique(query_bins).tolist():
            b_int = int(b)
            rows = query_bins == b_int
            parts: List[torch.Tensor] = []
            for key in ("user_hist_items", "anchor_item", "anchor_neighbors"):
                if key not in batch:
                    continue
                part = batch[key][rows]
                if part.numel() > 0:
                    parts.append(part.reshape(-1))
            parts.append(pos_items[rows].reshape(-1))
            if neg_items is not None and neg_items.numel() > 0:
                parts.append(neg_items[rows].reshape(-1))
            if self.cfg.USE_INBATCH_NEGATIVES and self.training:
                parts.append(pos_items.reshape(-1))

            if len(parts) == 0:
                continue
            ids = torch.cat(parts, dim=0)
            ids = ids[(ids != self.pad_idx)]
            if ids.numel() == 0:
                continue

            uniq_ids = torch.unique(ids)
            uniq_ids, _ = torch.sort(uniq_ids)
            states = self._dynamic_item_state_for_bin(uniq_ids, b_int)

            item_bias = self.get_item_bias(uniq_ids) if self.cfg.USE_ITEM_BIAS else torch.zeros_like(uniq_ids, dtype=torch.float32)
            if self.cfg.USE_DYNAMIC_ITEM_BIAS:
                bin_vec = torch.full_like(uniq_ids, b_int, dtype=torch.long)
                dynamic_bias = self._dynamic_bias_pointwise(uniq_ids, bin_vec)
            else:
                dynamic_bias = torch.zeros_like(item_bias)

            cache[b_int] = {
                "ids": uniq_ids,
                "states": states,
                "item_bias": item_bias,
                "dynamic_bias": dynamic_bias,
            }
        return cache

    def _lookup_cached_point(
        self,
        item_ids: torch.Tensor,
        query_bins: torch.Tensor,
        item_state_cache: Optional[Dict[int, Dict[str, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if item_state_cache is None:
            state = self._dynamic_item_state_point(item_ids, query_bins)
            bias = torch.zeros(item_ids.size(0), device=item_ids.device, dtype=state.dtype)
            if self.cfg.USE_ITEM_BIAS:
                bias = bias + self.get_item_bias(item_ids)
            if self.cfg.USE_DYNAMIC_ITEM_BIAS:
                bias = bias + self._dynamic_bias_pointwise(item_ids, query_bins)
            return state, bias

        H = self.cfg.HIDDEN_DIM
        state = torch.zeros(item_ids.size(0), H, device=item_ids.device, dtype=self.item_emb.weight.dtype)
        bias = torch.zeros(item_ids.size(0), device=item_ids.device, dtype=torch.float32)
        for b in torch.unique(query_bins).tolist():
            b_int = int(b)
            rows = torch.where(query_bins == b_int)[0]
            if rows.numel() == 0:
                continue
            entry = item_state_cache.get(b_int)
            ids = item_ids[rows]
            if entry is None or entry["ids"].numel() == 0:
                continue
            lookup_ids = ids.clamp(0, self.num_items - 1)
            pos = torch.searchsorted(entry["ids"], lookup_ids)
            valid = (ids != self.pad_idx) & (pos < entry["ids"].numel())
            if valid.any():
                pos_valid = pos[valid]
                valid_match = entry["ids"][pos_valid] == lookup_ids[valid]
                if valid_match.any():
                    dst_rows = rows[valid][valid_match]
                    state[dst_rows] = entry["states"][pos_valid[valid_match]]
                    bias[dst_rows] = entry["item_bias"][pos_valid[valid_match]] + entry["dynamic_bias"][pos_valid[valid_match]]
        return state, bias

    def _lookup_cached_pair(
        self,
        item_ids: torch.Tensor,
        query_bins: torch.Tensor,
        item_state_cache: Optional[Dict[int, Dict[str, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if item_state_cache is None:
            state = self._dynamic_item_state_pair(item_ids, query_bins)
            bias = torch.zeros(item_ids.size(0), item_ids.size(1), device=item_ids.device, dtype=state.dtype)
            if self.cfg.USE_ITEM_BIAS:
                bias = bias + self.get_item_bias(item_ids)
            if self.cfg.USE_DYNAMIC_ITEM_BIAS:
                bias = bias + self._dynamic_bias_pairwise(item_ids, query_bins)
            return state, bias

        B, N = item_ids.shape
        H = self.cfg.HIDDEN_DIM
        state = torch.zeros(B, N, H, device=item_ids.device, dtype=self.item_emb.weight.dtype)
        bias = torch.zeros(B, N, device=item_ids.device, dtype=torch.float32)
        for b in torch.unique(query_bins).tolist():
            b_int = int(b)
            rows = torch.where(query_bins == b_int)[0]
            if rows.numel() == 0:
                continue
            entry = item_state_cache.get(b_int)
            if entry is None or entry["ids"].numel() == 0:
                continue
            ids = item_ids[rows]
            flat_ids = ids.reshape(-1)
            lookup_ids = flat_ids.clamp(0, self.num_items - 1)
            pos = torch.searchsorted(entry["ids"], lookup_ids)
            valid = (flat_ids != self.pad_idx) & (pos < entry["ids"].numel())

            flat_state = torch.zeros(flat_ids.size(0), H, device=item_ids.device, dtype=self.item_emb.weight.dtype)
            flat_bias = torch.zeros(flat_ids.size(0), device=item_ids.device, dtype=torch.float32)
            if valid.any():
                pos_valid = pos[valid]
                valid_match = entry["ids"][pos_valid] == lookup_ids[valid]
                if valid_match.any():
                    valid_idx = torch.where(valid)[0][valid_match]
                    flat_state[valid_idx] = entry["states"][pos_valid[valid_match]]
                    flat_bias[valid_idx] = entry["item_bias"][pos_valid[valid_match]] + entry["dynamic_bias"][pos_valid[valid_match]]

            state[rows] = flat_state.view(-1, N, H)
            bias[rows] = flat_bias.view(-1, N)
        return state, bias

    def encode_user(
        self,
        batch: Dict[str, torch.Tensor],
        deviation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        user_event = self.user_event_encoder(
            batch["user_hist_items"],
            batch["user_hist_num"],
            batch["user_hist_time_age"],
            batch["user_hist_title_emb"],
            batch["user_hist_text_emb"],
            batch["user_hist_image_emb"],
            missing_pattern=batch["user_hist_missing_pattern"],
            bin_ids=batch["user_hist_bin_ids"],
        )
        z_base, user_tokens = self.user_seq_encoder(user_event, batch["user_hist_mask"], return_sequence=True)
        user_empty = batch["user_hist_len"] <= 0
        if user_empty.any():
            z_base = z_base.clone()
            z_base[user_empty] = self.user_empty_state.unsqueeze(0)
            user_tokens = user_tokens.clone()
            user_tokens[user_empty] = 0.0
        user_sys = self.user_time_context(
            batch["user_group"],
            batch["target_bin"],
            batch["global_ot"],
            batch["group_ot"],
            batch["global_ot_delta"],
            batch["group_ot_delta"],
        )
        if deviation is None:
            deviation = batch["deviation"]
        u_up = self.user_updater(z_base, user_sys, deviation)
        return {
            "z_base": z_base,
            "z_state": u_up["z_new"],
            "user_tokens": user_tokens,
            "u_gate": u_up["gate"],
            "u_eta2": u_up["eta2"],
            "u_eta3": u_up["eta3"],
            "u_actual_ratio": u_up["actual_ratio"],
            "u_systematic_norm": u_up["systematic_norm"],
            "u_individual_norm": u_up["individual_norm"],
        }

    def encode_item(
        self,
        batch: Dict[str, torch.Tensor],
        user_deviation: Optional[torch.Tensor] = None,
        item_deviation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        item_event = self.item_event_encoder(
            batch["item_hist_items"],
            batch["item_hist_num"],
            batch["item_hist_time_age"],
            batch["item_hist_title_emb"],
            batch["item_hist_text_emb"],
            batch["item_hist_image_emb"],
            missing_pattern=batch["item_hist_missing_pattern"],
            bin_ids=batch["item_hist_bin_ids"],
        )
        v_base, item_tokens = self.item_seq_encoder(item_event, batch["item_hist_mask"], return_sequence=True)
        item_empty = batch["item_hist_len"] <= 0
        if item_empty.any():
            v_base = v_base.clone()
            empty_base = self._dynamic_item_state_point(batch["target_item"], batch["target_bin"])
            v_base[item_empty] = empty_base[item_empty]
            item_tokens = item_tokens.clone()
            item_tokens[item_empty] = 0.0
        item_sys = self.item_time_context(
            batch["item_group"],
            batch["target_bin"],
            batch["item_global_ot"],
            batch["item_group_ot"],
            batch["item_global_ot_delta"],
            batch["item_group_ot_delta"],
        )
        if user_deviation is None:
            user_deviation = batch["deviation"]
        if item_deviation is None:
            item_deviation = batch["item_deviation"]
        carry = self.carryover_memory(
            item_tokens,
            batch["item_hist_num"],
            batch["item_hist_bin_ids"],
            batch["target_bin"],
            batch["item_hist_missing_pattern"],
            batch["item_hist_mask"],
            user_deviation,
            item_deviation,
        )
        i_up = self.item_updater(v_base, item_sys, item_deviation, carry["memory"])
        return {
            "v_base": v_base,
            "v_state": i_up["v_new"],
            "item_tokens": item_tokens,
            "carry_memory": carry["memory"],
            "i_rho": i_up["rho"],
            "i_beta2": i_up["beta2"],
            "i_beta3": i_up["beta3"],
            "i_beta4": i_up["beta4"],
            "i_actual_ratio": i_up["item_actual_ratio"],
            "i_systematic_norm": i_up["item_systematic_norm"],
            "i_individual_norm": i_up["item_individual_norm"],
            "i_carry_norm": i_up["carry_norm"],
            "carry_omega_mean": carry["omega_mean"],
            "carry_shock_mean": carry["shock_mean"],
            "carry_lambda_pos": carry["lambda_pos"],
            "carry_lambda_neg": carry["lambda_neg"],
        }

    def build_query_user_state(
        self,
        batch: Dict[str, torch.Tensor],
        user_out: Dict[str, torch.Tensor],
        shock_out: Dict[str, torch.Tensor],
        rel_out: Dict[str, torch.Tensor],
        deviation: Optional[torch.Tensor] = None,
        item_state_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        user_hist_item_states, _ = self._lookup_cached_pair(batch["user_hist_items"], batch["target_bin"], item_state_cache)
        anchor_item_state, _ = self._lookup_cached_point(batch["anchor_item"], batch["target_bin"], item_state_cache)
        anchor_neighbor_states, _ = self._lookup_cached_pair(batch["anchor_neighbors"], batch["target_bin"], item_state_cache)
        anchor_neighbor_mask = batch["anchor_neighbors"] != self.pad_idx

        if deviation is None:
            deviation = batch["deviation"]
        msg = self.message_passing(
            z_user=user_out["z_state"],
            user_hist_tokens=user_out["user_tokens"],
            user_hist_item_states=user_hist_item_states,
            user_hist_num=batch["user_hist_num"],
            user_hist_time_age=batch["user_hist_time_age"],
            user_hist_missing_pattern=batch["user_hist_missing_pattern"],
            user_hist_mask=batch["user_hist_mask"],
            deviation=deviation,
            anchor_item_state=anchor_item_state,
            anchor_neighbor_states=anchor_neighbor_states,
            anchor_neighbor_mask=anchor_neighbor_mask,
            current_shock_msg=shock_out["message"],
            current_shock=shock_out["shock"],
            current_rel=rel_out["rel"],
        )
        return msg

    def score_point(
        self,
        user_state: torch.Tensor,
        item_ids: torch.Tensor,
        query_bins: torch.Tensor,
        item_state_override: Optional[torch.Tensor] = None,
        item_state_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if item_state_override is not None:
            item_vec = item_state_override
            bias = torch.zeros(item_ids.size(0), device=item_ids.device, dtype=user_state.dtype)
            if self.cfg.USE_ITEM_BIAS:
                bias = bias + self.get_item_bias(item_ids)
            if self.cfg.USE_DYNAMIC_ITEM_BIAS:
                bias = bias + self._dynamic_bias_pointwise(item_ids, query_bins)
        else:
            item_vec, bias = self._lookup_cached_point(item_ids, query_bins, item_state_cache)
        score = (user_state * item_vec).sum(dim=-1)
        return score + bias.to(score.dtype)

    def score_pair(
        self,
        user_state: torch.Tensor,
        item_ids: torch.Tensor,
        query_bins: torch.Tensor,
        item_state_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        item_vec, bias = self._lookup_cached_pair(item_ids, query_bins, item_state_cache)
        score = torch.einsum("bh,bnh->bn", user_state, item_vec)
        return score + bias.to(score.dtype)

    def score_chunk(self, user_state: torch.Tensor, item_ids_chunk: torch.Tensor, query_bins: torch.Tensor) -> torch.Tensor:
        B = user_state.size(0)
        N = item_ids_chunk.size(0)
        score = torch.empty(B, N, device=user_state.device, dtype=user_state.dtype)
        bin_state_cache: Dict[int, torch.Tensor] = {}
        unique_bins = torch.unique(query_bins).tolist()
        for b in unique_bins:
            b_int = int(b)
            if b_int not in bin_state_cache:
                bin_state_cache[b_int] = self._dynamic_item_state_for_bin(item_ids_chunk, b_int)
            item_vec = bin_state_cache[b_int]  # [N,H]
            rows = torch.where(query_bins == b_int)[0]
            if rows.numel() == 0:
                continue
            score[rows] = user_state[rows] @ item_vec.t()
        if self.cfg.USE_ITEM_BIAS:
            score = score + self.get_item_bias(item_ids_chunk).unsqueeze(0)
        if self.cfg.USE_DYNAMIC_ITEM_BIAS:
            score = score + self._dynamic_bias_chunk(item_ids_chunk, query_bins)
        return score

    def _build_inbatch_negatives(
        self,
        user_state: torch.Tensor,
        pos_items: torch.Tensor,
        query_bins: torch.Tensor,
        item_state_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = pos_items.size(0)
        if B <= 1:
            return (
                torch.zeros(B, 0, device=user_state.device, dtype=user_state.dtype),
                torch.zeros(B, 0, device=user_state.device, dtype=torch.bool),
            )
        item_mat = pos_items.unsqueeze(0).expand(B, -1)
        dyn_item, bias = self._lookup_cached_pair(item_mat, query_bins, item_state_cache)
        scores = torch.einsum("bh,bnh->bn", user_state, dyn_item)
        scores = scores + bias.to(scores.dtype)
        mask = pos_items.unsqueeze(1) != pos_items.unsqueeze(0)
        mask = mask & (pos_items.unsqueeze(0) != self.pad_idx)
        return scores, mask

    def forward(self, batch: Dict[str, torch.Tensor], compute_aux: bool = True) -> Dict[str, torch.Tensor]:
        current = self.encode_current_conditioning(batch)
        user_deviation = current["user_deviation"]
        item_deviation = current["item_deviation"]
        shock_out = current["shock_out"]
        rel_out = current["rel_out"]
        user_out = self.encode_user(batch, deviation=user_deviation)
        item_out = self.encode_item(batch, user_deviation=user_deviation, item_deviation=item_deviation)
        item_state_cache = self._build_forward_item_state_cache(batch)
        msg_out = self.build_query_user_state(
            batch,
            user_out,
            shock_out,
            rel_out,
            deviation=user_deviation,
            item_state_cache=item_state_cache,
        )

        z_final = msg_out["z_final"]
        pos_items = batch["target_item"]
        # Positives and candidates share the same query-conditioned backbone.
        pos_score = self.score_point(
            z_final,
            pos_items,
            batch["target_bin"],
            item_state_cache=item_state_cache,
        )

        neg_items = batch["negative_items"]
        if neg_items.numel() > 0 and neg_items.size(1) > 0:
            neg_scores = self.score_pair(z_final, neg_items, batch["target_bin"], item_state_cache=item_state_cache)
            neg_mask = batch["neg_valid_mask"].to(dtype=torch.bool)
        else:
            neg_scores = torch.zeros(z_final.size(0), 0, device=z_final.device)
            neg_mask = torch.zeros(z_final.size(0), 0, device=z_final.device, dtype=torch.bool)

        if self.cfg.USE_INBATCH_NEGATIVES and self.training:
            inb_scores, inb_mask = self._build_inbatch_negatives(
                z_final,
                pos_items,
                batch["target_bin"],
                item_state_cache=item_state_cache,
            )
            if inb_scores.numel() > 0:
                neg_scores = torch.cat([neg_scores, inb_scores], dim=1)
                neg_mask = torch.cat([neg_mask, inb_mask], dim=1)

        out = {
            "user_state": z_final,
            "item_state": item_out["v_state"],
            "pos_score": pos_score,
            "neg_scores": neg_scores,
            "neg_mask": neg_mask,
            "obs_pred": rel_out["obs"],
            "rel_pred": rel_out["rel"],
            "conflict_pred": rel_out["conflict"],
            "shock_value": shock_out["shock"],
            "shock_message": shock_out["message"],
            "carry_memory": item_out["carry_memory"],
            "u_eta2": user_out["u_eta2"],
            "u_eta3": user_out["u_eta3"],
            "u_actual_ratio": user_out["u_actual_ratio"],
            "u_gate_mean": user_out["u_gate"].mean(),
            "u_systematic_norm": user_out["u_systematic_norm"],
            "u_individual_norm": user_out["u_individual_norm"],
            "i_beta2": item_out["i_beta2"],
            "i_beta3": item_out["i_beta3"],
            "i_beta4": item_out["i_beta4"],
            "i_actual_ratio": item_out["i_actual_ratio"],
            "i_rho_mean": item_out["i_rho"].mean(),
            "i_systematic_norm": item_out["i_systematic_norm"],
            "i_individual_norm": item_out["i_individual_norm"],
            "i_carry_norm": item_out["i_carry_norm"],
            "carry_omega_mean": item_out["carry_omega_mean"],
            "carry_shock_mean": item_out["carry_shock_mean"],
            "carry_lambda_pos": item_out["carry_lambda_pos"],
            "carry_lambda_neg": item_out["carry_lambda_neg"],
            "msg_omega_mean": msg_out["omega_mean"],
            "msg_gate_ui": msg_out["msg_gate_ui"],
            "msg_gate_fb": msg_out["msg_gate_fb"],
            "msg_time_lambda": msg_out["msg_time_lambda"],
            "msg_current_weight": msg_out["msg_current_weight"],
        }

        if compute_aux:
            out["drift_pred"] = self.drift_head(item_out["v_state"])
            out["carry_pred"] = self.carry_head(item_out["v_state"], item_out["carry_memory"], shock_out["message"])

        return out


# Loss


class TSSSMLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def bpr_loss(self, pos: torch.Tensor, neg: torch.Tensor, neg_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if neg.numel() == 0:
            return torch.tensor(0.0, device=pos.device)
        tau = max(float(self.cfg.BPR_TEMPERATURE), 1e-6)
        x = (neg - pos.unsqueeze(1)) / tau
        x = x.clamp(-30.0, 30.0)
        pair = F.softplus(x)

        if not self.cfg.USE_HARD_NEG_FOCUS:
            if neg_mask is None:
                return pair.mean()
            pair = pair.masked_fill(~neg_mask, 0.0)
            n = neg_mask.sum()
            if n == 0:
                return torch.tensor(0.0, device=pos.device)
            return pair.sum() / n

        hard_tau = max(float(self.cfg.HARD_NEG_TEMPERATURE), 1e-6)
        if neg_mask is None:
            scaled = (neg / hard_tau).clamp(-30.0, 30.0)
            w = F.softmax(scaled, dim=1)
            return (w * pair).sum(dim=1).mean()

        valid = neg_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pos.device)
        neg_v = neg[valid]
        pair_v = pair[valid]
        mask_v = neg_mask[valid]
        neg_w = (neg_v / hard_tau).clamp(-30.0, 30.0)
        neg_w = neg_w.masked_fill(~mask_v, -1e9)
        w = F.softmax(neg_w, dim=1)
        pair_v = pair_v.masked_fill(~mask_v, 0.0)
        return (w * pair_v).sum(dim=1).mean()

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], aux_weight_scale: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        logs: Dict[str, float] = {}
        neg_mask = out.get("neg_mask", batch.get("neg_valid_mask", None))

        loss_rank = self.bpr_loss(out["pos_score"], out["neg_scores"], neg_mask)
        total = self.cfg.W_RANK * loss_rank
        logs["loss_rank"] = float(loss_rank.detach().item())

        if neg_mask is not None and neg_mask.numel() > 0:
            logs["neg_valid_rate"] = float(neg_mask.float().mean().detach().item())

        if aux_weight_scale > 0 and "drift_pred" in out:
            mask = batch["item_future_mask"]
            if mask.any():
                pred = out["drift_pred"][mask]
                tgt = batch["item_future_targets"][mask]
                loss_drift = self.mse(pred, tgt)
                total = total + aux_weight_scale * self.cfg.W_DRIFT * loss_drift
                logs["loss_drift"] = float(loss_drift.detach().item())
            else:
                logs["loss_drift"] = 0.0

            cmask = batch["carry_mask"]
            if cmask.any():
                cpred = out["carry_pred"][cmask]
                ctgt = batch["carry_targets"][cmask]
                loss_carry = self.mse(cpred, ctgt)
                total = total + aux_weight_scale * self.cfg.W_CARRY * loss_carry
                logs["loss_carry"] = float(loss_carry.detach().item())
            else:
                logs["loss_carry"] = 0.0

            rt = batch["reliability_targets"]
            obs_t = rt[:, 0].clamp(0, 1)
            rel_t = rt[:, 1].clamp(0, 1)
            conf_t = rt[:, 2].clamp(0, 1)
            l_obs = self.bce(out["obs_pred"], obs_t)
            l_rel = self.bce(out["rel_pred"], rel_t)
            l_conf = self.bce(out["conflict_pred"], conf_t)
            loss_rel = (l_obs + l_rel + l_conf) / 3.0
            total = total + aux_weight_scale * self.cfg.W_RELIABILITY * loss_rel
            logs["loss_reliability"] = float(loss_rel.detach().item())

        logs["loss_total"] = float(total.detach().item())
        logs["aux_scale"] = float(aux_weight_scale)

        # Monitoring
        logs["u_eta2"] = float(out["u_eta2"].detach().item())
        logs["u_eta3"] = float(out["u_eta3"].detach().item())
        logs["u_ratio"] = float(out["u_actual_ratio"].detach().item())
        logs["u_gate"] = float(out["u_gate_mean"].detach().item())
        logs["u_sys_norm"] = float(out["u_systematic_norm"].detach().item())
        logs["u_ind_norm"] = float(out["u_individual_norm"].detach().item())

        logs["i_beta2"] = float(out["i_beta2"].detach().item())
        logs["i_beta3"] = float(out["i_beta3"].detach().item())
        logs["i_beta4"] = float(out["i_beta4"].detach().item())
        logs["i_ratio"] = float(out["i_actual_ratio"].detach().item())
        logs["i_rho"] = float(out["i_rho_mean"].detach().item())
        logs["i_sys_norm"] = float(out["i_systematic_norm"].detach().item())
        logs["i_ind_norm"] = float(out["i_individual_norm"].detach().item())
        logs["i_carry_norm"] = float(out["i_carry_norm"].detach().item())

        logs["carry_omega"] = float(out["carry_omega_mean"].detach().item())
        logs["carry_shock"] = float(out["carry_shock_mean"].detach().item())
        logs["carry_lambda_pos"] = float(out["carry_lambda_pos"].detach().item())
        logs["carry_lambda_neg"] = float(out["carry_lambda_neg"].detach().item())

        logs["msg_omega"] = float(out["msg_omega_mean"].detach().item())
        logs["msg_gate_ui"] = float(out["msg_gate_ui"].detach().item())
        logs["msg_gate_fb"] = float(out["msg_gate_fb"].detach().item())
        logs["msg_lambda"] = float(out["msg_time_lambda"].detach().item())
        logs["msg_current_w"] = float(out["msg_current_weight"].detach().item())

        return total, logs


# EMA


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def sync_shadow_from_model(self):
        """Make a freshly loaded checkpoint the EMA source for evaluation."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()


# Trainer


class Trainer:
    def __init__(self, cfg: Config, model: TSSSMModel, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = TSSSMLoss(cfg)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=cfg.SCHEDULER_T_MAX, eta_min=cfg.SCHEDULER_ETA_MIN) if cfg.USE_SCHEDULER else None
        self.amp_enabled = bool(cfg.AMP and self.device.type == "cuda")
        self.amp_dtype = self._resolve_amp_dtype()
        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            scaler_device,
            enabled=(self.amp_enabled and self.amp_dtype == torch.float16),
        )
        self.base_lr = cfg.LR
        self.best_val = -1e9
        self.best_epoch = -1
        self.patience_counter = 0
        self.ema = EMA(self.model, decay=cfg.EMA_DECAY) if cfg.USE_EMA else None

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _resolve_amp_dtype(self) -> torch.dtype:
        if not self.amp_enabled:
            return torch.float32
        mode = str(self.cfg.AMP_DTYPE).lower()
        if mode == "auto":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if mode in {"bf16", "bfloat16"}:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float16

    def _autocast_context(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype if self.amp_enabled else torch.float32,
            enabled=self.amp_enabled,
        )

    def _get_aux_scale(self, epoch: int) -> float:
        if epoch <= self.cfg.AUX_WARMUP_EPOCHS:
            return 0.0
        if epoch <= self.cfg.AUX_WARMUP_EPOCHS + self.cfg.AUX_RAMPUP_EPOCHS:
            return float(epoch - self.cfg.AUX_WARMUP_EPOCHS) / float(self.cfg.AUX_RAMPUP_EPOCHS)
        return 1.0

    def _adjust_lr_warmup(self, epoch: int):
        if epoch <= self.cfg.WARMUP_EPOCHS:
            factor = float(epoch) / max(1.0, float(self.cfg.WARMUP_EPOCHS))
            cur_lr = self.base_lr * factor
            for pg in self.opt.param_groups:
                pg["lr"] = cur_lr

    def _adjust_update_ratio(self, epoch: int):
        warm = self.cfg.UPDATE_RATIO_WARMUP_EPOCHS
        if epoch <= warm:
            new_ratio = self.cfg.MAX_UPDATE_RATIO
            new_item_ratio = self.cfg.ITEM_MAX_UPDATE_RATIO
        else:
            progress = min(1.0, (epoch - warm) / max(1, self.cfg.EPOCHS - warm))
            new_ratio = self.cfg.MAX_UPDATE_RATIO + progress * (self.cfg.MAX_UPDATE_RATIO_FINAL - self.cfg.MAX_UPDATE_RATIO)
            new_item_ratio = self.cfg.ITEM_MAX_UPDATE_RATIO + progress * (
                self.cfg.ITEM_MAX_UPDATE_RATIO_FINAL - self.cfg.ITEM_MAX_UPDATE_RATIO
            )
        self.model.user_updater.cfg.MAX_UPDATE_RATIO = float(new_ratio)
        self.model.item_updater.cfg.ITEM_MAX_UPDATE_RATIO = float(new_item_ratio)
        return float(new_ratio), float(new_item_ratio)

    def _adjust_temperature_schedule(self, epoch: int) -> Tuple[float, float]:
        warm = max(1, int(self.cfg.TEMP_SCHEDULE_WARMUP_EPOCHS))
        if epoch <= warm:
            bpr_temp = float(self.cfg.BPR_TEMP_INIT)
            hard_temp = float(self.cfg.HARD_NEG_TEMP_INIT)
        else:
            progress = min(1.0, (epoch - warm) / max(1, self.cfg.EPOCHS - warm))
            bpr_temp = float(self.cfg.BPR_TEMP_INIT + progress * (self.cfg.BPR_TEMP_FINAL - self.cfg.BPR_TEMP_INIT))
            hard_temp = float(
                self.cfg.HARD_NEG_TEMP_INIT + progress * (self.cfg.HARD_NEG_TEMP_FINAL - self.cfg.HARD_NEG_TEMP_INIT)
            )
        self.cfg.BPR_TEMPERATURE = bpr_temp
        self.cfg.HARD_NEG_TEMPERATURE = hard_temp
        return bpr_temp, hard_temp

    def _should_validate(self, epoch: int) -> bool:
        if epoch <= self.cfg.FAST_VAL_EARLY_EPOCHS:
            return epoch % max(1, self.cfg.FAST_VAL_EARLY_INTERVAL) == 0
        return epoch % self.cfg.VAL_EVERY == 0

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
    ):
        if self.ema is not None:
            self.ema.apply_shadow()
        carryover_stats = self.model.get_carryover_stats()
        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "best_val": float(val_metrics.get("Recall@20", 0.0)),
                "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                "cfg": vars(self.cfg),
                "carryover_stats": carryover_stats,
            },
            self.cfg.BEST_CKPT_PATH,
        )
        if self.ema is not None:
            self.ema.restore()
        return carryover_stats

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self._adjust_lr_warmup(epoch)
        ratio_u, ratio_i = self._adjust_update_ratio(epoch)
        bpr_temp, hard_temp = self._adjust_temperature_schedule(epoch)
        aux_scale = self._get_aux_scale(epoch)
        compute_aux = aux_scale > 0

        self.model.train()
        logs_accum: Dict[str, float] = {}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            b = self._to_device(batch)
            with self._autocast_context():
                out = self.model(b, compute_aux=compute_aux)
                loss, logs = self.loss_fn(out, b, aux_weight_scale=aux_scale)

            if not torch.isfinite(loss).item():
                print(
                    f"\n[WARN] Non-finite loss at epoch={epoch} step={step}; skipping batch. "
                    f"loss={loss.detach().item()}"
                )
                self.opt.zero_grad(set_to_none=True)
                continue

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.GRAD_CLIP > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update()

            for k, v in logs.items():
                logs_accum[k] = logs_accum.get(k, 0.0) + float(v)
            n_batches += 1
            pbar.set_postfix(
                {
                    "loss": f"{logs.get('loss_total', 0.0):.4f}",
                    "u_ratio": f"{logs.get('u_ratio', 0.0):.3f}",
                    "i_ratio": f"{logs.get('i_ratio', 0.0):.3f}",
                }
            )

        if self.scheduler is not None:
            self.scheduler.step()

        avg = {k: v / max(1, n_batches) for k, v in logs_accum.items()}
        avg["schedule_u_ratio_cap"] = ratio_u
        avg["schedule_i_ratio_cap"] = ratio_i
        avg["bpr_temp"] = bpr_temp
        avg["hard_temp"] = hard_temp
        return avg

    @torch.no_grad()
    def evaluate_sampled(self, loader: DataLoader, num_neg: int, max_batches: int) -> Dict[str, float]:
        if self.ema is not None:
            self.ema.apply_shadow()
        self.model.eval()
        ranks = []
        neg_valid_rates = []

        for bi, batch in enumerate(tqdm(loader, desc="Eval (sampled)", leave=False)):
            if bi >= max_batches:
                break
            b = self._to_device(batch)
            with self._autocast_context():
                current = self.model.encode_current_conditioning(b)
                user_deviation = current["user_deviation"]
                shock_out = current["shock_out"]
                rel_out = current["rel_out"]
                user_out = self.model.encode_user(b, deviation=user_deviation)
                item_state_cache = self.model._build_forward_item_state_cache(b)
                msg_out = self.model.build_query_user_state(
                    b,
                    user_out,
                    shock_out,
                    rel_out,
                    deviation=user_deviation,
                    item_state_cache=item_state_cache,
                )
                z = msg_out["z_final"]

                pos_items = b["target_item"]
                pos_scores = self.model.score_point(
                    z,
                    pos_items,
                    b["target_bin"],
                    item_state_cache=item_state_cache,
                )
                pos_scores = torch.where(torch.isnan(pos_scores), torch.full_like(pos_scores, float("-inf")), pos_scores)

            B = pos_items.size(0)
            sample_min = 1 if self.model.pad_idx == 0 else 0
            sample_size = num_neg * 5
            cand = torch.randint(sample_min, self.model.num_items, (B, sample_size), device=self.device)
            valid = torch.ones(B, sample_size, dtype=torch.bool, device=self.device)
            valid &= (cand != pos_items.unsqueeze(1))
            valid &= (cand != self.model.pad_idx)

            seen_items = b["user_seen_items"]
            seen_mask = b["user_seen_mask"]
            if seen_mask.any():
                S = seen_items.size(1)
                blk = self.cfg.SEEN_BLOCK_SIZE
                is_seen_any = torch.zeros(B, sample_size, dtype=torch.bool, device=self.device)
                for s in range(0, S, blk):
                    e = min(S, s + blk)
                    seen_blk = seen_items[:, s:e]
                    mask_blk = seen_mask[:, s:e]
                    if not mask_blk.any():
                        continue
                    m = (cand.unsqueeze(2) == seen_blk.unsqueeze(1)) & mask_blk.unsqueeze(1)
                    is_seen_any |= m.any(dim=2)
                valid &= ~is_seen_any

            sort_key = torch.where(
                valid,
                torch.arange(sample_size, device=self.device).unsqueeze(0).expand(B, -1),
                torch.full((B, sample_size), sample_size + 1, device=self.device),
            )
            idx = sort_key.argsort(dim=1)[:, :num_neg]
            neg_items = cand.gather(1, idx)
            neg_valid = valid.gather(1, idx)
            with self._autocast_context():
                neg_scores = self.model.score_pair(z, neg_items, b["target_bin"])
                neg_scores = torch.where(torch.isnan(neg_scores), torch.full_like(neg_scores, float("-inf")), neg_scores)
                neg_scores = neg_scores.masked_fill(~neg_valid, float("-inf"))

            scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            rank = (scores[:, 1:] > scores[:, :1]).sum(dim=1) + 1
            ranks.extend(rank.detach().cpu().tolist())
            neg_valid_rates.append(float(neg_valid.float().mean().item()))

        if self.ema is not None:
            self.ema.restore()

        m = self._metrics_from_ranks(ranks)
        m["neg_valid_rate_eval"] = float(np.mean(neg_valid_rates)) if len(neg_valid_rates) > 0 else 1.0
        return m

    @torch.no_grad()
    def evaluate_full_sort(self, loader: DataLoader, max_batches: Optional[int]) -> Dict[str, float]:
        if max_batches is not None and int(max_batches) <= 0:
            max_batches = None
        if self.ema is not None:
            self.ema.apply_shadow()
        self.model.eval()
        ranks = []
        num_items = self.model.num_items
        chunk_size = self.cfg.FULL_SORT_CHUNK_SIZE
        pad_idx = self.model.pad_idx
        for bi, batch in enumerate(tqdm(loader, desc="Eval (full-sort)", leave=False)):
            if max_batches is not None and bi >= max_batches:
                break
            b = self._to_device(batch)
            with self._autocast_context():
                current = self.model.encode_current_conditioning(b)
                user_deviation = current["user_deviation"]
                shock_out = current["shock_out"]
                rel_out = current["rel_out"]
                user_out = self.model.encode_user(b, deviation=user_deviation)
                item_state_cache = self.model._build_forward_item_state_cache(b)
                msg_out = self.model.build_query_user_state(
                    b,
                    user_out,
                    shock_out,
                    rel_out,
                    deviation=user_deviation,
                    item_state_cache=item_state_cache,
                )
                z = msg_out["z_final"]

                pos_items = b["target_item"]
                pos_scores = self.model.score_point(
                    z,
                    pos_items,
                    b["target_bin"],
                    item_state_cache=item_state_cache,
                )
                pos_scores = torch.where(torch.isnan(pos_scores), torch.full_like(pos_scores, float("-inf")), pos_scores)

            B = pos_items.size(0)
            seen_items = b["user_seen_items"]
            seen_mask = b["user_seen_mask"]
            higher = torch.zeros(B, dtype=torch.long, device=self.device)

            for st in range(0, num_items, chunk_size):
                ed = min(st + chunk_size, num_items)
                ids = torch.arange(st, ed, device=self.device)
                sc = self.model.score_chunk(z, ids, b["target_bin"])
                sc = torch.where(torch.isnan(sc), torch.full_like(sc, float("-inf")), sc)

                exclude = torch.zeros(B, ed - st, dtype=torch.bool, device=self.device)
                if st <= pad_idx < ed:
                    exclude[:, pad_idx - st] = True
                pos_in = (pos_items >= st) & (pos_items < ed)
                if pos_in.any():
                    bi_idx = torch.where(pos_in)[0]
                    cj = pos_items[pos_in] - st
                    exclude[bi_idx, cj] = True

                seen_in = (seen_items >= st) & (seen_items < ed) & seen_mask
                if seen_in.any():
                    bi_idx, sj = torch.where(seen_in)
                    cj = seen_items[bi_idx, sj] - st
                    exclude[bi_idx, cj] = True

                sc[exclude] = float("-inf")
                higher += (sc > pos_scores.unsqueeze(1)).sum(dim=1)

            ranks.extend((higher + 1).cpu().tolist())

        if self.ema is not None:
            self.ema.restore()
        return self._metrics_from_ranks(ranks)

    def _metrics_from_ranks(self, ranks: List[int]) -> Dict[str, float]:
        r = np.asarray(ranks, dtype=np.int64)
        if r.size == 0:
            out = {}
            for k in self.cfg.K_VALUES:
                out[f"Recall@{k}"] = 0.0
                out[f"NDCG@{k}"] = 0.0
            out["MRR"] = 0.0
            return out

        out = {}
        for K in self.cfg.K_VALUES:
            hit = (r <= K).astype(np.float64)
            out[f"Recall@{K}"] = float(hit.mean())
            out[f"NDCG@{K}"] = float(np.where(r <= K, 1.0 / np.log2(r.astype(np.float64) + 1.0), 0.0).mean())
        out["MRR"] = float((1.0 / r.astype(np.float64)).mean())
        return out

    def train(self):
        os.makedirs(self.cfg.RESULTS_DIR, exist_ok=True)
        print("=" * 72)
        print("Training TS-SSM")
        print("=" * 72)
        print(f"User update ratio schedule: {self.cfg.MAX_UPDATE_RATIO:.2f} -> {self.cfg.MAX_UPDATE_RATIO_FINAL:.2f}")
        print(f"Item update ratio schedule: {self.cfg.ITEM_MAX_UPDATE_RATIO:.2f} -> {self.cfg.ITEM_MAX_UPDATE_RATIO_FINAL:.2f}")
        print(f"Hard-negative focus: {self.cfg.USE_HARD_NEG_FOCUS} (tau={self.cfg.HARD_NEG_TEMPERATURE})")
        print(
            "Temperature schedule: "
            f"BPR {self.cfg.BPR_TEMP_INIT:.2f}->{self.cfg.BPR_TEMP_FINAL:.2f}, "
            f"hard-neg {self.cfg.HARD_NEG_TEMP_INIT:.2f}->{self.cfg.HARD_NEG_TEMP_FINAL:.2f}, "
            f"warmup={self.cfg.TEMP_SCHEDULE_WARMUP_EPOCHS}"
        )
        print(f"EMA: {self.cfg.USE_EMA} (decay={self.cfg.EMA_DECAY})")
        if self.amp_enabled:
            print(f"AMP runtime: {str(self.amp_dtype).replace('torch.', '')}")

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            train_logs = self.train_epoch(epoch)
            dt = time.time() - t0
            lr = self.opt.param_groups[0]["lr"]
            aux_scale = self._get_aux_scale(epoch)
            print(
                f"\n[Epoch {epoch}] time={dt:.1f}s lr={lr:.2e} aux_scale={aux_scale:.2f} "
                f"bpr_t={train_logs.get('bpr_temp', 0.0):.3f} hard_t={train_logs.get('hard_temp', 0.0):.3f}"
            )
            print(
                f"  user: eta2={train_logs.get('u_eta2', 0):.3f} eta3={train_logs.get('u_eta3', 0):.3f} "
                f"ratio={train_logs.get('u_ratio', 0):.3f} gate={train_logs.get('u_gate', 0):.3f}"
            )
            print(
                f"  item: beta2={train_logs.get('i_beta2', 0):.3f} beta3={train_logs.get('i_beta3', 0):.3f} "
                f"beta4={train_logs.get('i_beta4', 0):.3f} ratio={train_logs.get('i_ratio', 0):.3f}"
            )
            print(
                f"  carry/msg: omega={train_logs.get('carry_omega', 0):.3f}/{train_logs.get('msg_omega', 0):.3f} "
                f"lambdas={train_logs.get('carry_lambda_pos', 0):.3f},{train_logs.get('carry_lambda_neg', 0):.3f}"
            )
            print("  Losses: " + " ".join([f"{k}={v:.4f}" for k, v in train_logs.items() if k.startswith("loss_")]))

            if not self._should_validate(epoch):
                continue
            val_max_batches = None if int(self.cfg.FULL_SORT_VAL_BATCHES) <= 0 else int(self.cfg.FULL_SORT_VAL_BATCHES)
            val_metrics = self.evaluate_full_sort(self.val_loader, max_batches=val_max_batches)
            val_scope = "all" if val_max_batches is None else str(val_max_batches)
            print(
                f"  Val (full-sort, {val_scope} batches): "
                + " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
            )
            recall20 = float(val_metrics.get("Recall@20", 0.0))
            if recall20 > self.best_val:
                self.best_val = recall20
                self.best_epoch = epoch
                self.patience_counter = 0
                carryover_stats = self._save_checkpoint(epoch, val_metrics)
                print(
                    f"  Saved best checkpoint (epoch={epoch}, Recall@20={recall20:.4f}; "
                    f"carry half-life bins="
                    f"{carryover_stats['half_life_pos_bins']:.3f}/"
                    f"{carryover_stats['half_life_neg_bins']:.3f})"
                )
            else:
                self.patience_counter += 1
                print(
                    f"  No Recall@20 improvement ({self.patience_counter}/"
                    f"{self.cfg.EARLY_STOP_PATIENCE}); best epoch={self.best_epoch}"
                )
                if self.patience_counter >= self.cfg.EARLY_STOP_PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        self._final_test()

    def _final_test(self):
        print("\n" + "=" * 60)
        print("Final Test")
        print("=" * 60)
        if os.path.exists(self.cfg.BEST_CKPT_PATH):
            ckpt = torch.load(self.cfg.BEST_CKPT_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            if self.ema is not None:
                self.ema.sync_shadow_from_model()
            self.best_epoch = int(ckpt.get("epoch", -1))
            self.best_val = float(ckpt.get("best_val", self.best_val))
            print(
                f"Loaded best checkpoint (epoch={ckpt.get('epoch', '?')}, "
                f"val_R@20={ckpt.get('best_val', 0):.4f})"
            )

        sampled = self.evaluate_sampled(self.test_loader, self.cfg.SAMPLED_EVAL_NEGATIVES, max_batches=10000)
        neg_valid = sampled.pop("neg_valid_rate_eval", 1.0)
        print("[Sampled] " + " ".join([f"{k}={v:.4f}" for k, v in sampled.items()]) + f" neg_valid={neg_valid:.3f}")
        full = self.evaluate_full_sort(self.test_loader, max_batches=None)
        print("[Full-sort] " + " ".join([f"{k}={v:.4f}" for k, v in full.items()]))
        carryover_stats = self.model.get_carryover_stats()
        print(
            "[Carryover learned] "
            f"lambda/bin={carryover_stats['lambda_pos_per_bin']:.6f}/"
            f"{carryover_stats['lambda_neg_per_bin']:.6f} "
            f"half-life bins={carryover_stats['half_life_pos_bins']:.3f}/"
            f"{carryover_stats['half_life_neg_bins']:.3f}"
        )

        os.makedirs(self.cfg.RESULTS_DIR, exist_ok=True)
        result_path = os.path.join(self.cfg.RESULTS_DIR, "test_results.json")
        with open(result_path, "w") as f:
            json.dump(
                {
                    "test_sampled": sampled,
                    "test_full_sort": full,
                    "neg_valid_rate_eval": neg_valid,
                    "best_val": self.best_val,
                    "best_epoch": self.best_epoch,
                    "carryover_learned": carryover_stats,
                    "config": {
                        "MAX_UPDATE_RATIO": self.cfg.MAX_UPDATE_RATIO,
                        "MAX_UPDATE_RATIO_FINAL": self.cfg.MAX_UPDATE_RATIO_FINAL,
                        "ITEM_MAX_UPDATE_RATIO": self.cfg.ITEM_MAX_UPDATE_RATIO,
                        "ITEM_MAX_UPDATE_RATIO_FINAL": self.cfg.ITEM_MAX_UPDATE_RATIO_FINAL,
                        "BPR_TEMPERATURE": self.cfg.BPR_TEMPERATURE,
                        "BPR_TEMP_INIT": self.cfg.BPR_TEMP_INIT,
                        "BPR_TEMP_FINAL": self.cfg.BPR_TEMP_FINAL,
                        "USE_HARD_NEG_FOCUS": self.cfg.USE_HARD_NEG_FOCUS,
                        "HARD_NEG_TEMPERATURE": self.cfg.HARD_NEG_TEMPERATURE,
                        "HARD_NEG_TEMP_INIT": self.cfg.HARD_NEG_TEMP_INIT,
                        "HARD_NEG_TEMP_FINAL": self.cfg.HARD_NEG_TEMP_FINAL,
                        "TEMP_SCHEDULE_WARMUP_EPOCHS": self.cfg.TEMP_SCHEDULE_WARMUP_EPOCHS,
                        "USE_MIXED_NEG_SAMPLING": self.cfg.USE_MIXED_NEG_SAMPLING,
                        "UNIFORM_NEG_RATIO": self.cfg.UNIFORM_NEG_RATIO,
                        "USE_EMA": self.cfg.USE_EMA,
                        "EMA_DECAY": self.cfg.EMA_DECAY,
                        "W_DRIFT": self.cfg.W_DRIFT,
                        "W_CARRY": self.cfg.W_CARRY,
                        "W_RELIABILITY": self.cfg.W_RELIABILITY,
                    },
                },
                f,
                indent=2,
            )
        print(f"Results saved to {result_path}")


# Main


def main():
    parser = argparse.ArgumentParser(description="TS-SSM training/evaluation")
    parser.add_argument("--data_dir", type=str, default="./preprocessed_sequential")
    parser.add_argument("--results_dir", type=str, default="./results_ts_ssm")
    parser.add_argument("--best_ckpt_path", type=str, default="./results_ts_ssm/best.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val_max_batches",
        type=int,
        default=0,
        help="Full-sort validation batches; <=0 evaluates the complete split.",
    )
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    args = parser.parse_args()

    cfg = Config()
    cfg.DATA_DIR = args.data_dir
    cfg.RESULTS_DIR = args.results_dir
    cfg.BEST_CKPT_PATH = args.best_ckpt_path
    cfg.EPOCHS = int(args.epochs)
    cfg.BATCH_SIZE = int(args.batch_size)
    cfg.NUM_WORKERS = int(args.num_workers)
    cfg.LR = float(args.lr)
    cfg.SEED = int(args.seed)
    cfg.FULL_SORT_VAL_BATCHES = int(args.val_max_batches)
    cfg.VAL_EVERY = max(1, int(args.val_every))
    cfg.FAST_VAL_EARLY_INTERVAL = cfg.VAL_EVERY
    cfg.EARLY_STOP_PATIENCE = max(1, int(args.early_stop_patience))

    if hasattr(os, "sched_getaffinity"):
        available_cpus = len(os.sched_getaffinity(0))
    else:
        available_cpus = os.cpu_count() or cfg.NUM_WORKERS or 1
    if cfg.PRELOAD_HOT_DATA:
        tuned_workers = min(cfg.NUM_WORKERS, max(0, min(8, available_cpus // 2)))
    else:
        tuned_workers = min(cfg.NUM_WORKERS, max(0, min(8, available_cpus)))
    if tuned_workers != cfg.NUM_WORKERS:
        print(f"[Config] Adjusting num_workers from {cfg.NUM_WORKERS} to {tuned_workers} for stable throughput")
        cfg.NUM_WORKERS = tuned_workers

    set_seed(cfg.SEED)
    print("=" * 72)
    print("TS-SSM")
    print("=" * 72)
    print(f"Device: {cfg.DEVICE}")
    print(f"Data dir: {cfg.DATA_DIR}")
    print(f"AMP: {cfg.AMP} ({cfg.AMP_DTYPE})")

    print("\n[1/4] Loading shared resources...")
    shared = SharedResources(cfg)
    cfg.N_GROUPS = shared.n_groups
    cfg.NUM_BINS = shared.num_bins
    cfg.PAD_IDX = shared.pad_idx
    cfg.N_ITEM_GROUPS = shared.n_item_groups

    print("\n[2/4] Creating datasets...")
    train_ds = TSSSMDataset(
        shared=shared,
        split="train",
        max_user_len=cfg.MAX_USER_SEQ_LEN,
        max_item_len=cfg.MAX_ITEM_SEQ_LEN,
        num_negatives=cfg.NUM_NEGATIVES_TRAIN,
        return_user_seen=False,
        cfg=cfg,
    )
    val_ds = TSSSMDataset(
        shared=shared,
        split="val",
        max_user_len=cfg.MAX_USER_SEQ_LEN,
        max_item_len=cfg.MAX_ITEM_SEQ_LEN,
        num_negatives=0,
        return_user_seen=True,
        cfg=cfg,
    )
    test_ds = TSSSMDataset(
        shared=shared,
        split="test",
        max_user_len=cfg.MAX_USER_SEQ_LEN,
        max_item_len=cfg.MAX_ITEM_SEQ_LEN,
        num_negatives=0,
        return_user_seen=True,
        cfg=cfg,
    )
    print(f"  Train: {len(train_ds):,}")
    print(f"  Val: {len(val_ds):,}")
    print(f"  Test: {len(test_ds):,}")

    loader_kwargs = {
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": cfg.PIN_MEMORY,
        "collate_fn": collate_fn,
        "drop_last": False,
    }
    if cfg.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 1

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, **loader_kwargs)

    print("\n[3/4] Building model...")
    model = TSSSMModel(shared.num_items, cfg.PAD_IDX, cfg, item_group_arr=shared.item_group_arr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    print("\n[4/4] Training...")
    trainer = Trainer(cfg, model, train_loader, val_loader, test_loader)
    trainer.train()
    print("\nComplete.")


if __name__ == "__main__":
    main()
