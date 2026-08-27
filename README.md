# TS-SSM

_Official implementation of “Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback.”_

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

TS-SSM is an event-conditioned sequential recommendation model for platforms where both user preferences and item states evolve as multimodal review feedback accumulates. It treats review content and modality availability as informative observations, jointly updates user and item representations, and preserves asymmetric carryover from positive and negative review shocks.

## 📋 Paper and authors

**Title:** Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback

**Authors:**

- Ziwen Pan[^equal]
- Zihan Liang[^equal]
- Ruoxuan Xiong

**Affiliation:** Emory University, Atlanta, USA

**Contact:** `{ziwen.pan, zihan.liang, ruoxuan.xiong}@emory.edu`

[^equal]: Equal contribution.

## 🎯 Model overview

TS-SSM combines four main ideas:

- **MNAR multimodal review encoding:** jointly represents rating, title, review text, images, numeric cues, and the availability pattern of each modality
- **Two-sided state evolution:** updates both user and item states using history, train-only temporal context, and current-event innovation
- **Observation-pattern-aware propagation:** weights local user–item–item messages by modality use, reliability cues, signed evidence, and recency
- **Asymmetric carryover memory:** learns separate decay rates for positive and negative item-side review shocks

```mermaid
flowchart LR
    accTitle: TS-SSM Model Pipeline
    accDescr: Review events are preprocessed into multimodal representations, used to update user and item states, propagated through a bounded local graph, and scored against candidate items.

    raw_reviews([📥 Raw review events]) --> preprocess[⚙️ Train-aware preprocessing]
    preprocess --> event_encoding[🧠 Multimodal event encoding]
    event_encoding --> user_state[👤 User-state evolution]
    event_encoding --> item_state[📦 Item-state evolution]
    item_state --> carryover[🔄 Signed carryover memory]
    user_state --> propagation[🔗 Weighted local propagation]
    carryover --> propagation
    propagation --> ranking([📊 Full-catalog ranking])

    classDef data fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef model fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef output fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class raw_reviews,preprocess data
    class event_encoding,user_state,item_state,carryover,propagation model
    class ranking output
```

The implementation uses frozen `all-MiniLM-L6-v2` encoders for review titles and bodies and CLIP ViT-B/32 for review images. Their event-level embeddings are generated during preprocessing and subsequently projected into the trainable TS-SSM hidden space.[^minilm][^clip]

## 📊 Reported results

The accompanying paper evaluates TS-SSM on six Amazon Reviews 2023 categories and the Goodreads Fantasy subset. The following Recall@20 values are reported under the paper's common full-sort evaluation protocol:

| Dataset | BSARec | HM4SR | TS-SSM |
| ------- | -----: | ----: | -----: |
| Toys & Games | 0.1427 | 0.1489 | **0.1683** |
| Pet Supplies | 0.1500 | 0.1563 | **0.1756** |
| Sports & Outdoors | 0.0838 | 0.0872 | **0.0962** |
| Electronics | 0.1317 | 0.1375 | **0.1530** |
| Clothing | 0.1391 | 0.1461 | **0.1653** |
| Home & Kitchen | 0.1365 | 0.1425 | **0.1570** |
| Goodreads Fantasy | 0.4976 | 0.5191 | **0.5847** |

Across the six Amazon categories, the paper reports relative Recall@20 improvements of 14.8%–18.8% over BSARec and an average improvement of 11.7% over HM4SR. On Goodreads Fantasy, TS-SSM improves Recall@20 by 12.6% relative to HM4SR.

## 🔧 Installation

### Requirements

- Python 3.10 or later
- PyTorch 2.1 or later
- A CUDA-capable GPU is recommended for multimodal preprocessing and training
- Sufficient local storage for Amazon review shards, cached images, embeddings, and preprocessed Parquet files

Create an isolated environment and install the supplied dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file installs NumPy, pandas, PyArrow, Hugging Face Datasets, Sentence Transformers, PyTorch, Pillow, and OpenAI CLIP.

## ⚙️ Data preprocessing

The default configuration downloads `McAuley-Lab/Amazon-Reviews-2023` and processes the `Toys_and_Games` category.[^amazon]

Before running, review the configuration constants near the top of `preprocess.py`, especially:

| Setting | Default | Purpose |
| ------- | ------- | ------- |
| `CATEGORY` | `Toys_and_Games` | Amazon category configuration |
| `OUTPUT_DIR` | `./preprocessed_sequential` | Preprocessed dataset directory |
| `MIN_YEAR` | `2014` | Earliest retained review year |
| `MIN_USER_REVIEWS` | `10` | Iterative user-side core threshold |
| `MIN_ITEM_REVIEWS` | `10` | Iterative item-side core threshold |
| `USE_TEXT_EMB` | `True` | Generate title and review-body embeddings |
| `USE_IMAGE_EMB` | `True` | Download and encode review images |

Run the complete preprocessing pipeline:

```bash
python preprocess.py
```

The pipeline performs required-field and timestamp validation, removes events before 2014, applies iterative bipartite 10-core filtering, constructs chronological leave-last-two-out splits, extracts multimodal embeddings, and builds train-only temporal statistics, user/item groups, item histories, item neighbors, and auxiliary targets.

## 🚀 Training and evaluation

Train TS-SSM with the preprocessed data:

```bash
python ts-ssm.py \
  --data_dir ./preprocessed_sequential \
  --results_dir ./results_ts_ssm \
  --best_ckpt_path ./results_ts_ssm/best.pt \
  --epochs 50 \
  --batch_size 128 \
  --lr 5e-4 \
  --seed 42
```

Important command-line options:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--data_dir` | `./preprocessed_sequential` | Preprocessed dataset directory |
| `--results_dir` | `./results_ts_ssm` | Metrics and result directory |
| `--best_ckpt_path` | `./results_ts_ssm/best.pt` | Best-checkpoint path |
| `--epochs` | `50` | Maximum training epochs |
| `--batch_size` | `128` | Training batch size |
| `--num_workers` | `4` | Requested data-loader workers |
| `--lr` | `5e-4` | AdamW learning rate |
| `--seed` | `42` | Random seed |
| `--val_max_batches` | `0` | Full validation when non-positive |
| `--early_stop_patience` | `6` | Validation patience |

## 📚 Evaluation protocol

The implementation follows the protocol described in the paper:

1. Retain reviews from January 2014 onward
2. Apply iterative bipartite 10-core filtering
3. Sort each user history chronologically
4. Use the penultimate event for validation and the final event for testing
5. Build groups, temporal summaries, item graphs, and auxiliary targets from training events only
6. Select checkpoints by full-sort validation Recall@20
7. Filter padding and previously observed items during full-sort ranking while retaining a separately computed ground-truth score

The default model uses a maximum user history of 64 events, item history of 40 reviews, hidden dimension 320, 48 mixed negatives per training query, and a bounded item graph with 20 neighbors from a co-occurrence window of three interactions.

## 📦 Repository structure

| File | Description |
| ---- | ----------- |
| `preprocess.py` | Amazon review download, filtering, multimodal feature extraction, splitting, and train-only resource construction |
| `ts-ssm.py` | TS-SSM dataset loader, model, objectives, training loop, checkpointing, and evaluation |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |

## 🔗 License

This project is released under the [MIT License](./LICENSE).

[^minilm]: Sentence Transformers. “all-MiniLM-L6-v2.” https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[^clip]: OpenAI. “CLIP.” https://github.com/openai/CLIP

[^amazon]: McAuley Lab. “Amazon Reviews 2023.” https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023