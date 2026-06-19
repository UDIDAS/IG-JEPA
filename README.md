# IG-JEPA: Image-Graph Joint Embedding Predictive Architecture

Self-supervised representation learning on graph-structured image decompositions. Images are converted to superpixel graphs via Rust-accelerated graph-minor pooling, enriched with frozen DINO ViT-S/16 patch embeddings, then trained with a graph JEPA objective featuring per-graph BFS subgraph masking, context-neighbor prediction, and graph-level BYOL alignment.

## Pipeline Overview

```
Raw Image (H x W x 3)
    |
    v
[1] Graph-Minor Pooling (Rust: fastloops)
    - Flood-fill merge similar pixels
    - Cut dissimilar boundaries
    - Delete too-small / too-large regions
    -> Superpixel graph (~800 nodes for 96x96)
    |
    v
[2] DINO Feature Extraction (frozen ViT-S/16)
    - Image -> 14x14 patch grid (384-dim each)
    - Each superpixel node: centroid -> nearest DINO patch
    |
    v
[3] Node Features (398-dim per node)
    - 14-dim boundary: RGB mean/std, log(area), centroid,
      bbox dims, compactness, boundary ratio, relative area
    - 384-dim DINO: semantic patch embedding
    |
    v
[4] IG-JEPA Self-Supervised Training
    - Teacher (EMA): encodes full clean graph
    - Student: encodes masked + augmented graph
    - Per-graph BFS masking: connected subgraph (40% nodes)
    - Context-neighbor prediction (no info leak)
    - Graph-level BYOL + VICReg regularization
    |
    v
[5] Evaluation
    - Freeze encoder, extract graph-level embeddings (mean pool)
    - Linear probe (LogReg) / MLP probe on labeled data
    - Label efficiency at 1%, 2%, 5%, 10%, 20%, 50%, 100%
```

## Architecture

**GraphTransformerEncoder** (3.26M trainable params):
- Input projection: 398 -> 384
- 4 layers of multi-head TransformerConv (4 heads, 96 dim/head)
- LayerNorm + residual per layer
- Global residual: output = encoder(x) + proj(x)  (preserves input features)

**IG-JEPA Training Objective** (3 losses):

| Loss | Weight | Description |
|------|:------:|-------------|
| Prediction | 1.0 | Student predicts teacher's masked node embeddings from context neighbor aggregation |
| BYOL | 1.0 | Student graph embedding (via predictor) matches teacher graph embedding |
| VICReg Variance | 25.0 | Prevents embedding collapse (maintains unit std) |
| VICReg Covariance | 1.0 | Decorrelates embedding dimensions |

**Key Design Decisions:**
- **Per-graph BFS masking**: Each graph in the batch gets its own connected subgraph mask via BFS (not random nodes). This teaches spatial part-whole reasoning.
- **Context-neighbor prediction**: Masked nodes are disconnected from the student graph (no info leak through attention). Predictions come from aggregating context neighbors' embeddings via the original edge structure.
- **Global residual**: Preserves DINO features through the network, preventing information destruction.
- **Frozen DINO**: No fine-tuning of the vision backbone. The graph framework adds structural reasoning on top.

## Project Structure

```
.
├── src/
│   └── run_dino.py              # Complete pipeline: graph build + train + eval
├── fastloops/
│   └── src/lib.rs               # Rust kernel: graph-minor pooling + BFS masking
├── signals/                     # JSON result files from completed experiments
├── archive/
│   ├── v1_src/                  # Previous experiment scripts (10 files)
│   ├── v1_signals/              # Previous result JSONs
│   └── v1_reports/              # Previous technical reports
└── README.md
```

## Setup

```bash
# Python dependencies
pip install torch torchvision torch_geometric
pip install scikit-learn scipy numpy maturin

# Build the Rust kernel
cd fastloops
maturin develop --release
cd ..
```

The `fastloops` Rust module provides:
- `merge_and_cut()`: Graph-minor pooling (image -> superpixel adjacency + features)
- `subgraph_mask()`: BFS-based connected subgraph masking for JEPA training

## Running Experiments

```bash
# STL-10 (standard SSL protocol: 100K unlabeled pretrain, 5K/8K labeled eval)
python src/run_dino.py --dataset stl10 --unlabeled --gpu 0 --epochs 100 --bs 64 --hid 384

# CIFAR-10 (50K train pretrain, 10K test eval)
python src/run_dino.py --dataset cifar10 --gpu 0 --epochs 200 --bs 64 --hid 384

# TinyImageNet (100K train pretrain, 10K test eval)
python src/run_dino.py --dataset tinyimagenet --gpu 1 --epochs 200 --bs 64 --hid 384
```

Graphs are cached to `/scratch/ud3d4/igjepa_cache/` (persistent) on first run. Subsequent runs skip graph construction and go directly to training.

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `cifar10`, `stl10`, or `tinyimagenet` |
| `--gpu` | 0 | CUDA device index |
| `--epochs` | 100 | Training epochs |
| `--hid` | 384 | Hidden dimension (matches DINO patch dim) |
| `--bs` | 32 | Batch size |
| `--n_layers` | 4 | Transformer layers |
| `--n_heads` | 4 | Attention heads |
| `--lr` | 1e-4 | Learning rate (cosine decay to 1e-6) |
| `--unlabeled` | flag | Use STL-10 100K unlabeled split for pretraining |

## Results

### STL-10 (100K unlabeled pretrain, 3.26M params)

| Method | Params | Accuracy | F1 (wtd) |
|--------|:------:|:--------:|:--------:|
| SimCLR (ResNet-50) | 25M | ~85% | - |
| BYOL (ResNet-50) | 25M | ~87% | - |
| Raw DINO + LogReg | - | 89.91% | 89.93% |
| **IG-JEPA + LogReg** | **3.26M** | **94.09%** | **94.10%** |
| IG-JEPA + MLP | 3.26M | 93.17% | - |

### Label Efficiency (STL-10)

| Labels | N | Raw + LogReg | IG-JEPA + LogReg | Gap |
|:------:|----:|:------------:|:----------------:|:---:|
| 1% | 50 | 54.77% | **67.74%** | +12.96% |
| 2% | 100 | 66.62% | **78.22%** | +11.60% |
| 5% | 250 | 76.54% | **85.97%** | +9.44% |
| 10% | 500 | 82.29% | **89.60%** | +7.31% |
| 20% | 1000 | 85.69% | **92.11%** | +6.42% |
| 50% | 2500 | 88.81% | **93.75%** | +4.94% |
| 100% | 5000 | 89.91% | **94.09%** | +4.18% |

JEPA with 500 labels (89.6%) matches raw features with all 5000 labels (89.9%) -- **10x label efficiency**.

### CIFAR-10 (50K train pretrain, 3.26M params)

| Method | Accuracy | F1 (wtd) | Precision | Recall |
|--------|:--------:|:--------:|:---------:|:------:|
| Raw DINO + LogReg | 87.57% | 87.54% | 87.53% | 87.57% |
| **IG-JEPA + LogReg** | **88.27%** | **88.24%** | **88.23%** | **88.27%** |
| IG-JEPA + MLP | 89.46% | - | - | - |

### Label Efficiency (CIFAR-10)

| Labels | N | Raw + LogReg | IG-JEPA + LogReg | Gap |
|:------:|-----:|:------------:|:----------------:|:---:|
| 1% | 500 | 73.72% | **74.06%** | +0.34% |
| 2% | 1000 | 77.64% | **78.79%** | +1.15% |
| 5% | 2500 | 79.68% | **82.17%** | +2.49% |
| 10% | 5000 | 80.46% | **83.78%** | +3.32% |
| 20% | 10000 | 82.51% | **84.91%** | +2.40% |
| 50% | 25000 | 85.97% | **86.94%** | +0.97% |
| 100% | 50000 | 87.57% | **88.27%** | +0.70% |

### TinyImageNet (100K train pretrain, 200 classes, 3.26M params)

| Method | Accuracy | F1 (wtd) | Precision | Recall |
|--------|:--------:|:--------:|:---------:|:------:|
| Raw DINO + LogReg | 57.87% | 57.56% | 57.69% | 57.87% |
| **IG-JEPA + LogReg** | **58.85%** | **58.58%** | **58.84%** | **58.85%** |
| IG-JEPA + MLP | 53.82% | - | - | - |

### Label Efficiency (TinyImageNet)

| Labels | N | Raw + LogReg | IG-JEPA + LogReg | Gap |
|:------:|-----:|:------------:|:----------------:|:---:|
| 1% | 1000 | 25.21% | **25.95%** | +0.74% |
| 2% | 2000 | 33.04% | **33.64%** | +0.60% |
| 5% | 5000 | 40.95% | **41.77%** | +0.82% |
| 10% | 10000 | 45.21% | **46.29%** | +1.08% |
| 20% | 20000 | 47.42% | **49.67%** | +2.25% |
| 50% | 50000 | 51.53% | **55.61%** | +4.08% |
| 100% | 100000 | 57.87% | **58.85%** | +0.98% |

### Summary Across Datasets

| Dataset | Classes | Raw+LR | JEPA+LR | Gap | Best Label Eff. |
|---------|:-------:|:------:|:-------:|:---:|:---------------:|
| STL-10 | 10 | 89.91% | **94.09%** | +4.18% | +12.96% (1%) |
| CIFAR-10 | 10 | 87.57% | **88.27%** | +0.70% | +3.32% (10%) |
| TinyImageNet | 200 | 57.87% | **58.85%** | +0.98% | +4.08% (50%) |

IG-JEPA outperforms raw features on every dataset at every label fraction, with 3.26M trainable parameters.

## Output Format

Results are saved to `signals/done_{dataset}_dino.json`:

```json
{
  "acc_raw": 0.8991,
  "acc_jepa_lr": 0.9409,
  "acc_jepa_mlp": 0.9317,
  "f1_raw": 0.8993,
  "f1_jepa": 0.9410,
  "precision_raw": 0.8997,
  "precision_jepa": 0.9411,
  "recall_raw": 0.8991,
  "recall_jepa": 0.9409,
  "confusion_matrix": [...],
  "label_efficiency": {
    "0.01": {"raw": 0.5477, "jepa": 0.6774, "n": 50},
    ...
  },
  "dataset": "stl10",
  "params": 5782656
}
```

## Graph-Minor Pooling Parameters

| Dataset | Image Size | merge_dist | cut_dist | del_small | del_large | Avg Nodes |
|---------|:----------:|:----------:|:--------:|:---------:|:---------:|:---------:|
| CIFAR-10 | 32x32 | 5 | 80 | 0 | 1024 | ~150 |
| STL-10 | 96x96 | 8 | 80 | 2 | 9216 | ~800 |
| TinyImageNet | 64x64 | 6 | 80 | 1 | 4096 | ~400 |
