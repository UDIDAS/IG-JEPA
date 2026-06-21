# IG-JEPA: Image-Graph Joint Embedding Predictive Architecture

Self-supervised representation learning on graph-structured image decompositions. Images are converted to superpixel graphs via Rust-accelerated graph-minor pooling, enriched with engineered pixel features (color, texture, shape, gradient), then trained with a graph JEPA objective featuring per-graph BFS subgraph masking, context-neighbor prediction, and graph-level BYOL alignment.

**No pretrained models are used.** All features are derived from raw pixels — the same starting materials available to SimCLR, BYOL, and other SSL benchmarks.

## Pipeline Overview

```
Raw Image (H x W x 3)
    |
    v
[1] Graph-Minor Pooling (Rust: fastloops)
    - Flood-fill merge similar pixels
    - Cut dissimilar boundaries
    - Delete too-small / too-large regions
    -> Superpixel graph (~150-800 nodes depending on resolution)
    |
    v
[2] Pixel Feature Engineering (72-dim per node, NO pretrained models)
    - Color: RGB mean/std (6), HSV mean/std (6), grayscale mean/std (2)
    - Geometry: log(area), centroid, bbox dims, compactness, relative area (8)
    - Higher-order color: RGB skewness + kurtosis (6)
    - Texture: grayscale histogram 16 bins (16)
    - Gradient: magnitude mean/std + 6-bin direction histogram (8)
    - Spatial: 4x4 grid position encoding (16)
    - Shape: 2nd order moments Ixx, Iyy, Ixy, magnitude (4)
    |
    v
[3] IG-JEPA Self-Supervised Training
    - Teacher (EMA): encodes full clean graph
    - Student: encodes masked + augmented graph
    - Per-graph BFS masking: connected subgraph (40% nodes)
    - Context-neighbor prediction (no info leak)
    - Graph-level BYOL + VICReg regularization
    |
    v
[4] Evaluation
    - Freeze encoder, extract graph-level embeddings (mean pool)
    - Linear probe (LogReg) / MLP probe on labeled data
    - Label efficiency at 1%, 2%, 5%, 10%, 20%, 50%, 100%
    - Detailed metrics: accuracy, F1, precision, recall, confusion matrix
```

## Architecture

**GraphTransformerEncoder** (~1.5M trainable params with hid=256):
- Input projection: 72 -> 256
- 4 layers of multi-head TransformerConv (4 heads, 64 dim/head)
- LayerNorm + residual per layer
- Global residual: output = encoder(x) + proj(x) (preserves input features)

**IG-JEPA Training Objective** (3 losses):

| Loss | Weight | Description |
|------|:------:|-------------|
| Prediction | 1.0 | Student predicts teacher's masked node embeddings from context neighbor aggregation |
| BYOL | 1.0 | Student graph embedding (via predictor) matches teacher graph embedding |
| VICReg Variance | 25.0 | Prevents embedding collapse (maintains unit std) |
| VICReg Covariance | 1.0 | Decorrelates embedding dimensions |

**Key Design Decisions:**
- **Per-graph BFS masking**: Each graph in the batch gets its own connected subgraph mask via BFS. This teaches spatial part-whole reasoning — "given the head and body, predict the tail."
- **Context-neighbor prediction**: Masked nodes are disconnected from the student graph (no info leak through attention). Predictions come from aggregating context neighbors' embeddings via the original edge structure.
- **Global residual**: Preserves input features through the network, preventing information destruction.
- **No pretrained models**: All node features are engineered from raw pixels. Fair comparison with SSL benchmarks.

## Project Structure

```
.
├── src/
│   └── run_dino.py              # Complete pipeline: graph build + train + eval
├── fastloops/
│   ├── src/lib.rs               # Rust: graph-minor pooling + BFS masking
│   ├── Cargo.toml
│   └── Cargo.lock
├── signals/                     # JSON result files from completed experiments
├── archive/
│   ├── v1_src/                  # Previous experiment scripts (v1 with DINO features)
│   └── v1_signals/              # Previous result JSONs
├── README.md
└── .gitignore
```

## Setup

```bash
# Python dependencies
pip install torch torchvision torch_geometric
pip install scikit-learn scipy numpy maturin

# Build the Rust kernel (required)
cd fastloops
maturin develop --release
cd ..

# IMPORTANT: If an old fastloops package (v0.3.0) is installed, remove it first:
# pip uninstall fastloops -y && cd fastloops && maturin develop --release
```

The `fastloops` Rust module provides:
- `merge_and_cut()`: Graph-minor pooling (image -> superpixel adjacency + node features)
- `subgraph_mask()`: BFS-based connected subgraph masking for JEPA training

## Running Experiments

```bash
# STL-10 (standard SSL protocol: 100K unlabeled pretrain, 5K/8K labeled eval)
python src/run_dino.py --dataset stl10 --unlabeled --gpu 0 --epochs 100 --bs 64

# CIFAR-10 (50K train pretrain, 10K test eval)
python src/run_dino.py --dataset cifar10 --gpu 0 --epochs 200 --bs 64

# TinyImageNet (100K train pretrain, 10K test eval)
python src/run_dino.py --dataset tinyimagenet --gpu 1 --epochs 200 --bs 64
```

Graphs are cached on first run. Subsequent runs skip graph construction.

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `cifar10`, `stl10`, or `tinyimagenet` |
| `--gpu` | 0 | CUDA device index |
| `--epochs` | 100 | Training epochs |
| `--hid` | 256 | Hidden dimension |
| `--bs` | 32 | Batch size |
| `--n_layers` | 4 | Transformer layers |
| `--n_heads` | 4 | Attention heads |
| `--lr` | 1e-4 | Learning rate (cosine decay to 1e-6) |
| `--unlabeled` | flag | Use STL-10 100K unlabeled split for pretraining |

## Results

### Benchmark Context

All published SSL benchmarks below use **ResNet-18** backbones (~11M params) trained from scratch on each dataset. Our method uses a **GraphTransformer** (~1.5M params) with 72-dim pixel-engineered features. No pretrained models (DINO, CLIP, etc.) are used — same starting materials as all compared methods.

### STL-10 (100K unlabeled pretrain, linear probe on 5K/8K labeled)

| Method | Backbone | Params | Accuracy | Source |
|--------|----------|:------:|:--------:|--------|
| DINO | ResNet-18 | 11M | ~82.0% | Caron et al., ICCV 2021 [7] |
| MoCo v2 | ResNet-18 | 11M | ~83.6% | Chen et al., 2020 [2] |
| BYOL | ResNet-18 | 11M | ~88.6% | Grill et al., NeurIPS 2020 [3]; pNNCLR [15] |
| SimCLR | ResNet-18 | 11M | ~89.3% | Chen et al., ICML 2020 [1]; pNNCLR [15] |
| SimSiam | ResNet-18 | 11M | ~90.0% | Chen & He, CVPR 2021 [6]; pNNCLR [15] |
| Raw (72-dim) + LogReg | - | - | 43.16% | Ours (no learning) |
| **IG-JEPA + LogReg** | **GraphTransformer** | **~1.5M** | **49.79%** | **Ours** |
| **IG-JEPA + MLP** | **GraphTransformer** | **~1.5M** | **50.15%** | **Ours** |

**Note:** Our 72-dim hand-crafted features cannot match learned ResNet-18 features. The gap is expected — we use a fixed feature extractor while benchmarks learn hierarchical features end-to-end. See "What We Beat" below.

#### Label Efficiency (STL-10)

| Labels | N | Raw + LogReg | IG-JEPA + LogReg | Gap |
|:------:|----:|:------------:|:----------------:|:---:|
| 1% | 50 | 23.52% | **27.61%** | +4.09% |
| 2% | 100 | 25.05% | **33.69%** | +8.64% |
| 5% | 250 | 29.83% | **37.62%** | +7.80% |
| 10% | 500 | 33.83% | **41.80%** | +7.97% |
| 20% | 1000 | 37.49% | **45.32%** | +7.84% |
| 50% | 2500 | 40.19% | **47.91%** | +7.73% |
| 100% | 5000 | 43.16% | **49.79%** | +6.63% |

### CIFAR-10 (50K train pretrain, linear probe on 10K test)

| Method | Backbone | Params | Accuracy | Source |
|--------|----------|:------:|:--------:|--------|
| DINO | ResNet-18 | 11M | 89.19% | Caron et al., ICCV 2021 [7]; CueCo [13] |
| SwAV | ResNet-18 | 11M | 89.17% | Caron et al., NeurIPS 2020 [4]; CueCo [13] |
| VICReg | ResNet-18 | 11M | 90.07% | Bardes et al., ICLR 2022 [9]; CueCo [13] |
| SimSiam | ResNet-18 | 11M | 90.51% | Chen & He, CVPR 2021 [6]; CueCo [13] |
| SimCLR | ResNet-18 | 11M | 90.74% | Chen et al., ICML 2020 [1]; CueCo [13] |
| Barlow Twins | ResNet-18 | 11M | 92.10% | Zbontar et al., ICML 2021 [5]; CueCo [13] |
| BYOL | ResNet-18 | 11M | 92.61% | Grill et al., NeurIPS 2020 [3]; CueCo [13] |
| MoCo v2 | ResNet-18 | 11M | 92.94% | Chen et al., 2020 [2]; CueCo [13] |
| MoCo v3 | ResNet-18 | 11M | 93.10% | Chen et al., ICCV 2021 [8]; CueCo [13] |
| Raw (72-dim) + LogReg | - | - | 42.09% | Ours (no learning) |
| **IG-JEPA + LogReg** | **GraphTransformer** | **~1.5M** | **50.11%** | **Ours** |
| **IG-JEPA + MLP** | **GraphTransformer** | **~1.5M** | **56.31%** | **Ours** |

#### Label Efficiency (CIFAR-10)

| Labels | N | Raw + LogReg | IG-JEPA + LogReg | Gap |
|:------:|-----:|:------------:|:----------------:|:---:|
| 1% | 500 | 30.42% | **38.16%** | +7.74% |
| 2% | 1000 | 32.96% | **41.38%** | +8.42% |
| 5% | 2500 | 36.27% | **45.06%** | +8.79% |
| 10% | 5000 | 38.44% | **46.90%** | +8.46% |
| 20% | 10000 | 39.23% | **47.84%** | +8.61% |
| 50% | 25000 | 41.16% | **49.59%** | +8.43% |
| 100% | 50000 | 42.09% | **50.11%** | +8.02% |

### TinyImageNet (100K train pretrain, 200 classes)

| Method | Backbone | Params | Accuracy | Source |
|--------|----------|:------:|:--------:|--------|
| VICReg | ResNet-18 | 11M | 37.5% | Bardes et al., ICLR 2022 [9]; FroSSL [12] |
| DINO | ResNet-18 | 11M | 34.9% | Caron et al., ICCV 2021 [7]; FroSSL [12] |
| BYOL | ResNet-18 | 11M | 40.1% | Grill et al., NeurIPS 2020 [3]; FroSSL [12] |
| SwAV | ResNet-18 | 11M | 41.2% | Caron et al., NeurIPS 2020 [4]; FroSSL [12] |
| SimCLR | ResNet-18 | 11M | 41.9% | Chen et al., ICML 2020 [1]; FroSSL [12] |
| MoCo v2 | ResNet-18 | 11M | 41.9% | Chen et al., 2020 [2]; FroSSL [12] |
| Barlow Twins | ResNet-18 | 11M | 45.3% | Zbontar et al., ICML 2021 [5]; FroSSL [12] |
| SimSiam | ResNet-18 | 11M | 45.6% | Chen & He, CVPR 2021 [6]; FroSSL [12] |
| Raw (72-dim) + LogReg | - | - | 8.95% | Ours (no learning) |
| **IG-JEPA + LogReg** | **GraphTransformer** | **~1.5M** | **14.14%** | **Ours** |
| **IG-JEPA + MLP** | **GraphTransformer** | **~1.5M** | **17.25%** | **Ours** |

### Summary & What We Beat

#### Label Efficiency (TinyImageNet)

| Labels | N | Raw + LogReg | IG-JEPA + LogReg | Gap |
|:------:|-----:|:------------:|:----------------:|:---:|
| 1% | 1000 | 2.63% | **4.79%** | +2.16% |
| 2% | 2000 | 3.33% | **5.68%** | +2.35% |
| 5% | 5000 | 4.94% | **8.02%** | +3.08% |
| 10% | 10000 | 5.91% | **9.71%** | +3.80% |
| 20% | 20000 | 6.92% | **11.42%** | +4.50% |
| 50% | 50000 | 8.41% | **13.22%** | +4.81% |
| 100% | 100000 | 8.95% | **14.14%** | +5.19% |

### Summary & What We Beat

| Dataset | Classes | IG-JEPA (LR) | IG-JEPA (MLP) | vs Raw | Benchmarks Beaten |
|---------|:-------:|:------------:|:-------------:|:------:|-------------------|
| STL-10 | 10 | 49.79% | 50.15% | +6.63% | None (benchmarks: 82-90%) |
| CIFAR-10 | 10 | 50.11% | 56.31% | +8.02% | None (benchmarks: 89-93%) |
| TinyImageNet | 200 | 14.14% | 17.25% | +5.19% | None (benchmarks: 35-46%) |

**Honest assessment:**
- With 72-dim hand-crafted features, we do **not** beat any published ResNet-18 SSL benchmark. These methods learn hierarchical features end-to-end with 11M parameters and ~800 training epochs, while we use fixed 72-dim features with ~1.5M params.
- The **consistent +5-9% improvement** of IG-JEPA over raw features across all datasets and all label fractions demonstrates that graph structural learning adds genuine value. This improvement is feature-agnostic and could compound with stronger input features.
- The **MLP probe consistently outperforms LogReg** (e.g., 56.31% vs 50.11% on CIFAR-10), suggesting the learned representations contain nonlinear structure that a linear probe cannot fully exploit.

### With DINO Features (Archived — Not Fair Comparison)

When using frozen DINO ViT-S/16 features (384-dim, pretrained on ImageNet) instead of raw pixel features, IG-JEPA achieved **94.09%** on STL-10 — exceeding SimSiam (~90%), SimCLR (~89%), and BYOL (~89%). However, this comparison is not fair since DINO was pretrained on 1.2M ImageNet images while the benchmarks learn from scratch. These results are archived in `archive/v1_signals/`.

## Graph-Minor Pooling Parameters

| Dataset | Image Size | merge_dist | cut_dist | del_small | del_large | Avg Nodes |
|---------|:----------:|:----------:|:--------:|:---------:|:---------:|:---------:|
| CIFAR-10 | 32x32 | 5 | 80 | 0 | 1024 | ~150 |
| STL-10 | 96x96 | 8 | 80 | 2 | 9216 | ~800 |
| TinyImageNet | 64x64 | 6 | 80 | 1 | 4096 | ~400 |

## Output Format

Results are saved to `signals/done_{dataset}_dino.json`:

```json
{
  "acc_raw": 0.4209,
  "acc_jepa_lr": 0.5011,
  "acc_jepa_mlp": 0.5631,
  "f1_raw": 0.4139,
  "f1_jepa": 0.4968,
  "precision_raw": 0.4156,
  "precision_jepa": 0.4960,
  "recall_raw": 0.4209,
  "recall_jepa": 0.5011,
  "confusion_matrix": [...],
  "label_efficiency": {
    "0.01": {"raw": 0.3042, "jepa": 0.3816, "n": 500},
    ...
  },
  "dataset": "cifar10",
  "params": 1867776
}
```

## References

[1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020. arXiv:2002.05709.

[2] Chen, X., Fan, H., Girshick, R., & He, K. "Improved Baselines with Momentum Contrastive Learning." arXiv:2003.04297, 2020.

[3] Grill, J-B., Strub, F., Altché, F., et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." NeurIPS 2020. arXiv:2006.07733.

[4] Caron, M., Misra, I., Mairal, J., et al. "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments." NeurIPS 2020.

[5] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." ICML 2021. arXiv:2103.03230.

[6] Chen, X. & He, K. "Exploring Simple Siamese Representation Learning." CVPR 2021. arXiv:2011.10566.

[7] Caron, M., Touvron, H., Misra, I., et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021. arXiv:2104.14294.

[8] Chen, X., Xie, S., & He, K. "An Empirical Study of Training Self-Supervised Vision Transformers." ICCV 2021. arXiv:2104.02057.

[9] Bardes, A., Ponce, J., & LeCun, Y. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR 2022. arXiv:2105.04906.

[10] He, K., Chen, X., Xie, S., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. arXiv:2111.06377.

[11] Assran, M., Duval, Q., Misra, I., et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR 2023. arXiv:2301.08243.

[12] Halvagal, M.S. & Bhatt, D. "FroSSL: Frobenius Norm Minimization for Efficient Multiview Self-Supervised Learning." arXiv:2310.02903, 2023.

[13] Giakoumoglou, N., et al. "Cluster Contrast for Unsupervised Visual Representation Learning." arXiv:2507.12359, 2025.

[14] Zheng, M., et al. "ReSSL: Relational Self-Supervised Learning with Weak Augmentation." NeurIPS 2021. arXiv:2107.09282.

[15] "Stochastic Pseudo Neighborhoods for Contrastive Learning." arXiv:2308.06983, 2023.
