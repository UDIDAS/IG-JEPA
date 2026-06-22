# IG-JEPA: Image-Graph Joint Embedding Predictive Architecture

Self-supervised representation learning on graph-structured image decompositions. Images are converted to superpixel graphs via Rust-accelerated graph-minor pooling, then trained with a graph JEPA objective featuring per-graph BFS subgraph masking, context-neighbor prediction, and graph-level BYOL alignment.

Two pipelines are provided:
1. **`run_dino.py`** — Hand-crafted 72-dim pixel features (no pretrained models). Validates that graph structure + JEPA adds value over raw features.
2. **`run_cnn_jepa.py`** — ResNet-18 CNN backbone + Graph JEPA (end-to-end). Same backbone as SimCLR/BYOL benchmarks for **fair head-to-head comparison**.

## Pipeline Overview

### Pipeline 1: Raw Pixel Features (`run_dino.py`)

```
Raw Image (H x W x 3)
    |
    v
[1] Graph-Minor Pooling (Rust: fastloops)
    - Flood-fill merge similar pixels → superpixels
    - Cut dissimilar boundaries, delete small/large regions
    -> Superpixel graph (~150-800 nodes)
    |
    v
[2] Pixel Feature Engineering (72-dim per node, NO pretrained models)
    - Color: RGB mean/std, HSV mean/std, grayscale mean/std (14)
    - Geometry: log(area), centroid, bbox, compactness, relative area (8)
    - Higher-order: RGB skewness + kurtosis (6)
    - Texture: grayscale histogram 16 bins (16)
    - Gradient: magnitude mean/std + direction histogram (8)
    - Spatial: 4x4 grid encoding (16), 2nd order moments (4)
    |
    v
[3] Graph JEPA Training → [4] Linear Probe Evaluation
```

### Pipeline 2: CNN + Graph JEPA (`run_cnn_jepa.py`)

```
Raw Image (H x W x 3)
    |
    +---> [1] ResNet-18 CNN → spatial feature map (512-dim, H/32 x W/32)
    |
    +---> [2] Graph-Minor Pooling → superpixel graph (edges + pixel-to-node map)
    |
    v
[3] Pool CNN features per superpixel → 512-dim learned node features
    (each node = mean of CNN features within its superpixel region)
    |
    v
[4] GraphTransformer JEPA Training (end-to-end with CNN)
    |
    v
[5] Linear Probe Evaluation

Benchmark:  Image → ResNet-18 → global pool → 512d → probe
Ours:       Image → ResNet-18 → feature map → pool per superpixel → graph JEPA → probe

Same backbone, same params — only difference is graph structure on top.
```

## Architecture

**Graph JEPA Training Objective** (3 losses):

| Loss | Weight | Description |
|------|:------:|-------------|
| Prediction | 1.0 | Student predicts teacher's masked node embeddings from context neighbor aggregation |
| BYOL | 1.0 | Student graph embedding matches teacher graph embedding (cosine similarity) |
| VICReg Var | 25.0 | Prevents embedding collapse |
| VICReg Cov | 1.0 | Decorrelates embedding dimensions |

**Key Design Decisions:**
- **Per-graph BFS masking**: Each graph in the batch gets its own connected subgraph mask via BFS — not random nodes, but spatially coherent regions.
- **Context-neighbor prediction**: Masked nodes are disconnected from the student graph (no info leak through attention). Predictions use aggregated context neighbor embeddings.
- **Global residual**: Encoder output = layers(x) + proj(x), preserving input signal.
- **Step-wise validation**: Graph structure, CNN pooling, JEPA forward/backward, and embedding health are all validated before training begins, with runtime NaN/collapse checks.

## Project Structure

```
.
├── src/
│   ├── run_dino.py              # Pipeline 1: raw pixel features (72-dim)
│   └── run_cnn_jepa.py          # Pipeline 2: ResNet-18 + Graph JEPA (end-to-end)
├── fastloops/
│   ├── src/lib.rs               # Rust: graph-minor pooling + BFS masking
│   ├── Cargo.toml
│   └── Cargo.lock
├── signals/                     # JSON result files
├── archive/
│   ├── v1_src/                  # Previous experiment scripts
│   └── v1_signals/              # Previous result JSONs
├── README.md
└── .gitignore
```

## Setup

```bash
# Python dependencies
pip install torch torchvision torch_geometric
pip install scikit-learn scipy numpy maturin

# Build the Rust kernel
cd fastloops && maturin develop --release && cd ..

# IMPORTANT: Remove old fastloops if installed (v0.3.0 conflicts)
pip uninstall fastloops -y
cd fastloops && maturin develop --release && cd ..

# Verify
python -c "import fastloops; assert 'subgraph_mask' in dir(fastloops), 'Wrong version!'"
```

## Running Experiments

### Pipeline 1: Raw Pixel Features

```bash
python src/run_dino.py --dataset stl10 --unlabeled --gpu 0 --epochs 100 --bs 64
python src/run_dino.py --dataset cifar10 --gpu 0 --epochs 200 --bs 64
python src/run_dino.py --dataset tinyimagenet --gpu 1 --epochs 200 --bs 64
```

### Pipeline 2: CNN + Graph JEPA (fair benchmark comparison)

```bash
python src/run_cnn_jepa.py --dataset stl10 --unlabeled --gpu 0 --epochs 200 --bs 128 --lr 1e-3
python src/run_cnn_jepa.py --dataset cifar10 --gpu 1 --epochs 200 --bs 256 --lr 1e-3
python src/run_cnn_jepa.py --dataset tinyimagenet --gpu 0 --epochs 200 --bs 128 --lr 1e-3
```

## Results by Dataset

All published benchmarks use **ResNet-18** (~11M params) trained from scratch on each dataset.

---

### STL-10

Protocol: Pretrain on 100K unlabeled, linear probe on 5K labeled train, eval on 8K labeled test.

**Accuracy comparison:**

| Method | Backbone | Params | Accuracy | Source |
|--------|----------|:------:|:--------:|--------|
| DINO | ResNet-18 | 11M | ~82.0% | [7] |
| MoCo v2 | ResNet-18 | 11M | ~83.6% | [2]; [15] |
| BYOL | ResNet-18 | 11M | ~88.6% | [3]; [15] |
| SimCLR | ResNet-18 | 11M | ~89.3% | [1]; [15] |
| **SimSiam (SOTA)** | **ResNet-18** | **11M** | **~90.0%** | **[6]; [15]** |
| IG-JEPA P1 (raw pixels) | GraphTransformer | 1.5M | 49.79% | Ours |
| **IG-JEPA P2 (CNN+Graph)** | **ResNet-18 + GraphTransformer** | **17M** | ***in progress*** | **Ours** |

**Label efficiency (Pipeline 1 — raw pixel features):**

| Labels | N | Raw+LR | JEPA+LR | Gap |
|:------:|----:|:------:|:-------:|:---:|
| 1% | 50 | 23.52% | **27.61%** | +4.09% |
| 2% | 100 | 25.05% | **33.69%** | +8.64% |
| 5% | 250 | 29.83% | **37.62%** | +7.80% |
| 10% | 500 | 33.83% | **41.80%** | +7.97% |
| 20% | 1000 | 37.49% | **45.32%** | +7.84% |
| 50% | 2500 | 40.19% | **47.91%** | +7.73% |
| 100% | 5000 | 43.16% | **49.79%** | +6.63% |

---

### CIFAR-10

Protocol: Pretrain on 50K train (self-supervised), linear probe eval on 10K test.

**Accuracy comparison:**

| Method | Backbone | Params | Accuracy | Source |
|--------|----------|:------:|:--------:|--------|
| SwAV | ResNet-18 | 11M | 89.17% | [4]; [13] |
| DINO | ResNet-18 | 11M | 89.19% | [7]; [13] |
| VICReg | ResNet-18 | 11M | 90.07% | [9]; [13] |
| SimSiam | ResNet-18 | 11M | 90.51% | [6]; [13] |
| SimCLR | ResNet-18 | 11M | 90.74% | [1]; [13] |
| Barlow Twins | ResNet-18 | 11M | 92.10% | [5]; [13] |
| BYOL | ResNet-18 | 11M | 92.61% | [3]; [13] |
| MoCo v2 | ResNet-18 | 11M | 92.94% | [2]; [13] |
| **MoCo v3 (SOTA)** | **ResNet-18** | **11M** | **93.10%** | **[8]; [13]** |
| IG-JEPA P1 (raw pixels) | GraphTransformer | 1.5M | 50.11% | Ours |
| **IG-JEPA P2 (CNN+Graph)** | **ResNet-18 + GraphTransformer** | **17M** | ***in progress*** | **Ours** |

**Label efficiency (Pipeline 1 — raw pixel features):**

| Labels | N | Raw+LR | JEPA+LR | Gap |
|:------:|-----:|:------:|:-------:|:---:|
| 1% | 500 | 30.42% | **38.16%** | +7.74% |
| 2% | 1000 | 32.96% | **41.38%** | +8.42% |
| 5% | 2500 | 36.27% | **45.06%** | +8.79% |
| 10% | 5000 | 38.44% | **46.90%** | +8.46% |
| 20% | 10000 | 39.23% | **47.84%** | +8.61% |
| 50% | 25000 | 41.16% | **49.59%** | +8.43% |
| 100% | 50000 | 42.09% | **50.11%** | +8.02% |

---

### TinyImageNet

Protocol: Pretrain on 100K train (self-supervised), linear probe eval on 10K validation. 200 classes.

**Accuracy comparison:**

| Method | Backbone | Params | Accuracy | Source |
|--------|----------|:------:|:--------:|--------|
| DINO | ResNet-18 | 11M | 34.9% | [7]; [12] |
| VICReg | ResNet-18 | 11M | 37.5% | [9]; [12] |
| BYOL | ResNet-18 | 11M | 40.1% | [3]; [12] |
| SwAV | ResNet-18 | 11M | 41.2% | [4]; [12] |
| SimCLR | ResNet-18 | 11M | 41.9% | [1]; [12] |
| MoCo v2 | ResNet-18 | 11M | 41.9% | [2]; [12] |
| Barlow Twins | ResNet-18 | 11M | 45.3% | [5]; [12] |
| **SimSiam (SOTA)** | **ResNet-18** | **11M** | **45.6%** | **[6]; [12]** |
| IG-JEPA P1 (raw pixels) | GraphTransformer | 1.5M | 14.14% | Ours |
| **IG-JEPA P2 (CNN+Graph)** | **ResNet-18 + GraphTransformer** | **17M** | ***pending*** | **Ours** |

**Label efficiency (Pipeline 1 — raw pixel features):**

| Labels | N | Raw+LR | JEPA+LR | Gap |
|:------:|-----:|:------:|:-------:|:---:|
| 1% | 1000 | 2.63% | **4.79%** | +2.16% |
| 2% | 2000 | 3.33% | **5.68%** | +2.35% |
| 5% | 5000 | 4.94% | **8.02%** | +3.08% |
| 10% | 10000 | 5.91% | **9.71%** | +3.80% |
| 20% | 20000 | 6.92% | **11.42%** | +4.50% |
| 50% | 50000 | 8.41% | **13.22%** | +4.81% |
| 100% | 100000 | 8.95% | **14.14%** | +5.19% |

## Graph-Minor Pooling Parameters

| Dataset | Image Size | merge_dist | cut_dist | del_small | del_large | Avg Nodes |
|---------|:----------:|:----------:|:--------:|:---------:|:---------:|:---------:|
| CIFAR-10 | 32x32 | 5 | 80 | 0 | 1024 | ~150 |
| STL-10 | 96x96 | 8 | 80 | 2 | 9216 | ~800 |
| TinyImageNet | 64x64 | 6 | 80 | 1 | 4096 | ~400 |

## References

[1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020. arXiv:2002.05709.

[2] Chen, X., Fan, H., Girshick, R., & He, K. "Improved Baselines with Momentum Contrastive Learning." arXiv:2003.04297, 2020.

[3] Grill, J-B., Strub, F., Altche, F., et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." NeurIPS 2020. arXiv:2006.07733.

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
