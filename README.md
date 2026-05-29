# IG-JEPA: Image-Graph Joint Embedding Predictive Architecture

Self-supervised representation learning on graph-structured image decompositions. Images are converted to graphs via graph-minor pooling, enriched with DINO patch embeddings, then trained with a JEPA objective (student-teacher with topology-aware masking).

## Project Structure

```
.
├── src/                    # All runnable scripts
├── fastloops/              # Rust graph-minor pooling kernel (PyO3)
├── signals/                # JSON result files from completed experiments
├── reports/                # Technical report, figures, result tables
├── data/                   # Medical datasets (LiTS, Pancreas)
└── README.md
```

## Prerequisites

```bash
# Python dependencies
pip install torch torchvision torch_geometric
pip install scikit-learn scipy numpy rdflib

# Build the Rust kernel (required by all scripts except run_kg_jepa.py)
cd fastloops
pip install maturin
maturin develop --release
cd ..
```

The `fastloops` module provides `merge_and_cut()` — a fast Rust kernel that performs flood-fill merge, gradient cut, and noise deletion to convert images into supernodes.

## Scripts — What Each Does

### Core Pipeline

| Script | Paper Table | Description |
|--------|-------------|-------------|
| `run_dino.py` | Table 1 | **Main pipeline.** Loads images, extracts DINO features, builds 398-dim graphs via graph-minor pooling, trains IG-JEPA, evaluates with linear probe. Supports CIFAR-10, STL-10, TinyImageNet. |
| `run_lowlabel.py` | Table 2 | **Low-label transfer.** Reuses cached graphs from `run_dino.py`. Trains JEPA once, then evaluates linear probe at 1%, 5%, 10%, and 100% label fractions. |
| `run_table3_all.py` | Table 3 (CIFAR-10, TinyImageNet) | **Graphization ablation.** Compares three graph construction methods (proposed graph-minor, SLIC superpixel, patch kNN) on CIFAR-10 and TinyImageNet. |
| `run_ablation_table3.py` | Table 3 (STL-10) | **Graphization ablation on STL-10.** Same comparisons as above plus S=3 hierarchy variant, run on STL-10. |
| `run_table4.py` | Table 4 | **Efficiency benchmark.** Measures token count, GPU memory, throughput (images/sec), and parameter count. No training — just forward-pass profiling. |

### Medical / Knowledge Graph

| Script | Paper Table | Description |
|--------|-------------|-------------|
| `run_kg_jepa.py` | KG Results | **Clinical knowledge graph JEPA.** Builds KGs from LiTS/Pancreas medical data (RDF schema), applies JEPA for tumor burden classification and size regression. Does not require `fastloops`. |
| `run_medical_v2.py` | Medical Results | **Medical CT segmentation.** Per-slice graph construction with liver-specific CT windowing, multi-slice context features, MLP probe with Dice loss. Targets LiTS liver and Pancreas segmentation. |

### Ablations / Extensions

| Script | Description |
|--------|-------------|
| `run_graphonly_enriched.py` | **Graph-only ablation (no DINO).** Enriches 14-dim boundary features to ~45-dim with color histograms, texture stats, and WL hash features. Tests how far graph structure alone can go. |
| `run_gap_close.py` | **Experimental.** Larger model (hid=512, 6 layers), graph augmentation (edge drop + feature noise), attention pooling. Attempts to close accuracy gap on smaller-resolution datasets. |
| `run_imagenet100.py` | **ImageNet-100 benchmark.** Same pipeline as `run_dino.py` applied to 100-class ImageNet subset from HuggingFace. |

### Utility

| Script | Description |
|--------|-------------|
| `generate_medical_figure.py` | **Figure generation.** Produces the medical qualitative figure (graph-minor pooling on CT slices). Designed to run on a local machine with access to MICCAI data. |

## Run Order

The scripts are mostly independent — each is self-contained with its own model definition, training loop, and evaluation. However, `run_lowlabel.py` depends on cached graphs.

### Step 1: Build fastloops

```bash
cd fastloops && maturin develop --release && cd ..
```

### Step 2: Main results (Table 1)

```bash
# Each dataset runs independently. Results saved to signals/.
python src/run_dino.py --dataset cifar10 --gpu 0
python src/run_dino.py --dataset stl10 --gpu 0
python src/run_dino.py --dataset tinyimagenet --gpu 0
```

This caches the 398-dim graphs to `/tmp/igjepa_cache/` for reuse by downstream scripts.

### Step 3: Low-label transfer (Table 2)

Requires cached graphs from Step 2.

```bash
python src/run_lowlabel.py --dataset cifar10 --gpu 0
python src/run_lowlabel.py --dataset tinyimagenet --gpu 0
```

### Step 4: Ablations (Table 3)

Can run in parallel with Step 2 — builds its own graphs.

```bash
python src/run_table3_all.py --dataset cifar10 --gpu 0
python src/run_table3_all.py --dataset tinyimagenet --gpu 0
python src/run_ablation_table3.py --gpu 0   # STL-10 variants
```

### Step 5: Efficiency (Table 4)

No training required. Quick profiling run.

```bash
python src/run_table4.py --dataset cifar10 --gpu 0
python src/run_table4.py --dataset tinyimagenet --gpu 0
```

### Step 6: Medical experiments (independent)

Requires medical data in `data/`.

```bash
python src/run_kg_jepa.py --dataset lits --gpu 0
python src/run_kg_jepa.py --dataset pancreas --gpu 0
python src/run_medical_v2.py --dataset lits --gpu 0
python src/run_medical_v2.py --dataset pancreas --gpu 0
```

### Optional: Ablations and extensions

```bash
python src/run_graphonly_enriched.py --gpu 0       # No-DINO ablation
python src/run_gap_close.py --dataset cifar10 --gpu 0  # Experimental V2
python src/run_imagenet100.py --gpu 0              # ImageNet-100
```

## Output

All scripts write JSON results to `signals/`. Example output from `run_dino.py`:

```json
{
  "acc_raw": 0.0985,
  "acc_jepa_lr": 0.6501,
  "acc_jepa_mlp": 0.6124,
  "dataset": "cifar10",
  "params": 2511104
}
```

- `acc_raw`: Linear probe on raw mean-pooled features (no JEPA)
- `acc_jepa_lr`: Linear probe on JEPA embeddings
- `acc_jepa_mlp`: MLP probe on JEPA embeddings

## Key Hyperparameters

| Parameter | Value | Set in |
|-----------|-------|--------|
| Hidden dim | 256 | `--hid` (all scripts) |
| Layers | 4 | Model class |
| Mask ratio | 40% | Model class |
| EMA momentum | 0.996 | Model class |
| Learning rate | 1e-4 | `--lr` |
| Batch size | 32 | `--bs` |
| Epochs | 100 | `--epochs` |
| Loss weights | pred + 25*var + 1*cov | Model forward() |
