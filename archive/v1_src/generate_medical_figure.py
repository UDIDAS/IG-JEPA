"""
Generate Figure 5 medical panel — run this on your Windows machine
where the MICCAI 26 data lives.

Requirements:
  pip install numpy matplotlib pillow fastloops scipy

Usage:
  python generate_medical_figure.py \
    --nii_viz "C:/Users/udipt/Desktop/MICCAI 26/data/nii/viz" \
    --lits_viz "C:/Users/udipt/Desktop/MICCAI 26/data/LiTS_newUpdate/3D_Visualizations" \
    --output "./fig_medical_qualitative.png"

It will:
  1. Pick sample images from Pancreas (nii) and LiTS datasets
  2. Run graph-minor pooling on each
  3. Generate a 2-row x 4-col figure similar to Figure 5
"""

import argparse
import os
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import fastloops
except ImportError:
    print("ERROR: fastloops not installed. Build it first:")
    print("  cd fastloops && maturin develop --release")
    exit(1)

EDGE_MERGED = 0b0001_0000

# Kernel params tuned for medical 3D visualizations (~400-600px)
KERNEL_MEDICAL = {
    "merge_distance": 12,
    "cut_distance": 100,
    "delete_small_node_max_size": 10,
    "delete_large_node_min_size": 50000,
}


def region_labels(adj):
    H2, W2 = adj.shape
    H, W = (H2 + 1) // 2, (W2 + 1) // 2
    n = H * W
    rr, rc = np.where((adj[::2, 1::2] & EDGE_MERGED) != 0)
    dr, dc = np.where((adj[1::2, ::2] & EDGE_MERGED) != 0)
    src = np.concatenate([rr * W + rc, dr * W + dc])
    dst = np.concatenate([rr * W + rc + 1, dr * W + dc + W])
    if src.size == 0:
        return np.arange(n, dtype=np.int32).reshape(H, W)
    g = coo_matrix((np.ones(src.size, dtype=bool), (src, dst)), shape=(n, n)).tocsr()
    _, lab = connected_components(g, directed=False)
    return lab.astype(np.int32).reshape(H, W)


def get_graph(img, kernel):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    adj, feat = fastloops.merge_and_cut(img, **kernel)
    N = feat.shape[0]
    if N < 2:
        return None
    labels_map = region_labels(adj)
    canon = feat[:, 14:16].astype(np.int64)
    ml = int(labels_map.max()) + 1
    cc2s = np.full(ml, -1, dtype=np.int32)
    for j in range(N):
        cc2s[labels_map[int(canon[j, 0]), int(canon[j, 1])]] = j
    snode_map = cc2s[labels_map]

    hnm = (adj[::2, 1::2] & EDGE_MERGED) == 0
    ll, lr = labels_map[:, :-1], labels_map[:, 1:]
    hd = (ll != lr) & hnm
    hs1, hs2 = cc2s[ll[hd]], cc2s[lr[hd]]
    hv = (hs1 >= 0) & (hs2 >= 0) & (hs1 != hs2)
    vnm = (adj[1::2, ::2] & EDGE_MERGED) == 0
    lt, lb = labels_map[:-1, :], labels_map[1:, :]
    vd = (lt != lb) & vnm
    vs1, vs2 = cc2s[lt[vd]], cc2s[lb[vd]]
    vv = (vs1 >= 0) & (vs2 >= 0) & (vs1 != vs2)
    ps = []
    if hv.any():
        ps.append(np.stack([hs1[hv], hs2[hv]], 1))
    if vv.any():
        ps.append(np.stack([vs1[vv], vs2[vv]], 1))
    if not ps:
        return None
    edges = np.unique(np.sort(np.concatenate(ps), axis=1), axis=0)
    ff = feat.astype(np.float64)
    area = ff[:, 0].clip(min=1)
    cx = ff[:, 1] / area
    cy = ff[:, 2] / area
    return {
        "centroids": np.stack([cx, cy], axis=1),
        "edges": edges,
        "areas": ff[:, 0],
        "snode_map": snode_map,
        "N": N,
    }


def draw_graph(ax, img, gd, max_edges=600):
    ax.imshow(img)
    for s, d in gd["edges"][:max_edges]:
        ax.plot(
            [gd["centroids"][s, 0], gd["centroids"][d, 0]],
            [gd["centroids"][s, 1], gd["centroids"][d, 1]],
            "-", color="cyan", linewidth=0.4, alpha=0.4,
        )
    sizes = np.clip(gd["areas"] / gd["areas"].max() * 25, 2, 25)
    ax.scatter(
        gd["centroids"][:, 0], gd["centroids"][:, 1],
        s=sizes, c="yellow", zorder=5, edgecolors="black", linewidths=0.2,
    )
    ax.axis("off")


def draw_regions(ax, img, gd):
    overlay = img.astype(np.float32) / 255.0
    np.random.seed(42)
    nc = np.random.rand(gd["N"], 3) * 0.6 + 0.2
    for nid in range(gd["N"]):
        m = gd["snode_map"] == nid
        if m.sum() == 0:
            continue
        overlay[m] = overlay[m] * 0.4 + nc[nid] * 0.6
    ax.imshow(np.clip(overlay, 0, 1))
    ax.axis("off")


def draw_boundaries(ax, img, gd):
    sm = gd["snode_map"]
    boundary = np.zeros_like(img[:, :, 0], dtype=bool)
    boundary[:-1, :] |= sm[:-1, :] != sm[1:, :]
    boundary[:, :-1] |= sm[:, :-1] != sm[:, 1:]
    bnd = img.copy()
    bnd[boundary] = [255, 255, 0]
    ax.imshow(bnd)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nii_viz", required=True, help="Path to nii/viz folder with pancreas PNGs")
    parser.add_argument("--lits_viz", required=True, help="Path to LiTS 3D_Visualizations folder")
    parser.add_argument("--output", default="fig_medical_qualitative.png")
    args = parser.parse_args()

    # Find sample images
    nii_files = sorted([f for f in os.listdir(args.nii_viz) if f.endswith(".png")])
    lits_files = sorted([f for f in os.listdir(args.lits_viz) if f.endswith(".png")])

    if not nii_files:
        print(f"No PNGs found in {args.nii_viz}")
        return
    if not lits_files:
        print(f"No PNGs found in {args.lits_viz}")
        return

    # Pick visually interesting samples (not the first one which might be boring)
    pan_img = np.asarray(Image.open(os.path.join(args.nii_viz, nii_files[5])))
    lits_img = np.asarray(Image.open(os.path.join(args.lits_viz, lits_files[10])))

    if pan_img.ndim == 2:
        pan_img = np.stack([pan_img] * 3, axis=-1)
    if lits_img.ndim == 2:
        lits_img = np.stack([lits_img] * 3, axis=-1)
    # Handle RGBA
    if pan_img.shape[2] == 4:
        pan_img = pan_img[:, :, :3]
    if lits_img.shape[2] == 4:
        lits_img = lits_img[:, :, :3]

    print(f"Pancreas: {nii_files[5]}, shape={pan_img.shape}")
    print(f"LiTS: {lits_files[10]}, shape={lits_img.shape}")

    # Build graphs
    print("Building graphs...")
    gd_pan = get_graph(pan_img, KERNEL_MEDICAL)
    gd_lits = get_graph(lits_img, KERNEL_MEDICAL)

    if gd_pan is None or gd_lits is None:
        print("Graph building failed — try different kernel params or different images")
        # Try with more aggressive merge
        alt_kernel = {**KERNEL_MEDICAL, "merge_distance": 15}
        if gd_pan is None:
            gd_pan = get_graph(pan_img, alt_kernel)
        if gd_lits is None:
            gd_lits = get_graph(lits_img, alt_kernel)

    print(f"  Pancreas: {gd_pan['N']} nodes, {len(gd_pan['edges'])} edges")
    print(f"  LiTS: {gd_lits['N']} nodes, {len(gd_lits['edges'])} edges")

    # === Build figure: 2 rows x 4 cols ===
    # Row 1: Pancreas CT — Input, Graph, Regions, Boundaries
    # Row 2: LiTS (Liver) — Input, Graph, Regions, Boundaries
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Pancreas
    axes[0, 0].imshow(pan_img)
    axes[0, 0].set_title("Input", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    draw_graph(axes[0, 1], pan_img, gd_pan)
    axes[0, 1].set_title(f"Graph ({gd_pan['N']} nodes)", fontsize=10, fontweight="bold")

    draw_regions(axes[0, 2], pan_img, gd_pan)
    axes[0, 2].set_title("Supernode Regions", fontsize=10, fontweight="bold")

    draw_boundaries(axes[0, 3], pan_img, gd_pan)
    axes[0, 3].set_title("Preserved Boundaries", fontsize=10, fontweight="bold")

    axes[0, 0].text(
        -0.15, 0.5, "Pancreas\nCT",
        transform=axes[0, 0].transAxes,
        fontsize=11, fontweight="bold", va="center", ha="center", rotation=90,
    )

    # Row 2: LiTS
    axes[1, 0].imshow(lits_img)
    axes[1, 0].set_title("Input", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    draw_graph(axes[1, 1], lits_img, gd_lits)
    axes[1, 1].set_title(f"Graph ({gd_lits['N']} nodes)", fontsize=10, fontweight="bold")

    draw_regions(axes[1, 2], lits_img, gd_lits)
    axes[1, 2].set_title("Supernode Regions", fontsize=10, fontweight="bold")

    draw_boundaries(axes[1, 3], lits_img, gd_lits)
    axes[1, 3].set_title("Preserved Boundaries", fontsize=10, fontweight="bold")

    axes[1, 0].text(
        -0.15, 0.5, "Liver\n(LiTS)",
        transform=axes[1, 0].transAxes,
        fontsize=11, fontweight="bold", va="center", ha="center", rotation=90,
    )

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight", dpi=300)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
