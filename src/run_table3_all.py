"""
Table 3 ablation: graphization variants on CIFAR-10 and TinyImageNet.

Compares three graph construction methods:
  1. Proposed (graph-minor pooling)
  2. Superpixel Graph (SLIC)
  3. Patch kNN Graph

Same architecture/training as run_dino.py. Reports linear probe accuracy.
Run on two datasets in sequence, or specify --dataset for one.
"""

import os, sys, copy, random, time, hashlib, pickle, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from concurrent.futures import ThreadPoolExecutor, as_completed
import fastloops

EDGE_MERGED = 0b0001_0000
CACHE_DIR = "/home/ud3d4/Desktop/NIPS 26/cache"

KERNELS = {
    "cifar10": {"merge_distance": 5, "cut_distance": 80,
                "delete_small_node_max_size": 0, "delete_large_node_min_size": 1024},
    "tinyimagenet": {"merge_distance": 6, "cut_distance": 80,
                     "delete_small_node_max_size": 1, "delete_large_node_min_size": 4096},
}


def region_labels(adj):
    H2, W2 = adj.shape; H, W = (H2+1)//2, (W2+1)//2; n = H*W
    rr, rc = np.where((adj[::2, 1::2] & EDGE_MERGED) != 0)
    dr, dc = np.where((adj[1::2, ::2] & EDGE_MERGED) != 0)
    src = np.concatenate([rr*W+rc, dr*W+dc])
    dst = np.concatenate([rr*W+rc+1, dr*W+dc+W])
    if src.size == 0: return np.arange(n, dtype=np.int32).reshape(H,W)
    g = coo_matrix((np.ones(src.size, dtype=bool), (src, dst)), shape=(n,n)).tocsr()
    _, lab = connected_components(g, directed=False)
    return lab.astype(np.int32).reshape(H,W)


# ── Graph construction variants ──

def build_proposed_graph(img_np, kernel, dino_grid):
    """Graph-minor pooling + DINO features (our proposed method)."""
    if img_np.ndim == 2: img_np = np.stack([img_np]*3, axis=-1)
    H, W = img_np.shape[:2]
    adj, feat = fastloops.merge_and_cut(img_np, **kernel)
    N = feat.shape[0]
    if N < 2: return None

    labels = region_labels(adj)
    canon = feat[:, 14:16].astype(np.int64)
    ml = int(labels.max()) + 1
    cc2s = np.full(ml, -1, dtype=np.int32)
    for j in range(N): cc2s[labels[int(canon[j,0]), int(canon[j,1])]] = j
    snode_map = cc2s[labels]

    # Edges
    hnm = (adj[::2, 1::2] & EDGE_MERGED) == 0
    ll, lr = labels[:, :-1], labels[:, 1:]
    hd = (ll != lr) & hnm; hs1, hs2 = cc2s[ll[hd]], cc2s[lr[hd]]; hv = (hs1>=0)&(hs2>=0)&(hs1!=hs2)
    vnm = (adj[1::2, ::2] & EDGE_MERGED) == 0
    lt, lb = labels[:-1,:], labels[1:,:]
    vd = (lt != lb) & vnm; vs1, vs2 = cc2s[lt[vd]], cc2s[lb[vd]]; vv = (vs1>=0)&(vs2>=0)&(vs1!=vs2)
    ps = []
    if hv.any(): ps.append(np.stack([hs1[hv], hs2[hv]], 1))
    if vv.any(): ps.append(np.stack([vs1[vv], vs2[vv]], 1))
    if not ps: return None
    ap = np.sort(np.concatenate(ps), axis=1); u = np.unique(ap, axis=0)
    s_arr = np.concatenate([u[:,0], u[:,1]]); d_arr = np.concatenate([u[:,1], u[:,0]])

    # Features (14 boundary + 384 DINO = 398)
    img_f = img_np.astype(np.float64) / 255.0
    flat_map = snode_map.ravel(); valid = flat_map >= 0
    counts = np.bincount(flat_map[valid], minlength=N).astype(np.float64).clip(min=1)

    nf = np.zeros((N, 398), dtype=np.float32)
    for ch in range(3):
        flat_ch = img_f[:,:,ch].ravel()[valid]
        sums = np.bincount(flat_map[valid], weights=flat_ch, minlength=N)
        nf[:, ch*2] = sums / counts
        sq_sums = np.bincount(flat_map[valid], weights=flat_ch**2, minlength=N)
        nf[:, ch*2+1] = np.sqrt((sq_sums / counts - (sums/counts)**2).clip(min=0))

    ff = feat.astype(np.float64); area = ff[:,0].clip(min=1)
    nf[:, 6] = np.log1p(area)
    nf[:, 7] = ff[:,1] / area / W
    nf[:, 8] = ff[:,2] / area / H
    nf[:, 9] = (ff[:,10]-ff[:,9]) / W
    nf[:, 10] = (ff[:,12]-ff[:,11]) / H
    nf[:, 11] = ff[:,13] / area
    bbox_area = ((ff[:,10]-ff[:,9]+1)*(ff[:,12]-ff[:,11]+1)).clip(min=1)
    nf[:, 12] = (area / bbox_area).clip(max=1)
    nf[:, 13] = area / (H*W)

    if dino_grid is not None:
        cx, cy = nf[:, 7], nf[:, 8]
        px = np.clip((cx * 14).astype(np.int32), 0, 13)
        py = np.clip((cy * 14).astype(np.int32), 0, 13)
        nf[:, 14:] = dino_grid[py, px, :].astype(np.float32)

    nf = np.nan_to_num(nf, nan=0.0)
    return Data(x=torch.from_numpy(nf), edge_index=torch.tensor(np.stack([s_arr, d_arr]), dtype=torch.long), num_nodes=N)


def build_patch_knn_graph(img_np, dino_grid, k=8):
    """Patch kNN graph: 14x14 DINO patches connected to k nearest neighbors."""
    if dino_grid is None: return None
    N = 196
    features = dino_grid.reshape(N, -1).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(features)
    _, indices = nn.kneighbors(features)
    src, dst = [], []
    for i in range(N):
        for j in indices[i, 1:]:
            src.append(i); dst.append(j); src.append(j); dst.append(i)
    nf = np.zeros((N, 398), dtype=np.float32)
    nf[:, 14:] = features
    for i in range(N):
        nf[i, 7] = (i % 14) / 14.0
        nf[i, 8] = (i // 14) / 14.0
    nf = np.nan_to_num(nf, nan=0.0)
    ei = np.unique(np.stack([src, dst]).T, axis=0).T
    return Data(x=torch.from_numpy(nf), edge_index=torch.tensor(ei, dtype=torch.long), num_nodes=N)


def build_superpixel_graph(img_np, dino_grid, n_segments=100):
    """SLIC superpixel graph."""
    from skimage.segmentation import slic
    if img_np.ndim == 2: img_np = np.stack([img_np]*3, axis=-1)
    H, W = img_np.shape[:2]
    segments = slic(img_np, n_segments=n_segments, compactness=10, start_label=0)
    N = segments.max() + 1
    edges = set()
    for r in range(H):
        for c in range(W-1):
            s1, s2 = segments[r,c], segments[r,c+1]
            if s1 != s2: edges.add((min(s1,s2), max(s1,s2)))
    for r in range(H-1):
        for c in range(W):
            s1, s2 = segments[r,c], segments[r+1,c]
            if s1 != s2: edges.add((min(s1,s2), max(s1,s2)))
    if not edges: return None
    el = list(edges)
    src = [e[0] for e in el] + [e[1] for e in el]
    dst = [e[1] for e in el] + [e[0] for e in el]
    img_f = img_np.astype(np.float64) / 255.0
    nf = np.zeros((N, 398), dtype=np.float32)
    for sid in range(N):
        mask = segments == sid; area = mask.sum()
        if area == 0: continue
        for ch in range(3):
            vals = img_f[:,:,ch][mask]; nf[sid, ch*2] = vals.mean(); nf[sid, ch*2+1] = vals.std()
        ys, xs = np.where(mask)
        nf[sid, 6] = np.log1p(area); nf[sid, 7] = xs.mean()/W; nf[sid, 8] = ys.mean()/H
        if dino_grid is not None:
            px = int(np.clip(nf[sid,7]*14, 0, 13)); py = int(np.clip(nf[sid,8]*14, 0, 13))
            nf[sid, 14:] = dino_grid[py, px, :].astype(np.float32)
    nf = np.nan_to_num(nf, nan=0.0)
    return Data(x=torch.from_numpy(nf), edge_index=torch.tensor(np.stack([src,dst]), dtype=torch.long), num_nodes=N)


# ── Model ──

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim, hid, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hid)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(TransformerConv(hid, hid // n_heads, heads=n_heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hid))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index) + x); x = F.gelu(x); x = self.dropout(x)
        return x

class IGJEPA(nn.Module):
    def __init__(self, in_dim, hid=256, n_layers=4, n_heads=4, mask_ratio=0.4, ema=0.996):
        super().__init__()
        self.mask_ratio, self.ema_m, self.hid = mask_ratio, ema, hid
        self.enc = GraphTransformerEncoder(in_dim, hid, n_layers, n_heads)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters(): p.requires_grad = False
        self.pred = nn.Sequential(nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,hid))
    @torch.no_grad()
    def ema_update(self):
        for a, b in zip(self.enc.parameters(), self.tgt.parameters()):
            b.data.mul_(self.ema_m).add_(a.data, alpha=1-self.ema_m)
    def forward(self, batch):
        x, ei = batch.x, batch.edge_index; N = x.size(0)
        nm = max(1, int(N * self.mask_ratio))
        mi = torch.randperm(N, device=x.device)[:nm]
        with torch.no_grad(): target = F.layer_norm(self.tgt(x, ei)[mi], [self.hid])
        xc = x.clone(); xc[mi] = 0; c = self.enc(xc, ei)
        pred = F.layer_norm(self.pred(c[mi]), [self.hid])
        pl = F.smooth_l1_loss(pred, target)
        std = torch.sqrt(c.var(dim=0)+1e-4); vl = F.relu(1.0-std).mean()
        cm = c-c.mean(0); cov = (cm.T@cm)/max(N-1,1)
        od = cov.flatten()[1:].view(self.hid-1, self.hid+1)[:,:-1].flatten()
        cl = (od**2).sum()/self.hid
        return pl + 25*vl + 1*cl, {"pred": pl.item(), "std": std.mean().item()}
    def encode_graph(self, batch):
        return global_mean_pool(self.enc(batch.x, batch.edge_index), batch.batch)


def train_and_eval(graphs, labels, device, hid=256, epochs=100, bs=32):
    n = len(graphs)
    n_train = int(n * 0.8)
    for i in range(n): graphs[i].y = torch.tensor([labels[i]], dtype=torch.long)
    train_loader = DataLoader(graphs[:n_train], batch_size=bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(graphs[n_train:], batch_size=bs, num_workers=0)

    in_dim = graphs[0].x.shape[1]
    model = IGJEPA(in_dim, hid).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    for ep in range(1, epochs+1):
        model.train()
        for b in train_loader:
            b = b.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss, info = model(b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); model.ema_update()
        sch.step()
        if ep % 25 == 0:
            print(f"    Ep {ep:3d} | Loss: {loss.item():.4f} std: {info['std']:.3f}", flush=True)

    model.eval()
    @torch.no_grad()
    def extract(loader):
        f, l = [], []
        for b in loader:
            b = b.to(device); f.append(model.encode_graph(b).cpu()); l.append(b.y.cpu())
        return torch.cat(f).numpy(), torch.cat(l).numpy().ravel()

    X_tr, y_tr = extract(train_loader)
    X_te, y_te = extract(test_loader)
    clf = LogisticRegression(max_iter=2000); clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te))


def load_dataset(name):
    from torchvision import datasets
    if name == "cifar10":
        tr = datasets.CIFAR10("/home/ud3d4/datasets/cifar10", train=True, download=True)
        te = datasets.CIFAR10("/home/ud3d4/datasets/cifar10", train=False, download=True)
        return tr, te
    elif name == "tinyimagenet":
        from datasets import load_dataset as hf_load
        hf_tr = hf_load('Maysee/tiny-imagenet', split='train', cache_dir='/home/ud3d4/datasets/tinyimagenet')
        hf_te = hf_load('Maysee/tiny-imagenet', split='valid', cache_dir='/home/ud3d4/datasets/tinyimagenet')
        class HFW:
            def __init__(self, d): self.d = d
            def __len__(self): return len(self.d)
            def __getitem__(self, i):
                item = self.d[i]
                img = item['image']
                if hasattr(img, 'convert'): img = img.convert('RGB')
                return np.array(img), item['label']
        return HFW(hf_tr), HFW(hf_te)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def extract_dino(all_imgs, device):
    from torchvision import transforms
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    dino.eval().to(device)
    dino_tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    all_patches = np.zeros((len(all_imgs), 14, 14, 384), dtype=np.float16)
    BS = 256
    for i in range(0, len(all_imgs), BS):
        batch = [dino_tf(img if img.ndim == 3 else np.stack([img]*3, axis=-1)) for img in all_imgs[i:i+BS]]
        inp = torch.stack(batch).to(device)
        with torch.no_grad():
            tokens = dino.get_intermediate_layers(inp, n=1)[0]
            patches = tokens[:, 1:, :].reshape(-1, 14, 14, 384).cpu().numpy().astype(np.float16)
        all_patches[i:i+len(batch)] = patches
        if (i // BS) % 10 == 0:
            print(f"    DINO: {i}/{len(all_imgs)}", flush=True)
    del dino; torch.cuda.empty_cache()
    return all_patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "tinyimagenet"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=8000,
                        help="Total images to use (train+test)")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)

    # Load data
    tr_ds, te_ds = load_dataset(args.dataset)
    n_tr = min(args.n_samples, len(tr_ds))
    n_te = min(args.n_samples // 4, len(te_ds))
    all_imgs = [np.array(tr_ds[i][0]) for i in range(n_tr)] + [np.array(te_ds[i][0]) for i in range(n_te)]
    all_labels = [tr_ds[i][1] for i in range(n_tr)] + [te_ds[i][1] for i in range(n_te)]
    print(f"  Images: {len(all_imgs)}", flush=True)

    # DINO features
    cache_path = os.path.join(CACHE_DIR, f"dino_{args.dataset}_{len(all_imgs)}.npy")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        print(f"  Loading cached DINO: {cache_path}", flush=True)
        all_patches = np.load(cache_path)
    else:
        print("  Extracting DINO features...", flush=True)
        all_patches = extract_dino(all_imgs, device)
        np.save(cache_path, all_patches)
    print(f"  DINO: {all_patches.shape}", flush=True)

    kernel = KERNELS[args.dataset]
    n_seg = 50 if args.dataset == "cifar10" else 100  # fewer segments for smaller images
    results = {}

    # ── 1. Proposed (graph-minor) ──
    print(f"\n=== Proposed (graph-minor pooling) ===", flush=True)
    graphs = []
    for i in range(len(all_imgs)):
        g = build_proposed_graph(all_imgs[i], kernel, all_patches[i])
        if g is not None:
            graphs.append(g)
        else:
            # Fallback: single-node graph to avoid index mismatch
            nf = np.zeros((1, 398), dtype=np.float32)
            graphs.append(Data(x=torch.from_numpy(nf), edge_index=torch.zeros(2,0,dtype=torch.long), num_nodes=1))
    print(f"  Built {len(graphs)} graphs, avg nodes: {np.mean([g.num_nodes for g in graphs]):.0f}", flush=True)
    acc = train_and_eval(graphs, all_labels, device, epochs=args.epochs)
    results["Proposed"] = acc
    print(f"  Result: {acc*100:.2f}%", flush=True)

    # ── 2. Superpixel (SLIC) ──
    print(f"\n=== Superpixel Graph (SLIC) ===", flush=True)
    graphs_sp = []
    for i in range(len(all_imgs)):
        g = build_superpixel_graph(all_imgs[i], all_patches[i], n_segments=n_seg)
        if g is not None:
            graphs_sp.append(g)
        else:
            nf = np.zeros((1, 398), dtype=np.float32)
            graphs_sp.append(Data(x=torch.from_numpy(nf), edge_index=torch.zeros(2,0,dtype=torch.long), num_nodes=1))
    print(f"  Built {len(graphs_sp)} graphs, avg nodes: {np.mean([g.num_nodes for g in graphs_sp]):.0f}", flush=True)
    acc_sp = train_and_eval(graphs_sp, all_labels, device, epochs=args.epochs)
    results["Superpixel (SLIC)"] = acc_sp
    print(f"  Result: {acc_sp*100:.2f}%", flush=True)

    # ── 3. Patch kNN ──
    print(f"\n=== Patch kNN Graph ===", flush=True)
    graphs_knn = []
    for i in range(len(all_imgs)):
        g = build_patch_knn_graph(all_imgs[i], all_patches[i])
        if g is not None:
            graphs_knn.append(g)
        else:
            nf = np.zeros((1, 398), dtype=np.float32)
            graphs_knn.append(Data(x=torch.from_numpy(nf), edge_index=torch.zeros(2,0,dtype=torch.long), num_nodes=1))
    print(f"  Built {len(graphs_knn)} graphs, avg nodes: {np.mean([g.num_nodes for g in graphs_knn]):.0f}", flush=True)
    acc_knn = train_and_eval(graphs_knn, all_labels, device, epochs=args.epochs)
    results["Patch kNN"] = acc_knn
    print(f"  Result: {acc_knn*100:.2f}%", flush=True)

    # ── Summary ──
    print(f"\n{'='*55}", flush=True)
    print(f"TABLE 3: GRAPHIZATION ABLATION ({args.dataset.upper()})", flush=True)
    print(f"{'='*55}", flush=True)
    for k, v in results.items():
        delta = (v - results["Proposed"]) * 100 if k != "Proposed" else 0
        print(f"  {k:<25s}: {v*100:.2f}%  (Δ={delta:+.1f}%)", flush=True)

    sig_path = f"/home/ud3d4/Desktop/NIPS 26/signals/table3_{args.dataset}.json"
    with open(sig_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {sig_path}", flush=True)


if __name__ == "__main__":
    main()
