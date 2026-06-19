"""
Graph-only ablation with enriched boundary features (no DINO).

Enriches the 14-dim boundary features to ~45-dim by adding:
- Color histogram (per-channel 5-bin quantiles)
- Texture statistics (color variance of variance, gradient proxy)
- Local graph topology (degree, clustering coefficient, neighbor stats)
- WL neighborhood hash features (3 iterations)

Goal: push graph-only accuracy from 18.9% to 30%+ on STL-10.
"""

import os, sys, copy, random, time, hashlib
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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import fastloops
import json

EDGE_MERGED = 0b0001_0000


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


def build_enriched_graph(img_np, kernel):
    """Build graph with enriched boundary features (~45 dim), NO DINO."""
    if img_np.ndim == 2: img_np = np.stack([img_np]*3, axis=-1)
    H, W = img_np.shape[:2]
    adj, feat = fastloops.merge_and_cut(img_np, **kernel)
    N = feat.shape[0]
    if N < 2: return None

    labels_map = region_labels(adj)
    canon = feat[:, 14:16].astype(np.int64)
    ml = int(labels_map.max()) + 1
    cc2s = np.full(ml, -1, dtype=np.int32)
    for j in range(N): cc2s[labels_map[int(canon[j,0]), int(canon[j,1])]] = j
    snode_map = cc2s[labels_map]

    # Edges
    hnm = (adj[::2, 1::2] & EDGE_MERGED) == 0
    ll, lr = labels_map[:, :-1], labels_map[:, 1:]
    hd = (ll != lr) & hnm; hs1, hs2 = cc2s[ll[hd]], cc2s[lr[hd]]; hv = (hs1>=0)&(hs2>=0)&(hs1!=hs2)
    vnm = (adj[1::2, ::2] & EDGE_MERGED) == 0
    lt, lb = labels_map[:-1,:], labels_map[1:,:]
    vd = (lt != lb) & vnm; vs1, vs2 = cc2s[lt[vd]], cc2s[lb[vd]]; vv = (vs1>=0)&(vs2>=0)&(vs1!=vs2)
    ps = []
    if hv.any(): ps.append(np.stack([hs1[hv], hs2[hv]], 1))
    if vv.any(): ps.append(np.stack([vs1[vv], vs2[vv]], 1))
    if not ps: return None
    ap = np.sort(np.concatenate(ps), axis=1); u = np.unique(ap, axis=0)
    s_arr = np.concatenate([u[:,0], u[:,1]]); d_arr = np.concatenate([u[:,1], u[:,0]])
    edge_index = np.stack([s_arr, d_arr])

    # === Enriched features ===
    img_f = img_np.astype(np.float64) / 255.0
    flat = snode_map.ravel(); valid = flat >= 0
    counts = np.bincount(flat[valid], minlength=N).astype(np.float64).clip(min=1)
    ff = feat.astype(np.float64); area = ff[:,0].clip(min=1)

    features = []

    # Block 1: Original 14 boundary features (normalized)
    nf_base = np.zeros((N, 14), dtype=np.float32)
    for ch in range(3):
        fc = img_f[:,:,ch].ravel()[valid]
        sums = np.bincount(flat[valid], weights=fc, minlength=N)
        nf_base[:, ch*2] = sums / counts  # mean color
        sq = np.bincount(flat[valid], weights=fc**2, minlength=N)
        nf_base[:, ch*2+1] = np.sqrt((sq/counts - (sums/counts)**2).clip(min=0))  # std color
    nf_base[:, 6] = np.log1p(area)
    nf_base[:, 7] = ff[:,1] / area / W  # centroid x
    nf_base[:, 8] = ff[:,2] / area / H  # centroid y
    nf_base[:, 9] = (ff[:,10]-ff[:,9]) / W  # bbox width
    nf_base[:, 10] = (ff[:,12]-ff[:,11]) / H  # bbox height
    nf_base[:, 11] = ff[:,13] / area  # boundary ratio
    bbox_a = ((ff[:,10]-ff[:,9]+1)*(ff[:,12]-ff[:,11]+1)).clip(min=1)
    nf_base[:, 12] = (area / bbox_a).clip(max=1)  # compactness
    nf_base[:, 13] = area / (H*W)  # relative area
    features.append(nf_base)

    # Block 2: Color histogram (5 quantile bins per channel = 15 dim)
    nf_hist = np.zeros((N, 15), dtype=np.float32)
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    for ch in range(3):
        ch_img = img_f[:,:,ch]
        for nid in range(N):
            m = snode_map == nid
            if m.sum() < 2: continue
            vals = ch_img[m]
            for qi, q in enumerate(quantiles):
                nf_hist[nid, ch*5 + qi] = np.quantile(vals, q)
    features.append(nf_hist)

    # Block 3: Texture (gradient magnitude proxy = 6 dim)
    nf_tex = np.zeros((N, 6), dtype=np.float32)
    for ch in range(3):
        ch_img = img_f[:,:,ch]
        # Simple gradient magnitude via abs diff with neighbors
        grad_h = np.abs(np.diff(ch_img, axis=0))
        grad_w = np.abs(np.diff(ch_img, axis=1))
        # Pad to original size
        gh = np.zeros_like(ch_img); gh[:-1,:] = grad_h
        gw = np.zeros_like(ch_img); gw[:,:-1] = grad_w
        grad = np.sqrt(gh**2 + gw**2)
        for nid in range(N):
            m = snode_map == nid
            if m.sum() < 2: continue
            nf_tex[nid, ch*2] = grad[m].mean()  # mean gradient
            nf_tex[nid, ch*2+1] = grad[m].std()  # std gradient
    features.append(nf_tex)

    # Block 4: Local graph topology (5 dim)
    adj_list = [[] for _ in range(N)]
    for s, d in zip(edge_index[0], edge_index[1]):
        adj_list[s].append(d)

    nf_topo = np.zeros((N, 5), dtype=np.float32)
    for nid in range(N):
        neighbors = adj_list[nid]
        deg = len(neighbors)
        nf_topo[nid, 0] = np.log1p(deg)  # log degree
        nf_topo[nid, 1] = deg / max(N-1, 1)  # normalized degree
        # Clustering coefficient
        if deg >= 2:
            n_set = set(neighbors)
            triangles = sum(1 for i, ni in enumerate(neighbors) for nj in neighbors[i+1:] if nj in set(adj_list[ni]))
            nf_topo[nid, 2] = 2 * triangles / (deg * (deg - 1))
        # Neighbor area statistics
        if neighbors:
            n_areas = area[neighbors]
            nf_topo[nid, 3] = np.mean(n_areas) / max(area.mean(), 1)  # relative neighbor area
            nf_topo[nid, 4] = np.std(n_areas) / max(area.std(), 1) if area.std() > 0 else 0
    features.append(nf_topo)

    # Block 5: WL hash features (3 iterations, homogeneity = 3 dim)
    nf_wl = np.zeros((N, 3), dtype=np.float32)
    wl_labels = [len(adj_list[i]) for i in range(N)]  # init = degree
    for it in range(3):
        new_labels = []
        for i in range(N):
            nb_labels = sorted(wl_labels[j] for j in adj_list[i])
            h = int(hashlib.md5(str((wl_labels[i], tuple(nb_labels))).encode()).hexdigest()[:8], 16)
            new_labels.append(h)
        wl_labels = new_labels
        # Homogeneity: fraction of neighbors with same label
        for i in range(N):
            if adj_list[i]:
                nf_wl[i, it] = sum(1 for j in adj_list[i] if wl_labels[j] == wl_labels[i]) / len(adj_list[i])
    features.append(nf_wl)

    # Concatenate all: 14 + 15 + 6 + 5 + 3 = 43 dim
    all_features = np.concatenate(features, axis=1).astype(np.float32)
    all_features = np.nan_to_num(all_features, nan=0.0)

    return Data(
        x=torch.from_numpy(all_features),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=N,
    )


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
    def subgraph_mask(self, ei, N):
        nm = max(1, int(N * self.mask_ratio))
        adj = [[] for _ in range(N)]
        for s, d in zip(ei[0].cpu().tolist(), ei[1].cpu().tolist()): adj[s].append(d)
        masked = set(); q = [random.randint(0, N-1)]
        while len(masked) < nm and q:
            n = q.pop(0)
            if n not in masked: masked.add(n); nbrs = adj[n]; random.shuffle(nbrs); q.extend(nbrs)
        if len(masked) < nm:
            rem = list(set(range(N)) - masked); random.shuffle(rem); masked.update(rem[:nm-len(masked)])
        mi = torch.tensor(sorted(masked), dtype=torch.long, device=ei.device)
        mask_set = torch.zeros(N, dtype=torch.bool, device=ei.device)
        mask_set[mi] = True
        ctx_ei = ei[:, ~(mask_set[ei[0]] | mask_set[ei[1]])]
        return mi, ctx_ei
    def forward(self, batch):
        x, ei = batch.x, batch.edge_index; N = x.size(0)
        mi, ctx_ei = self.subgraph_mask(ei, N)
        with torch.no_grad(): target = F.layer_norm(self.tgt(x, ei)[mi], [self.hid])
        xc = x.clone(); xc[mi] = 0; c = self.enc(xc, ctx_ei)
        pred = F.layer_norm(self.pred(c[mi]), [self.hid])
        pl = F.smooth_l1_loss(pred, target)
        ctx_mask = torch.ones(N, dtype=torch.bool, device=x.device); ctx_mask[mi] = False
        ctx_emb = c[ctx_mask]; Nc = ctx_emb.size(0)
        std = torch.sqrt(ctx_emb.var(dim=0)+1e-4); vl = F.relu(1.0-std).mean()
        cm = ctx_emb-ctx_emb.mean(0); cov = (cm.T@cm)/max(Nc-1,1)
        od = cov.flatten()[1:].view(self.hid-1,self.hid+1)[:,:-1].flatten()
        cl = (od**2).sum()/self.hid
        return pl + 25*vl + 1*cl, {"pred": pl.item(), "std": std.mean().item()}
    def encode_graph(self, batch):
        return global_mean_pool(self.enc(batch.x, batch.edge_index), batch.batch)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid", type=int, default=256)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device(f"cuda:{args.gpu}")
    kernel = {"merge_distance": 8, "cut_distance": 80, "delete_small_node_max_size": 2, "delete_large_node_min_size": 9216}

    # Load full STL-10
    print("Loading STL-10...", flush=True)
    stl_tr = datasets.STL10("/home/ud3d4/datasets/stl10", split='train', download=True)
    stl_te = datasets.STL10("/home/ud3d4/datasets/stl10", split='test', download=True)
    N_TR, N_TE = len(stl_tr), len(stl_te)
    all_imgs = [np.array(stl_tr[i][0]) for i in range(N_TR)] + [np.array(stl_te[i][0]) for i in range(N_TE)]
    all_labels = [stl_tr[i][1] for i in range(N_TR)] + [stl_te[i][1] for i in range(N_TE)]
    print(f"  {N_TR} train, {N_TE} test", flush=True)

    # Build enriched graphs
    print("Building enriched graphs (43-dim, no DINO)...", flush=True)
    graphs = []
    for i in range(len(all_imgs)):
        g = build_enriched_graph(all_imgs[i], kernel)
        if g is not None:
            g.y = torch.tensor([all_labels[i]], dtype=torch.long)
            graphs.append((i, g))
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(all_imgs)}", flush=True)
    graphs.sort(key=lambda x: x[0])
    graphs = [g for _, g in graphs]
    print(f"  Built {len(graphs)} graphs, feature dim: {graphs[0].x.shape[1]}", flush=True)

    train_graphs = graphs[:N_TR]
    test_graphs = graphs[N_TR:]
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False, num_workers=0)

    in_dim = graphs[0].x.shape[1]

    # Raw baseline
    raw_tr_f, raw_tr_l = [], []
    for b in train_loader: raw_tr_f.append(global_mean_pool(b.x, b.batch).numpy()); raw_tr_l.append(b.y.numpy().ravel())
    raw_te_f, raw_te_l = [], []
    for b in test_loader: raw_te_f.append(global_mean_pool(b.x, b.batch).numpy()); raw_te_l.append(b.y.numpy().ravel())
    Xr_tr, yr_tr = np.concatenate(raw_tr_f), np.concatenate(raw_tr_l)
    Xr_te, yr_te = np.concatenate(raw_te_f), np.concatenate(raw_te_l)
    acc_raw = accuracy_score(yr_te, LogisticRegression(max_iter=2000).fit(Xr_tr, yr_tr).predict(Xr_te))
    print(f"\nRaw enriched features ({in_dim}-dim): {acc_raw*100:.1f}%", flush=True)

    # Train JEPA
    print(f"\nJEPA training ({args.epochs} epochs, hid={args.hid})...", flush=True)
    model = IGJEPA(in_dim, args.hid).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    ckpt_path = f"/home/ud3d4/Desktop/Projects/NIPS 26/cache/best_graphonly_enriched.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train(); tl, n = 0, 0
        for b in train_loader:
            b = b.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss, info = model(b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); model.ema_update()
            tl += loss.item(); n += 1
        sch.step()
        avg = tl/n
        if avg < best_loss - 0.001:
            best_loss = avg
            torch.save({"epoch": ep, "model": model.state_dict(), "loss": avg}, ckpt_path)
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | Loss: {avg:.4f} | std: {info['std']:.4f}", flush=True)

    # Load best
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Best model: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}", flush=True)

    # Evaluate
    model.eval()
    @torch.no_grad()
    def extract(loader):
        f, l = [], []
        for b in loader: b=b.to(device); f.append(model.encode_graph(b).cpu().numpy()); l.append(b.y.cpu().numpy().ravel())
        return np.concatenate(f), np.concatenate(l)

    Xj_tr, yj_tr = extract(train_loader)
    Xj_te, yj_te = extract(test_loader)
    acc_jepa = accuracy_score(yj_te, LogisticRegression(max_iter=2000).fit(Xj_tr, yj_tr).predict(Xj_te))

    print(f"\n{'='*50}", flush=True)
    print(f"GRAPH-ONLY ABLATION (STL-10, no DINO)", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Feature dim:                {in_dim}", flush=True)
    print(f"  Raw features + LogReg:      {acc_raw*100:.1f}%", flush=True)
    print(f"  JEPA + LogReg:              {acc_jepa*100:.1f}%", flush=True)
    print(f"  Previous (14-dim):          18.9%", flush=True)
    print(f"  Improvement:                +{(acc_jepa - 0.189)*100:.1f}pp over previous", flush=True)

    sig = {"raw": acc_raw, "jepa": acc_jepa, "in_dim": in_dim, "prev": 0.189}
    with open("/home/ud3d4/Desktop/Projects/NIPS 26/signals/graphonly_enriched.json", "w") as f:
        json.dump(sig, f, indent=2)
    print(f"Signal saved", flush=True)


if __name__ == "__main__":
    main()
