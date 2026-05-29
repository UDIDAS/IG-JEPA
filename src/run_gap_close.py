"""
Close the accuracy gap on CIFAR-10 and TinyImageNet.

Changes from run_dino.py:
  1. Larger model: hid=512 (4.1M params vs 2.5M)
  2. Longer training: 200 epochs
  3. Graph augmentation: random edge drop + node feature noise during JEPA
  4. Better pooling: attention-weighted pool instead of mean pool
  5. Warm restarts: cosine annealing with warm restarts
"""

import os, sys, copy, argparse, random, time, hashlib, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from concurrent.futures import ThreadPoolExecutor, as_completed
import fastloops

EDGE_MERGED = 0b0001_0000
CACHE_DIR = "/tmp/igjepa_cache"

KERNELS = {
    "cifar10": {"merge_distance": 5, "cut_distance": 80,
                "delete_small_node_max_size": 0, "delete_large_node_min_size": 1024},
    "stl10": {"merge_distance": 8, "cut_distance": 80,
              "delete_small_node_max_size": 2, "delete_large_node_min_size": 9216},
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


def build_graph_with_dino(img_np, kernel, dino_grid):
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


# ── Model (scaled up) ──

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim, hid, n_layers=6, n_heads=8, dropout=0.1):
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
            x = norm(conv(x, edge_index) + x)
            x = F.gelu(x)
            x = self.dropout(x)
        return x


class AttentionPool(nn.Module):
    """Attention-weighted graph pooling instead of mean pool."""
    def __init__(self, hid):
        super().__init__()
        self.attn = nn.Linear(hid, 1)

    def forward(self, x, batch):
        weights = torch.softmax(self.attn(x), dim=0)  # per-node attention
        return global_add_pool(x * weights, batch)


class IGJEPA_V2(nn.Module):
    def __init__(self, in_dim, hid=512, n_layers=6, n_heads=8, mask_ratio=0.4, ema=0.996):
        super().__init__()
        self.mask_ratio, self.ema_m, self.hid = mask_ratio, ema, hid
        self.enc = GraphTransformerEncoder(in_dim, hid, n_layers, n_heads)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters(): p.requires_grad = False
        self.pred = nn.Sequential(
            nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid),
            nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid),
            nn.Linear(hid, hid))
        self.pool = AttentionPool(hid)

    def topology_mask(self, edge_index, N):
        """BFS-based topology-aware masking."""
        nm = max(1, int(N * self.mask_ratio))
        adj_list = [[] for _ in range(N)]
        ei = edge_index.cpu().numpy()
        for s, d in zip(ei[0], ei[1]): adj_list[s].append(d)
        seed = random.randint(0, N-1)
        visited = set(); queue = [seed]
        while len(visited) < nm and queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                neighbors = adj_list[node]
                random.shuffle(neighbors)
                queue.extend(neighbors)
        if len(visited) < nm:
            remaining = list(set(range(N)) - visited)
            random.shuffle(remaining)
            visited.update(remaining[:nm - len(visited)])
        return torch.tensor(list(visited)[:nm], device=edge_index.device)

    def augment_graph(self, x, edge_index):
        """Graph augmentation: edge drop + feature noise."""
        # Random edge drop (10%)
        E = edge_index.size(1)
        mask = torch.rand(E, device=edge_index.device) > 0.1
        edge_index = edge_index[:, mask]
        # Node feature noise (small Gaussian)
        x = x + torch.randn_like(x) * 0.01
        return x, edge_index

    @torch.no_grad()
    def ema_update(self):
        for a, b in zip(self.enc.parameters(), self.tgt.parameters()):
            b.data.mul_(self.ema_m).add_(a.data, alpha=1-self.ema_m)

    def forward(self, batch):
        x, ei = batch.x, batch.edge_index
        N = x.size(0)

        # Augment for context encoder only
        x_aug, ei_aug = self.augment_graph(x, ei)

        mi = self.topology_mask(ei, N)
        with torch.no_grad():
            target = F.layer_norm(self.tgt(x, ei)[mi], [self.hid])

        xc = x_aug.clone(); xc[mi] = 0
        c = self.enc(xc, ei_aug)
        pred = F.layer_norm(self.pred(c[mi]), [self.hid])

        pl = F.smooth_l1_loss(pred, target)
        std = torch.sqrt(c.var(dim=0) + 1e-4)
        vl = F.relu(1.0 - std).mean()
        cm = c - c.mean(0)
        cov = (cm.T @ cm) / max(N-1, 1)
        od = cov.flatten()[1:].view(self.hid-1, self.hid+1)[:, :-1].flatten()
        cl = (od**2).sum() / self.hid

        return pl + 25*vl + 1*cl, {"pred": pl.item(), "std": std.mean().item()}

    def encode_graph(self, batch):
        return self.pool(self.enc(batch.x, batch.edge_index), batch.batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "stl10", "tinyimagenet"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hid", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    kernel = KERNELS[args.dataset]
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset}, hid={args.hid}, layers={args.n_layers}, epochs={args.epochs}", flush=True)

    # Load cached graphs (from run_dino.py)
    ck = hashlib.md5(f"{args.dataset}_None_None_dino398".encode()).hexdigest()[:12]
    cache_dir = os.path.join(CACHE_DIR, f"graphs398_{ck}")

    if os.path.isdir(cache_dir) and os.path.exists(os.path.join(cache_dir, "meta.pt")):
        print(f"Loading cached graphs: {cache_dir}", flush=True)
        meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
        train_graphs = [torch.load(os.path.join(cache_dir, f"train_{i}.pt"), weights_only=False) for i in range(meta["n_train"])]
        test_graphs = [torch.load(os.path.join(cache_dir, f"test_{i}.pt"), weights_only=False) for i in range(meta["n_test"])]
        train_labels = meta["train_y"]
        test_labels = meta["test_y"]
        print(f"  {len(train_graphs)} train, {len(test_graphs)} test", flush=True)
    else:
        print(f"ERROR: No cached graphs at {cache_dir}. Run run_dino.py first.", flush=True)
        sys.exit(1)

    in_dim = train_graphs[0].x.shape[1]
    print(f"  Feature dim: {in_dim}, Avg nodes: {np.mean([g.num_nodes for g in train_graphs]):.0f}", flush=True)

    train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=args.bs, num_workers=0)

    # Build model
    model = IGJEPA_V2(in_dim, args.hid, args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sch = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    ckpt_path = os.path.join(CACHE_DIR, f"best_v2_{args.dataset}.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        tl, n = 0, 0
        for b in train_loader:
            b = b.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss, info = model(b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            model.ema_update()
            tl += loss.item(); n += 1
        sch.step()
        avg_loss = tl / n

        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            torch.save({"epoch": ep, "model": model.state_dict(), "loss": avg_loss}, ckpt_path)

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | Loss: {avg_loss:.4f} | std: {info['std']:.4f}", flush=True)

    # Load best
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Best model from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})", flush=True)

    # Evaluate
    print(f"\nEvaluation...", flush=True)
    model.eval()

    @torch.no_grad()
    def extract(loader):
        f, l = [], []
        for b in loader:
            b = b.to(device); f.append(model.encode_graph(b).cpu()); l.append(b.y.cpu())
        return torch.cat(f).numpy(), torch.cat(l).numpy().ravel()

    def extract_raw(loader):
        f, l = [], []
        for b in loader: f.append(global_mean_pool(b.x, b.batch)); l.append(b.y)
        return torch.cat(f).numpy(), torch.cat(l).numpy().ravel()

    Xj_tr, y_tr = extract(train_loader)
    Xj_te, y_te = extract(test_loader)
    Xr_tr, _ = extract_raw(train_loader)
    Xr_te, _ = extract_raw(test_loader)

    clf_j = LogisticRegression(max_iter=2000); clf_j.fit(Xj_tr, y_tr)
    clf_r = LogisticRegression(max_iter=2000); clf_r.fit(Xr_tr, y_tr)
    acc_j = accuracy_score(y_te, clf_j.predict(Xj_te))
    acc_r = accuracy_score(y_te, clf_r.predict(Xr_te))

    # MLP probe
    n_cls = len(np.unique(y_tr))
    mlp = nn.Sequential(
        nn.Linear(args.hid, args.hid), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(args.hid, args.hid//2), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(args.hid//2, n_cls)).to(device)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xj_tr, dtype=torch.float).to(device)
    yt = torch.tensor(y_tr, dtype=torch.long).to(device)
    mlp.train()
    for _ in range(200):
        mlp_opt.zero_grad()
        F.cross_entropy(mlp(Xt), yt).backward()
        mlp_opt.step()
    mlp.eval()
    with torch.no_grad():
        acc_mlp = accuracy_score(y_te, mlp(torch.tensor(Xj_te, dtype=torch.float).to(device)).argmax(1).cpu().numpy())

    print(f"\n{'='*55}", flush=True)
    print(f"RESULTS — {args.dataset} (V2: hid={args.hid}, layers={args.n_layers})", flush=True)
    print(f"{'='*55}", flush=True)
    print(f"  Raw features + LogReg:  {acc_r*100:.2f}%", flush=True)
    print(f"  JEPA + LogReg:          {acc_j*100:.2f}%", flush=True)
    print(f"  JEPA + MLP:             {acc_mlp*100:.2f}%", flush=True)
    print(f"  Previous best:          {'65.0' if args.dataset=='cifar10' else '37.2' if args.dataset=='tinyimagenet' else '81.7'}%", flush=True)

    sig = {"acc_raw": acc_r, "acc_jepa_lr": acc_j, "acc_jepa_mlp": acc_mlp,
           "dataset": args.dataset, "hid": args.hid, "n_layers": args.n_layers,
           "epochs": args.epochs, "params": n_params}
    sig_path = f"/home/ud3d4/Desktop/NIPS 26/signals/gap_close_{args.dataset}.json"
    with open(sig_path, "w") as f:
        json.dump(sig, f, indent=2)
    print(f"Signal: {sig_path}", flush=True)


if __name__ == "__main__":
    main()
