"""
IG-JEPA with DINO features — single-pass pipeline.

No separate precompute/attach steps. Builds 398-dim graphs directly:
1. Load all images into numpy array (fast)
2. Run ALL images through DINO in one batched pass (GPU, fast)
3. Build graphs with graph-minor + attach DINO by centroid mapping (CPU, parallel)
4. Train IG-JEPA
5. Evaluate

Everything cached as final 398-dim graphs.
"""

import os, sys, copy, argparse, random, time, hashlib, pickle
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
    """Build one graph with 14-dim boundary + 384-dim DINO = 398-dim per node."""
    if img_np.ndim == 2: img_np = np.stack([img_np]*3, axis=-1)
    H, W = img_np.shape[:2]
    adj, feat = fastloops.merge_and_cut(img_np, **kernel)
    N = feat.shape[0]
    if N < 2: return None

    labels = region_labels(adj)
    canon = feat[:, 14:16].astype(np.int64)
    ml = int(labels.max()) + 1
    cc2s = np.full(ml, -1, dtype=np.int32)
    for i in range(N): cc2s[labels[int(canon[i,0]), int(canon[i,1])]] = i
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
    s = np.concatenate([u[:,0], u[:,1]]); d = np.concatenate([u[:,1], u[:,0]])

    # Boundary features (14-dim)
    img_f = img_np.astype(np.float64) / 255.0
    ff = feat.astype(np.float64); area = ff[:,0].clip(min=1)
    flat_map = snode_map.ravel(); valid = flat_map >= 0
    counts = np.bincount(flat_map[valid], minlength=N).astype(np.float64).clip(min=1)

    nf = np.zeros((N, 398), dtype=np.float32)
    for ch in range(3):
        flat_ch = img_f[:,:,ch].ravel()[valid]
        sums = np.bincount(flat_map[valid], weights=flat_ch, minlength=N)
        means = sums / counts
        sq_sums = np.bincount(flat_map[valid], weights=flat_ch**2, minlength=N)
        stds = np.sqrt((sq_sums / counts - means**2).clip(min=0))
        nf[:, ch*2] = means; nf[:, ch*2+1] = stds

    nf[:, 6] = np.log1p(area)
    nf[:, 7] = ff[:,1] / area / W  # centroid_x
    nf[:, 8] = ff[:,2] / area / H  # centroid_y
    nf[:, 9] = (ff[:,10]-ff[:,9]) / W
    nf[:, 10] = (ff[:,12]-ff[:,11]) / H
    nf[:, 11] = ff[:,13] / area
    bbox_area = ((ff[:,10]-ff[:,9]+1)*(ff[:,12]-ff[:,11]+1)).clip(min=1)
    nf[:, 12] = (area / bbox_area).clip(max=1)
    nf[:, 13] = area / (H*W)

    # DINO features (384-dim) — map centroid to 14x14 patch grid
    if dino_grid is not None:
        cx = nf[:, 7]  # already normalized [0,1]
        cy = nf[:, 8]
        px = np.clip((cx * 14).astype(np.int32), 0, 13)
        py = np.clip((cy * 14).astype(np.int32), 0, 13)
        nf[:, 14:] = dino_grid[py, px, :].astype(np.float32)

    nf = np.nan_to_num(nf, nan=0.0, posinf=1.0, neginf=-1.0)

    return Data(x=torch.from_numpy(nf), edge_index=torch.tensor(np.stack([s,d]), dtype=torch.long), num_nodes=N)


# Model (same as run_tier1.py)
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
        for a, b in zip(self.enc.parameters(), self.tgt.parameters()): b.data.mul_(self.ema_m).add_(a.data, alpha=1-self.ema_m)
    def topology_mask(self, ei, N):
        nm = max(1, int(N * self.mask_ratio))
        adj = [[] for _ in range(N)]
        for s, d in zip(ei[0].cpu().tolist(), ei[1].cpu().tolist()): adj[s].append(d)
        masked = set(); q = [random.randint(0, N-1)]
        while len(masked) < nm and q:
            n = q.pop(0)
            if n not in masked: masked.add(n); nbrs = adj[n]; random.shuffle(nbrs); q.extend(nbrs)
        if len(masked) < nm:
            rem = list(set(range(N)) - masked); random.shuffle(rem); masked.update(rem[:nm-len(masked)])
        return torch.tensor(list(masked), dtype=torch.long, device=ei.device)
    def forward(self, batch):
        x, ei = batch.x, batch.edge_index; N = x.size(0)
        mi = self.topology_mask(ei, N)
        with torch.no_grad(): target = F.layer_norm(self.tgt(x, ei)[mi], [self.hid])
        xc = x.clone(); xc[mi] = 0; c = self.enc(xc, ei)
        pred = F.layer_norm(self.pred(c[mi]), [self.hid])
        pl = F.smooth_l1_loss(pred, target)
        std = torch.sqrt(c.var(dim=0)+1e-4); vl = F.relu(1.0-std).mean()
        cm = c-c.mean(0); cov = (cm.T@cm)/max(N-1,1)
        od = cov.flatten()[1:].view(self.hid-1,self.hid+1)[:,:-1].flatten(); cl = (od**2).sum()/self.hid
        return pl+25*vl+1*cl, {"pred":pl.item(),"std":std.mean().item()}
    def encode_graph(self, batch):
        return global_mean_pool(self.enc(batch.x, batch.edge_index), batch.batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "stl10", "tinyimagenet"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    kernel = KERNELS[args.dataset]
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)

    # Check for cached final graphs (sharded .pt format for fast loading)
    ck = hashlib.md5(f"{args.dataset}_{args.n_train}_{args.n_test}_dino398".encode()).hexdigest()[:12]
    cache_dir_shard = os.path.join(CACHE_DIR, f"graphs398_{ck}")
    # Also check legacy single-pickle format
    cache_path_legacy = os.path.join(CACHE_DIR, f"graphs398_{ck}.pkl")

    if os.path.isdir(cache_dir_shard) and os.path.exists(os.path.join(cache_dir_shard, "meta.pt")):
        print(f"Loading cached sharded graphs: {cache_dir_shard}", flush=True)
        meta = torch.load(os.path.join(cache_dir_shard, "meta.pt"), weights_only=False)
        train_graphs = [torch.load(os.path.join(cache_dir_shard, f"train_{i}.pt"), weights_only=False) for i in range(meta["n_train"])]
        test_graphs = [torch.load(os.path.join(cache_dir_shard, f"test_{i}.pt"), weights_only=False) for i in range(meta["n_test"])]
        train_labels, test_labels = meta["train_y"], meta["test_y"]
        print(f"  Loaded {len(train_graphs)} train, {len(test_graphs)} test", flush=True)
    elif os.path.exists(cache_path_legacy):
        print(f"Loading legacy pickle cache: {cache_path_legacy}", flush=True)
        with open(cache_path_legacy, "rb") as f:
            cached = pickle.load(f)
        train_graphs, test_graphs, train_labels, test_labels = cached["train"], cached["test"], cached["train_y"], cached["test_y"]
        # Convert to sharded format for next time
        print(f"  Converting to sharded format...", flush=True)
        os.makedirs(cache_dir_shard, exist_ok=True)
        for i, g in enumerate(train_graphs): torch.save(g, os.path.join(cache_dir_shard, f"train_{i}.pt"))
        for i, g in enumerate(test_graphs): torch.save(g, os.path.join(cache_dir_shard, f"test_{i}.pt"))
        torch.save({"n_train": len(train_graphs), "n_test": len(test_graphs),
                     "train_y": train_labels, "test_y": test_labels},
                    os.path.join(cache_dir_shard, "meta.pt"))
        os.remove(cache_path_legacy)
        print(f"  Sharded cache saved, legacy pickle deleted", flush=True)
    else:
        # Step 1: Load images
        print("Step 1: Loading images...", flush=True)
        from torchvision import datasets, transforms
        data_dir = f"/home/ud3d4/datasets/{args.dataset}"
        if args.dataset == "cifar10":
            try:
                tr_ds = datasets.CIFAR10(data_dir, train=True, download=True)
                te_ds = datasets.CIFAR10(data_dir, train=False, download=True)
            except:
                from datasets import load_dataset as hf_load
                hf_tr = hf_load('cifar10', split='train', cache_dir='/home/ud3d4/datasets/hf_cifar10')
                hf_te = hf_load('cifar10', split='test', cache_dir='/home/ud3d4/datasets/hf_cifar10')
                class HFW:
                    def __init__(self, d): self.d = d
                    def __len__(self): return len(self.d)
                    def __getitem__(self, i): return self.d[i]['img'], self.d[i]['label']
                tr_ds, te_ds = HFW(hf_tr), HFW(hf_te)
        elif args.dataset == "stl10":
            tr_ds = datasets.STL10(data_dir, split='train', download=True)
            te_ds = datasets.STL10(data_dir, split='test', download=True)
        elif args.dataset == "tinyimagenet":
            from datasets import load_dataset as hf_load
            hf_tr = hf_load('Maysee/tiny-imagenet', split='train', cache_dir='/home/ud3d4/datasets/tinyimagenet')
            hf_te = hf_load('Maysee/tiny-imagenet', split='valid', cache_dir='/home/ud3d4/datasets/tinyimagenet')
            class HFW_TIN:
                def __init__(self, d): self.d = d
                def __len__(self): return len(self.d)
                def __getitem__(self, i): return self.d[i]['image'], self.d[i]['label']
            tr_ds, te_ds = HFW_TIN(hf_tr), HFW_TIN(hf_te)

        n_tr = min(args.n_train or len(tr_ds), len(tr_ds))
        n_te = min(args.n_test or len(te_ds), len(te_ds))
        print(f"  Train: {n_tr}, Test: {n_te}", flush=True)

        # Load all images as numpy arrays
        t0 = time.time()
        tr_imgs = [np.array(tr_ds[i][0]) for i in range(n_tr)]
        tr_labels = [tr_ds[i][1] for i in range(n_tr)]
        te_imgs = [np.array(te_ds[i][0]) for i in range(n_te)]
        te_labels = [te_ds[i][1] for i in range(n_te)]
        print(f"  Images loaded in {time.time()-t0:.1f}s", flush=True)

        # Step 2: DINO patch extraction (batched GPU)
        print("Step 2: DINO extraction...", flush=True)
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
        dino.eval().to(device)
        dino_tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        def extract_dino_patches(images):
            all_patches = np.zeros((len(images), 14, 14, 384), dtype=np.float16)
            BS = 256
            for i in range(0, len(images), BS):
                batch_imgs = []
                for img in images[i:i+BS]:
                    if img.ndim == 2: img = np.stack([img]*3, axis=-1)
                    batch_imgs.append(dino_tf(img))
                inp = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    tokens = dino.get_intermediate_layers(inp, n=1)[0]
                    patches = tokens[:, 1:, :].reshape(-1, 14, 14, 384).cpu().numpy().astype(np.float16)
                all_patches[i:i+len(batch_imgs)] = patches
                if (i+BS) % 5000 < BS:
                    print(f"    DINO: {min(i+BS, len(images))}/{len(images)}", flush=True)
            return all_patches

        tr_patches = extract_dino_patches(tr_imgs)
        te_patches = extract_dino_patches(te_imgs)
        del dino; torch.cuda.empty_cache()
        print(f"  DINO done: train={tr_patches.shape}, test={te_patches.shape}", flush=True)

        # Step 3: Build graphs with DINO (threaded CPU)
        print("Step 3: Building 398-dim graphs...", flush=True)
        t0 = time.time()

        def build_split(images, patches, labels_list):
            graphs, labels = [], []
            def proc(i):
                g = build_graph_with_dino(images[i], kernel, patches[i])
                return i, g
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(proc, i): i for i in range(len(images))}
                done = 0
                for f in as_completed(futures):
                    idx, g = f.result()
                    if g is not None:
                        g.y = torch.tensor([labels_list[idx]], dtype=torch.long)
                        graphs.append((idx, g))
                    done += 1
                    if done % 5000 == 0:
                        print(f"    {done}/{len(images)}", flush=True)
            graphs.sort(key=lambda x: x[0])
            return [g for _, g in graphs]

        train_graphs = build_split(tr_imgs, tr_patches, tr_labels)
        test_graphs = build_split(te_imgs, te_patches, te_labels)
        train_labels = [g.y.item() for g in train_graphs]
        test_labels = [g.y.item() for g in test_graphs]
        del tr_imgs, te_imgs, tr_patches, te_patches
        print(f"  Graphs built in {time.time()-t0:.1f}s: {len(train_graphs)} train, {len(test_graphs)} test", flush=True)
        print(f"  Feature dim: {train_graphs[0].x.shape[1]}, Nodes avg: {np.mean([g.num_nodes for g in train_graphs]):.0f}", flush=True)

        # Cache (sharded .pt format — fast to load)
        os.makedirs(cache_dir_shard, exist_ok=True)
        for i, g in enumerate(train_graphs): torch.save(g, os.path.join(cache_dir_shard, f"train_{i}.pt"))
        for i, g in enumerate(test_graphs): torch.save(g, os.path.join(cache_dir_shard, f"test_{i}.pt"))
        torch.save({"n_train": len(train_graphs), "n_test": len(test_graphs),
                     "train_y": train_labels, "test_y": test_labels},
                    os.path.join(cache_dir_shard, "meta.pt"))
        cache_size = sum(os.path.getsize(os.path.join(cache_dir_shard, f)) for f in os.listdir(cache_dir_shard))
        print(f"  Cached: {cache_dir_shard} ({cache_size/1e6:.0f} MB, {len(train_graphs)+len(test_graphs)} files)", flush=True)

    # Step 4: Train IG-JEPA
    print(f"\nStep 4: IG-JEPA training ({args.epochs} epochs)...", flush=True)
    in_dim = train_graphs[0].x.shape[1]
    train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=args.bs, num_workers=0)

    model = IGJEPA(in_dim, args.hid).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    patience_counter = 0
    patience = 15
    ckpt_path = os.path.join(CACHE_DIR, f"best_model_{args.dataset}.pt")

    for ep in range(1, args.epochs+1):
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
        avg_loss = tl / n
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({"epoch": ep, "model": model.state_dict(), "loss": avg_loss},
                       ckpt_path)
        else:
            patience_counter += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | Loss: {avg_loss:.4f} | std: {info['std']:.4f} | pat: {patience_counter}/{patience}", flush=True)
        if patience_counter >= patience:
            print(f"  Early stop at epoch {ep} (no improvement for {patience} epochs)", flush=True)
            break

    # Load best model if checkpoint exists
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded best model from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})", flush=True)

    # Step 5: Evaluate
    print("\nStep 5: Evaluation...", flush=True)
    model.eval()
    for p in model.enc.parameters(): p.requires_grad = False

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

    Xj_tr, y_tr = extract(train_loader); Xj_te, y_te = extract(test_loader)
    Xr_tr, _ = extract_raw(train_loader); Xr_te, _ = extract_raw(test_loader)

    clf_j = LogisticRegression(max_iter=2000); clf_j.fit(Xj_tr, y_tr)
    clf_r = LogisticRegression(max_iter=2000); clf_r.fit(Xr_tr, y_tr)
    acc_j = accuracy_score(y_te, clf_j.predict(Xj_te))
    acc_r = accuracy_score(y_te, clf_r.predict(Xr_te))

    # MLP probe
    n_cls = len(np.unique(y_tr))
    mlp = nn.Sequential(nn.Linear(args.hid, args.hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(args.hid, n_cls)).to(device)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    Xt = torch.tensor(Xj_tr, dtype=torch.float).to(device)
    yt = torch.tensor(y_tr, dtype=torch.long).to(device)
    mlp.train()
    for _ in range(100):
        mlp_opt.zero_grad(); F.cross_entropy(mlp(Xt), yt).backward(); mlp_opt.step()
    mlp.eval()
    with torch.no_grad(): acc_mlp = accuracy_score(y_te, mlp(torch.tensor(Xj_te, dtype=torch.float).to(device)).argmax(1).cpu().numpy())

    print(f"\nRESULTS — {args.dataset}", flush=True)
    print(f"  Raw (398-dim) + LogReg:  {acc_r*100:.2f}%", flush=True)
    print(f"  JEPA + LogReg:           {acc_j*100:.2f}%", flush=True)
    print(f"  JEPA + MLP:              {acc_mlp*100:.2f}%", flush=True)

    # Signal
    import json
    sig = {"acc_raw": acc_r, "acc_jepa_lr": acc_j, "acc_jepa_mlp": acc_mlp,
           "dataset": args.dataset, "params": sum(p.numel() for p in model.parameters())}
    with open(f"/home/ud3d4/Desktop/NIPS 26/signals/done_{args.dataset}_dino.json", "w") as f:
        json.dump(sig, f, indent=2)
    print(f"Signal: done_{args.dataset}_dino.json", flush=True)


if __name__ == "__main__":
    main()
