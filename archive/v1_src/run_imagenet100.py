"""
ImageNet-100 benchmark for IG-JEPA.

Uses HuggingFace imagenet-100 dataset. Same pipeline as run_dino.py:
graph-minor pooling + DINO features + JEPA + linear probe.
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
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import fastloops
import json

EDGE_MERGED = 0b0001_0000
CACHE_DIR = "/home/ud3d4/Desktop/Projects/NIPS 26/cache"

# ImageNet-100 images are 224x224 — use appropriate kernel
KERNEL = {"merge_distance": 10, "cut_distance": 80, "delete_small_node_max_size": 3, "delete_large_node_min_size": 50176}


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


def build_graph_with_dino(img_np, dino_grid):
    if img_np.ndim == 2: img_np = np.stack([img_np]*3, axis=-1)
    H, W = img_np.shape[:2]
    adj, feat = fastloops.merge_and_cut(img_np, **KERNEL)
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
    flat = snode_map.ravel(); valid = flat >= 0
    counts = np.bincount(flat[valid], minlength=N).astype(np.float64).clip(min=1)

    nf = np.zeros((N, 398), dtype=np.float32)
    for ch in range(3):
        fc = img_f[:,:,ch].ravel()[valid]
        sums = np.bincount(flat[valid], weights=fc, minlength=N)
        nf[:, ch*2] = sums / counts
        sq = np.bincount(flat[valid], weights=fc**2, minlength=N)
        nf[:, ch*2+1] = np.sqrt((sq/counts - (sums/counts)**2).clip(min=0))

    ff = feat.astype(np.float64); area = ff[:,0].clip(min=1)
    nf[:,6]=np.log1p(area); nf[:,7]=ff[:,1]/area/W; nf[:,8]=ff[:,2]/area/H
    nf[:,9]=(ff[:,10]-ff[:,9])/W; nf[:,10]=(ff[:,12]-ff[:,11])/H; nf[:,11]=ff[:,13]/area
    bbox_a = ((ff[:,10]-ff[:,9]+1)*(ff[:,12]-ff[:,11]+1)).clip(min=1)
    nf[:,12]=(area/bbox_a).clip(max=1); nf[:,13]=area/(H*W)

    if dino_grid is not None:
        cx, cy = nf[:,7], nf[:,8]
        px = np.clip((cx * 14).astype(np.int32), 0, 13)
        py = np.clip((cy * 14).astype(np.int32), 0, 13)
        nf[:, 14:] = dino_grid[py, px, :].astype(np.float32)

    nf = np.nan_to_num(nf, nan=0.0)
    return Data(x=torch.from_numpy(nf), edge_index=torch.tensor(np.stack([s_arr,d_arr]), dtype=torch.long), num_nodes=N)


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
        return pl+25*vl+1*cl, {"pred": pl.item(), "std": std.mean().item()}
    def encode_graph(self, batch):
        return global_mean_pool(self.enc(batch.x, batch.edge_index), batch.batch)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=5000)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Device: {device}", flush=True)

    # Load ImageNet-100
    print("Loading ImageNet-100...", flush=True)
    from datasets import load_dataset as hf_load
    ds = hf_load("clane9/imagenet-100", cache_dir="/home/ud3d4/datasets/imagenet100")
    tr_ds = ds['train']; te_ds = ds['validation']

    n_tr = min(args.n_train, len(tr_ds))
    n_te = min(args.n_test, len(te_ds))
    print(f"  Train: {n_tr}/{len(tr_ds)}, Test: {n_te}/{len(te_ds)}", flush=True)
    print(f"  Classes: {len(set(tr_ds['label'][:n_tr]))}", flush=True)

    # Load images
    print("Loading images...", flush=True)
    tr_imgs, tr_labels = [], []
    for i in range(n_tr):
        img = tr_ds[i]['image']
        if hasattr(img, 'convert'): img = img.convert('RGB')
        img = img.resize((224, 224))
        tr_imgs.append(np.array(img))
        tr_labels.append(tr_ds[i]['label'])

    te_imgs, te_labels = [], []
    for i in range(n_te):
        img = te_ds[i]['image']
        if hasattr(img, 'convert'): img = img.convert('RGB')
        img = img.resize((224, 224))
        te_imgs.append(np.array(img))
        te_labels.append(te_ds[i]['label'])

    all_imgs = tr_imgs + te_imgs
    all_labels = tr_labels + te_labels
    print(f"  Loaded {len(all_imgs)} images", flush=True)

    # DINO extraction
    print("DINO extraction...", flush=True)
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True).eval().to(device)
    dino_tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    all_patches = np.zeros((len(all_imgs), 14, 14, 384), dtype=np.float16)
    BS = 128
    for i in range(0, len(all_imgs), BS):
        batch = [dino_tf(img if img.ndim == 3 else np.stack([img]*3, axis=-1)) for img in all_imgs[i:i+BS]]
        with torch.no_grad():
            tokens = dino.get_intermediate_layers(torch.stack(batch).to(device), n=1)[0]
            all_patches[i:i+len(batch)] = tokens[:, 1:, :].reshape(-1, 14, 14, 384).cpu().numpy().astype(np.float16)
        if (i+BS) % 2560 < BS: print(f"  DINO: {min(i+BS, len(all_imgs))}/{len(all_imgs)}", flush=True)
    del dino; torch.cuda.empty_cache()
    print("DINO done", flush=True)

    # Build graphs
    print("Building graphs...", flush=True)
    def build_one(i):
        return build_graph_with_dino(all_imgs[i], all_patches[i])

    graphs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(build_one, i): i for i in range(len(all_imgs))}
        done = 0
        for f in as_completed(futures):
            idx = futures[f]; g = f.result()
            if g is not None:
                g.y = torch.tensor([all_labels[idx]], dtype=torch.long)
                graphs.append((idx, g))
            done += 1
            if done % 2000 == 0: print(f"  Graphs: {done}/{len(all_imgs)}", flush=True)
    graphs.sort(key=lambda x: x[0])
    graphs = [g for _, g in graphs]
    print(f"  Built {len(graphs)} graphs, avg nodes: {np.mean([g.num_nodes for g in graphs]):.0f}", flush=True)

    train_graphs = graphs[:n_tr]
    test_graphs = graphs[n_tr:]
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False, num_workers=0)

    in_dim = graphs[0].x.shape[1]

    # Raw baseline
    raw_tr_f, raw_tr_l = [], []
    for b in train_loader: raw_tr_f.append(global_mean_pool(b.x, b.batch).numpy()); raw_tr_l.append(b.y.numpy().ravel())
    raw_te_f, raw_te_l = [], []
    for b in test_loader: raw_te_f.append(global_mean_pool(b.x, b.batch).numpy()); raw_te_l.append(b.y.numpy().ravel())
    acc_raw = accuracy_score(np.concatenate(raw_te_l),
                             LogisticRegression(max_iter=2000).fit(np.concatenate(raw_tr_f), np.concatenate(raw_tr_l)).predict(np.concatenate(raw_te_f)))
    print(f"\nRaw features: {acc_raw*100:.1f}%", flush=True)

    # Train JEPA
    print(f"\nJEPA training ({args.epochs} epochs)...", flush=True)
    model = IGJEPA(in_dim, args.hid).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    ckpt_path = os.path.join(CACHE_DIR, "best_imagenet100.pt")

    for ep in range(1, args.epochs + 1):
        model.train(); tl, n = 0, 0
        for b in train_loader:
            b = b.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss, info = model(b)
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

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Best: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}", flush=True)

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
    print(f"IMAGENET-100 RESULTS", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Raw features + LogReg:  {acc_raw*100:.1f}%", flush=True)
    print(f"  JEPA + LogReg:          {acc_jepa*100:.1f}%", flush=True)
    print(f"  Classes: 100, Train: {n_tr}, Test: {n_te}", flush=True)

    sig = {"acc_raw": acc_raw, "acc_jepa": acc_jepa, "n_train": n_tr, "n_test": n_te,
           "n_classes": 100, "dataset": "imagenet-100"}
    with open("/home/ud3d4/Desktop/Projects/NIPS 26/signals/imagenet100.json", "w") as f:
        json.dump(sig, f, indent=2)
    print(f"Signal saved", flush=True)


if __name__ == "__main__":
    main()
