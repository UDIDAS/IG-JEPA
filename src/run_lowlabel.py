"""
Low-label transfer experiments for CIFAR-10 and TinyImageNet.

Reuses cached DINO graphs. Trains JEPA once, then evaluates linear probe
at 1%, 5%, 10%, and 100% label fractions.
"""

import os, sys, copy, random, time, hashlib, json
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

CACHE_DIR = "/tmp/igjepa_cache"


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
        od = cov.flatten()[1:].view(self.hid-1,self.hid+1)[:,:-1].flatten(); cl = (od**2).sum()/self.hid
        return pl+25*vl+1*cl
    def encode_graph(self, batch):
        return global_mean_pool(self.enc(batch.x, batch.edge_index), batch.batch)


def load_cached_graphs(dataset_name):
    """Load graphs from sharded cache."""
    ck = hashlib.md5(f"{dataset_name}_None_None_dino398".encode()).hexdigest()[:12]
    cache_dir = os.path.join(CACHE_DIR, f"graphs398_{ck}")

    if not os.path.isdir(cache_dir):
        print(f"  Cache not found: {cache_dir}", flush=True)
        return None, None, None, None

    meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
    train_graphs = [torch.load(os.path.join(cache_dir, f"train_{i}.pt"), weights_only=False) for i in range(meta["n_train"])]
    test_graphs = [torch.load(os.path.join(cache_dir, f"test_{i}.pt"), weights_only=False) for i in range(meta["n_test"])]
    return train_graphs, test_graphs, meta["train_y"], meta["test_y"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "stl10", "tinyimagenet"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--bs", type=int, default=32)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)

    # Load cached graphs
    print("Loading cached graphs...", flush=True)
    train_graphs, test_graphs, train_labels, test_labels = load_cached_graphs(args.dataset)
    if train_graphs is None:
        print("ERROR: No cached graphs found. Run run_dino.py first.", flush=True)
        return

    for i, g in enumerate(train_graphs): g.y = torch.tensor([train_labels[i]], dtype=torch.long)
    for i, g in enumerate(test_graphs): g.y = torch.tensor([test_labels[i]], dtype=torch.long)
    print(f"  Train: {len(train_graphs)}, Test: {len(test_graphs)}", flush=True)

    in_dim = train_graphs[0].x.shape[1]
    train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=args.bs, num_workers=0)

    # Train JEPA
    print(f"\nJEPA pretraining ({args.epochs} epochs)...", flush=True)
    model = IGJEPA(in_dim, args.hid).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    for ep in range(1, args.epochs+1):
        model.train()
        tl, n = 0, 0
        for b in train_loader:
            b = b.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = model(b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); model.ema_update()
            tl += loss.item(); n += 1
        sch.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | Loss: {tl/n:.4f}", flush=True)

    # Extract embeddings once
    print("\nExtracting embeddings...", flush=True)
    model.eval()
    @torch.no_grad()
    def extract(loader):
        f, l = [], []
        for b in loader:
            b = b.to(device); f.append(model.encode_graph(b).cpu()); l.append(b.y.cpu())
        return torch.cat(f).numpy(), torch.cat(l).numpy().ravel()

    X_train, y_train = extract(train_loader)
    X_test, y_test = extract(test_loader)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}", flush=True)

    # Low-label evaluation
    print(f"\n{'='*50}", flush=True)
    print(f"LOW-LABEL TRANSFER — {args.dataset}", flush=True)
    print(f"{'='*50}", flush=True)

    fractions = [0.01, 0.05, 0.10, 1.0]
    results = {}

    for frac in fractions:
        n_samples = max(1, int(len(X_train) * frac))
        # Stratified sampling
        classes = np.unique(y_train)
        per_class = max(1, n_samples // len(classes))

        idx = []
        for c in classes:
            c_idx = np.where(y_train == c)[0]
            np.random.shuffle(c_idx)
            idx.extend(c_idx[:per_class].tolist())
        idx = idx[:n_samples]

        X_sub = X_train[idx]
        y_sub = y_train[idx]

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_sub, y_sub)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[f"{frac}"] = acc
        pct = int(frac * 100)
        print(f"  {pct:3d}% labels ({len(idx):6d} samples): {acc*100:.2f}%", flush=True)

    # Also raw feature baseline at 100%
    def extract_raw(loader):
        f, l = [], []
        for b in loader: f.append(global_mean_pool(b.x, b.batch)); l.append(b.y)
        return torch.cat(f).numpy(), torch.cat(l).numpy().ravel()

    X_raw_tr, _ = extract_raw(train_loader)
    X_raw_te, _ = extract_raw(test_loader)
    clf_raw = LogisticRegression(max_iter=2000); clf_raw.fit(X_raw_tr, y_train)
    acc_raw = accuracy_score(y_test, clf_raw.predict(X_raw_te))
    print(f"\n  Raw features (100%): {acc_raw*100:.2f}%", flush=True)

    # Save signal
    sig = {"dataset": args.dataset, "low_label": results, "raw_100": acc_raw}
    sig_path = f"/home/ud3d4/Desktop/NIPS 26/signals/lowlabel_{args.dataset}.json"
    with open(sig_path, "w") as f:
        json.dump(sig, f, indent=2)
    print(f"\nSignal: {sig_path}", flush=True)


if __name__ == "__main__":
    main()
