"""
Table 4: Efficiency comparison. Builds a small batch of graphs inline,
measures tokens, memory, GFLOPs, throughput. No cache needed.
"""
import os, sys, time, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import fastloops

EDGE_MERGED = 0b0001_0000
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

def build_graph(img_np, kernel):
    """Build graph with dummy 398-dim features (14 boundary + 384 zeros for speed)."""
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
    nf = np.random.randn(N, 398).astype(np.float32) * 0.1  # dummy features for benchmarking
    return Data(x=torch.from_numpy(nf), edge_index=torch.tensor(np.stack([s_arr,d_arr]), dtype=torch.long), num_nodes=N)

class Enc(nn.Module):
    def __init__(self, in_dim, hid, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.convs = nn.ModuleList([TransformerConv(hid, hid//n_heads, heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
    def forward(self, x, ei):
        x = self.proj(x)
        for c, n in zip(self.convs, self.norms):
            x = n(c(x, ei) + x); x = F.gelu(x); x = self.drop(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "stl10", "tinyimagenet"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_graphs", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    kernel = KERNELS[args.dataset]

    # Load a few images
    from torchvision import datasets
    if args.dataset == "cifar10":
        ds = datasets.CIFAR10("/home/ud3d4/datasets/cifar10", train=True, download=True)
    elif args.dataset == "stl10":
        ds = datasets.STL10("/home/ud3d4/datasets/stl10", split='train', download=True)
    elif args.dataset == "tinyimagenet":
        from datasets import load_dataset as hf_load
        hf = hf_load('Maysee/tiny-imagenet', split='train', cache_dir='/home/ud3d4/datasets/tinyimagenet')
        class W:
            def __init__(self, d): self.d = d
            def __len__(self): return len(self.d)
            def __getitem__(self, i):
                img = self.d[i]['image']
                if hasattr(img, 'convert'): img = img.convert('RGB')
                return np.array(img), self.d[i]['label']
        ds = W(hf)

    print(f"Building {args.n_graphs} graphs for {args.dataset}...", flush=True)
    graphs = []
    for i in range(min(args.n_graphs * 2, len(ds))):
        img = np.array(ds[i][0])
        g = build_graph(img, kernel)
        if g is not None:
            graphs.append(g)
        if len(graphs) >= args.n_graphs:
            break

    avg_nodes = np.mean([g.num_nodes for g in graphs])
    avg_edges = np.mean([g.edge_index.shape[1] for g in graphs])
    print(f"Built {len(graphs)} graphs, avg nodes: {avg_nodes:.0f}, avg edges: {avg_edges:.0f}", flush=True)

    model = Enc(398, 256).eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loader = DataLoader(graphs, batch_size=32, shuffle=False, num_workers=0)

    # Warmup
    for b in loader:
        b = b.to(device)
        with torch.no_grad(): _ = global_mean_pool(model(b.x, b.edge_index), b.batch)
        break

    # Memory
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    b = next(iter(loader)).to(device)
    with torch.no_grad(): _ = global_mean_pool(model(b.x, b.edge_index), b.batch)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    # Throughput (5 passes)
    torch.cuda.synchronize()
    t0 = time.time()
    n_imgs = 0
    for _ in range(5):
        for b in loader:
            b = b.to(device)
            with torch.no_grad(): _ = global_mean_pool(model(b.x, b.edge_index), b.batch)
            n_imgs += b.num_graphs
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    throughput = n_imgs / elapsed

    # GFLOPs
    from torch.profiler import profile, ProfilerActivity
    b = next(iter(loader)).to(device)
    with profile(activities=[ProfilerActivity.CUDA], with_flops=True) as prof:
        with torch.no_grad(): _ = global_mean_pool(model(b.x, b.edge_index), b.batch)
    total_flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)
    gflops = total_flops / b.num_graphs / 1e9

    res = args.dataset.replace("cifar10","32x32").replace("stl10","96x96").replace("tinyimagenet","64x64")
    print(f"\nTABLE 4 — {args.dataset} ({res})", flush=True)
    print(f"  Tokens (avg): {avg_nodes:.0f} (sparse graph nodes)", flush=True)
    print(f"  Memory:       {peak_mem:.2f} GB", flush=True)
    print(f"  GFLOPs:       {gflops:.2f}", flush=True)
    print(f"  Throughput:   {throughput:.0f} img/s", flush=True)
    print(f"  Params:       {n_params:,}", flush=True)

    import json
    sig = {"tokens": avg_nodes, "memory_gb": peak_mem, "gflops": gflops,
           "img_per_sec": throughput, "params": n_params, "dataset": args.dataset}
    sig_path = f"/home/ud3d4/Desktop/NIPS 26/signals/table4_{args.dataset}.json"
    with open(sig_path, "w") as f:
        json.dump(sig, f, indent=2)
    print(f"Saved: {sig_path}", flush=True)

if __name__ == "__main__":
    main()
