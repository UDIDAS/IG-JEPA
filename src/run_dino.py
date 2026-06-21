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
CACHE_DIR = "/scratch/ud3d4/igjepa_cache"  # Persistent across restarts

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


N_FEAT = 72  # Total feature dims per node (all from raw pixels, NO pretrained models)

def build_graph(img_np, kernel):
    """Build one graph with pixel-derived features only. No pretrained models.
    Features per node (72-dim):
      [0:6]   RGB mean/std (6)
      [6:12]  HSV mean/std (6)
      [12:14] grayscale mean/std (2)
      [14]    log(area)
      [15:17] centroid x,y normalized (2)
      [17:19] bbox width, height normalized (2)
      [19]    boundary_len / area
      [20]    compactness (area / bbox_area)
      [21]    relative area (area / image_area)
      [22:28] RGB skewness + kurtosis (6)
      [28:44] color histogram (16 bins on grayscale)
      [44:52] gradient magnitude mean/std + direction histogram (8)
      [52:68] multi-scale means: 4x4 grid position features (16)
      [68:72] shape: 2nd order moments (4)
    """
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

    # === All features from raw pixels ===
    img_f = img_np.astype(np.float64) / 255.0
    ff = feat.astype(np.float64); area = ff[:,0].clip(min=1)
    flat_map = snode_map.ravel(); valid = flat_map >= 0
    counts = np.bincount(flat_map[valid], minlength=N).astype(np.float64).clip(min=1)

    nf = np.zeros((N, N_FEAT), dtype=np.float32)
    d_idx = 0

    # [0:6] RGB mean/std
    for ch in range(3):
        flat_ch = img_f[:,:,ch].ravel()[valid]
        sums = np.bincount(flat_map[valid], weights=flat_ch, minlength=N)
        means = sums / counts
        sq_sums = np.bincount(flat_map[valid], weights=flat_ch**2, minlength=N)
        stds = np.sqrt((sq_sums / counts - means**2).clip(min=0))
        nf[:, d_idx] = means; nf[:, d_idx+1] = stds; d_idx += 2

    # [6:12] HSV mean/std
    from colorsys import rgb_to_hsv
    hsv = np.zeros_like(img_f)
    for r in range(H):
        for c in range(W):
            hsv[r,c] = rgb_to_hsv(img_f[r,c,0], img_f[r,c,1], img_f[r,c,2])
    for ch in range(3):
        flat_ch = hsv[:,:,ch].ravel()[valid]
        sums = np.bincount(flat_map[valid], weights=flat_ch, minlength=N)
        means = sums / counts
        sq_sums = np.bincount(flat_map[valid], weights=flat_ch**2, minlength=N)
        stds = np.sqrt((sq_sums / counts - means**2).clip(min=0))
        nf[:, d_idx] = means; nf[:, d_idx+1] = stds; d_idx += 2

    # [12:14] Grayscale mean/std
    gray = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
    flat_g = gray.ravel()[valid]
    g_sums = np.bincount(flat_map[valid], weights=flat_g, minlength=N)
    g_means = g_sums / counts
    g_sq = np.bincount(flat_map[valid], weights=flat_g**2, minlength=N)
    g_stds = np.sqrt((g_sq / counts - g_means**2).clip(min=0))
    nf[:, d_idx] = g_means; nf[:, d_idx+1] = g_stds; d_idx += 2

    # [14:22] Geometry
    nf[:, d_idx] = np.log1p(area); d_idx += 1
    nf[:, d_idx] = ff[:,1] / area / W; d_idx += 1  # centroid_x
    nf[:, d_idx] = ff[:,2] / area / H; d_idx += 1  # centroid_y
    nf[:, d_idx] = (ff[:,10]-ff[:,9]) / W; d_idx += 1
    nf[:, d_idx] = (ff[:,12]-ff[:,11]) / H; d_idx += 1
    nf[:, d_idx] = ff[:,13] / area; d_idx += 1
    bbox_area = ((ff[:,10]-ff[:,9]+1)*(ff[:,12]-ff[:,11]+1)).clip(min=1)
    nf[:, d_idx] = (area / bbox_area).clip(max=1); d_idx += 1
    nf[:, d_idx] = area / (H*W); d_idx += 1

    # [22:28] RGB skewness + kurtosis (higher-order color stats)
    for ch in range(3):
        flat_ch = img_f[:,:,ch].ravel()[valid]
        ch_means = nf[:, ch*2]  # already computed
        ch_stds = nf[:, ch*2+1].clip(min=1e-6)
        # Skewness: E[(x-mu)^3] / std^3
        diff3 = (flat_ch - ch_means[flat_map[valid]])**3
        skew_sum = np.bincount(flat_map[valid], weights=diff3, minlength=N)
        nf[:, d_idx] = (skew_sum / counts) / (ch_stds**3 + 1e-8); d_idx += 1
        # Kurtosis: E[(x-mu)^4] / std^4 - 3
        diff4 = (flat_ch - ch_means[flat_map[valid]])**4
        kurt_sum = np.bincount(flat_map[valid], weights=diff4, minlength=N)
        nf[:, d_idx] = (kurt_sum / counts) / (ch_stds**4 + 1e-8) - 3.0; d_idx += 1

    # [28:44] Grayscale histogram (16 bins)
    n_bins = 16
    gray_q = np.clip((flat_g * n_bins).astype(np.int32), 0, n_bins-1)
    for b in range(n_bins):
        mask_b = gray_q == b
        nf[:, d_idx] = np.bincount(flat_map[valid][mask_b], minlength=N).astype(np.float32) / counts
        d_idx += 1

    # [44:52] Gradient features (magnitude mean/std + 6-bin direction histogram)
    gy, gx = np.gradient(gray)
    gmag = np.sqrt(gx**2 + gy**2)
    gdir = np.arctan2(gy, gx)  # [-pi, pi]
    flat_gmag = gmag.ravel()[valid]
    mag_sums = np.bincount(flat_map[valid], weights=flat_gmag, minlength=N)
    mag_means = mag_sums / counts
    mag_sq = np.bincount(flat_map[valid], weights=flat_gmag**2, minlength=N)
    mag_stds = np.sqrt((mag_sq / counts - mag_means**2).clip(min=0))
    nf[:, d_idx] = mag_means; nf[:, d_idx+1] = mag_stds; d_idx += 2
    # Direction histogram (6 bins over [-pi, pi])
    n_dir = 6
    flat_gdir = gdir.ravel()[valid]
    dir_q = np.clip(((flat_gdir + np.pi) / (2*np.pi) * n_dir).astype(np.int32), 0, n_dir-1)
    for b in range(n_dir):
        mask_b = dir_q == b
        nf[:, d_idx] = np.bincount(flat_map[valid][mask_b], minlength=N).astype(np.float32) / counts
        d_idx += 1

    # [52:68] Multi-scale spatial: which 4x4 grid cell does centroid fall in (16-dim one-hot-ish)
    cx = nf[:, 15]; cy = nf[:, 16]  # normalized centroids
    gx4 = np.clip((cx * 4).astype(np.int32), 0, 3)
    gy4 = np.clip((cy * 4).astype(np.int32), 0, 3)
    for r in range(4):
        for c in range(4):
            nf[:, d_idx] = ((gy4 == r) & (gx4 == c)).astype(np.float32); d_idx += 1

    # [68:72] 2nd order moments (shape descriptors)
    rows, cols = np.where(valid.reshape(H, W))  # pixel positions
    flat_valid = flat_map.reshape(H, W)
    pix_r = (rows / H).astype(np.float64)
    pix_c = (cols / W).astype(np.float64)
    node_ids = flat_valid[rows, cols]
    cx_nodes = nf[:, 15].astype(np.float64)
    cy_nodes = nf[:, 16].astype(np.float64)
    dr = pix_r - cy_nodes[node_ids]; dc = pix_c - cx_nodes[node_ids]
    nf[:, d_idx] = np.bincount(node_ids, weights=dr**2, minlength=N).astype(np.float32) / counts; d_idx += 1  # Ixx
    nf[:, d_idx] = np.bincount(node_ids, weights=dc**2, minlength=N).astype(np.float32) / counts; d_idx += 1  # Iyy
    nf[:, d_idx] = np.bincount(node_ids, weights=dr*dc, minlength=N).astype(np.float32) / counts; d_idx += 1  # Ixy
    nf[:, d_idx] = np.sqrt(nf[:, d_idx-3]**2 + nf[:, d_idx-2]**2).clip(min=1e-8); d_idx += 1  # moment magnitude

    nf = np.nan_to_num(nf, nan=0.0, posinf=1.0, neginf=-1.0)
    return Data(x=torch.from_numpy(nf), edge_index=torch.tensor(np.stack([s,d]), dtype=torch.long), num_nodes=N)


# Model — IG-JEPA v2
# Design principles:
#   1. Student sees context-only graph (edges to masked nodes REMOVED — no info leak)
#   2. Predictor uses context NEIGHBOR aggregation for masked nodes (not encoder output
#      at masked positions, which would be a constant since they're disconnected)
#   3. Global residual in encoder preserves input features
#   4. Graph-level BYOL loss forces globally discriminative representations

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
        h0 = self.input_proj(x)
        h = h0
        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index) + h); h = F.gelu(h); h = self.dropout(h)
        return h + h0  # Global residual — preserves projected input features

class IGJEPA(nn.Module):
    def __init__(self, in_dim, hid=384, n_layers=4, n_heads=4, mask_ratio=0.4, ema=0.996):
        super().__init__()
        self.mask_ratio, self.ema_m, self.hid = mask_ratio, ema, hid
        # Student encoder + EMA teacher
        self.enc = GraphTransformerEncoder(in_dim, hid, n_layers, n_heads)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters(): p.requires_grad = False
        # Node-level predictor: takes aggregated context neighbor embeddings → predicts target
        self.pred = nn.Sequential(nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,hid))
        # Graph-level predictor (BYOL — student predicts teacher's graph embedding)
        self.graph_pred = nn.Sequential(nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,hid))

    @torch.no_grad()
    def ema_update(self):
        for a, b in zip(self.enc.parameters(), self.tgt.parameters()):
            b.data.mul_(self.ema_m).add_(a.data, alpha=1-self.ema_m)

    def forward(self, batch):
        x, ei = batch.x, batch.edge_index; N = x.size(0)
        dev = ei.device

        # Per-graph BFS subgraph masking — each graph gets its own connected mask
        masked_flag = torch.zeros(N, dtype=torch.bool, device=dev)
        graph_ids = batch.batch
        unique_graphs = graph_ids.unique()
        ei_np_src, ei_np_dst = ei[0].cpu().numpy(), ei[1].cpu().numpy()
        for gid in unique_graphs:
            node_mask = (graph_ids == gid)
            g_nodes = node_mask.nonzero(as_tuple=True)[0]
            g_N = g_nodes.size(0)
            if g_N < 2: continue
            # Get edges within this graph
            g_start = g_nodes[0].item()
            g_edge_mask = node_mask[ei[0]] & node_mask[ei[1]]
            g_src = ei_np_src[g_edge_mask.cpu().numpy()] - g_start
            g_dst = ei_np_dst[g_edge_mask.cpu().numpy()] - g_start
            nm = max(1, int(g_N * self.mask_ratio))
            mi_local, _, _ = fastloops.subgraph_mask(g_src, g_dst, g_N, nm, random.randint(0, g_N-1))
            masked_flag[g_nodes[mi_local]] = True
        mi = masked_flag.nonzero(as_tuple=True)[0]
        # Build context edge index (remove edges touching masked nodes)
        edge_keep = ~(masked_flag[ei[0]] | masked_flag[ei[1]])
        ctx_ei = ei[:, edge_keep]

        # === Teacher: clean full graph ===
        with torch.no_grad():
            tgt_all = self.tgt(x, ei)
            target = F.layer_norm(tgt_all[mi], [self.hid])
            tgt_graph = global_mean_pool(tgt_all, batch.batch)

        # === Student: context-only graph (edges to masked nodes removed) ===
        xc = x.clone(); xc[mi] = 0  # Zero masked features (they're disconnected anyway)
        # Edge augmentation on context edges only
        ei_aug = ctx_ei[:, torch.rand(ctx_ei.size(1), device=dev) > 0.15]
        c = self.enc(xc, ei_aug)

        # === Loss 1: Node-level JEPA — predict from context neighbors ===
        c2m = masked_flag[ei[1]] & ~masked_flag[ei[0]]  # context→masked edges
        if c2m.any():
            nbr_sum = torch.zeros(N, self.hid, device=dev)
            nbr_cnt = torch.zeros(N, 1, device=dev)
            nbr_sum.index_add_(0, ei[1, c2m], c[ei[0, c2m]])
            nbr_cnt.index_add_(0, ei[1, c2m], torch.ones(c2m.sum(), 1, device=dev))
            nbr_mean = nbr_sum / nbr_cnt.clamp(min=1)
            pred = F.layer_norm(self.pred(nbr_mean[mi]), [self.hid])
            pred_loss = F.smooth_l1_loss(pred, target)
        else:
            pred_loss = torch.tensor(0.0, device=dev, requires_grad=True)

        # === Loss 2: Graph-level BYOL ===
        stu_graph = global_mean_pool(c, batch.batch)
        stu_pred = self.graph_pred(stu_graph)
        byol_loss = 2 - 2 * F.cosine_similarity(stu_pred, tgt_graph, dim=-1).mean()

        # === Loss 3: VICReg on context nodes ===
        ctx_emb = c[~masked_flag]; Nc = ctx_emb.size(0)
        std = torch.sqrt(ctx_emb.var(dim=0)+1e-4)
        vl = F.relu(1.0-std).mean()
        cm = ctx_emb-ctx_emb.mean(0); cov = (cm.T@cm)/max(Nc-1,1)
        od = cov.flatten()[1:].view(self.hid-1,self.hid+1)[:,:-1].flatten()
        cl = (od**2).sum()/self.hid

        total = pred_loss + 1.0*byol_loss + 25*vl + 1*cl
        return total, {"pred":pred_loss.item(), "byol":byol_loss.item(), "std":std.mean().item()}

    def encode_graph(self, batch):
        return global_mean_pool(self.enc(batch.x, batch.edge_index), batch.batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "stl10", "tinyimagenet"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlabeled", action="store_true", help="Use STL-10 100K unlabeled split for pretraining")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    kernel = KERNELS[args.dataset]
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)

    # Use --unlabeled flag to pretrain on STL-10's 100K unlabeled split
    use_unlabeled = args.unlabeled and (args.dataset == "stl10")

    # Cache key — v3 uses raw pixel features only (no DINO)
    cache_tag = f"{args.dataset}_{args.n_train}_{args.n_test}_raw{N_FEAT}" + ("_unlabeled" if use_unlabeled else "")
    ck = hashlib.md5(cache_tag.encode()).hexdigest()[:12]
    cache_dir_shard = os.path.join(CACHE_DIR, f"graphs{N_FEAT}_{ck}")

    if os.path.isdir(cache_dir_shard) and os.path.exists(os.path.join(cache_dir_shard, "meta.pt")):
        print(f"Loading cached sharded graphs: {cache_dir_shard}", flush=True)
        meta = torch.load(os.path.join(cache_dir_shard, "meta.pt"), weights_only=False)
        if use_unlabeled:
            pretrain_graphs = [torch.load(os.path.join(cache_dir_shard, f"pretrain_{i}.pt"), weights_only=False) for i in range(meta["n_pretrain"])]
            train_graphs = [torch.load(os.path.join(cache_dir_shard, f"train_{i}.pt"), weights_only=False) for i in range(meta["n_train"])]
            test_graphs = [torch.load(os.path.join(cache_dir_shard, f"test_{i}.pt"), weights_only=False) for i in range(meta["n_test"])]
            train_labels, test_labels = meta["train_y"], meta["test_y"]
            print(f"  Loaded {len(pretrain_graphs)} pretrain, {len(train_graphs)} train, {len(test_graphs)} test", flush=True)
        else:
            train_graphs = [torch.load(os.path.join(cache_dir_shard, f"train_{i}.pt"), weights_only=False) for i in range(meta["n_train"])]
            test_graphs = [torch.load(os.path.join(cache_dir_shard, f"test_{i}.pt"), weights_only=False) for i in range(meta["n_test"])]
            train_labels, test_labels = meta["train_y"], meta["test_y"]
            pretrain_graphs = train_graphs
            print(f"  Loaded {len(train_graphs)} train, {len(test_graphs)} test", flush=True)
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

        # Load all images as numpy arrays
        t0 = time.time()
        tr_imgs = [np.array(tr_ds[i][0]) for i in range(n_tr)]
        tr_labels = [tr_ds[i][1] for i in range(n_tr)]
        te_imgs = [np.array(te_ds[i][0]) for i in range(n_te)]
        te_labels = [te_ds[i][1] for i in range(n_te)]

        # For STL-10: load 100K unlabeled split for pretraining (standard SSL protocol)
        if use_unlabeled:
            un_ds = datasets.STL10(data_dir, split='unlabeled', download=True)
            n_un = len(un_ds)
            un_imgs = [np.array(un_ds[i][0]) for i in range(n_un)]
            un_labels = [-1] * n_un  # no labels
            print(f"  Pretrain (unlabeled): {n_un}, Train (labeled): {n_tr}, Test: {n_te}", flush=True)
        else:
            print(f"  Train: {n_tr}, Test: {n_te}", flush=True)
        print(f"  Images loaded in {time.time()-t0:.1f}s", flush=True)

        # Step 2: Build graphs from raw pixels (NO pretrained models)
        print(f"Step 2: Building {N_FEAT}-dim graphs from raw pixels...", flush=True)
        t0 = time.time()

        def build_split(images, labels_list):
            graphs = []
            def proc(i):
                g = build_graph(images[i], kernel)
                return i, g
            with ThreadPoolExecutor(max_workers=16) as ex:
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

        if use_unlabeled:
            pretrain_graphs = build_split(un_imgs, un_labels)
            del un_imgs
            print(f"  Pretrain graphs: {len(pretrain_graphs)}", flush=True)
        train_graphs = build_split(tr_imgs, tr_labels)
        test_graphs = build_split(te_imgs, te_labels)
        train_labels = [g.y.item() for g in train_graphs]
        test_labels = [g.y.item() for g in test_graphs]
        if not use_unlabeled:
            pretrain_graphs = train_graphs
        del tr_imgs, te_imgs
        print(f"  Graphs built in {time.time()-t0:.1f}s: {len(pretrain_graphs)} pretrain, {len(train_graphs)} train, {len(test_graphs)} test", flush=True)
        print(f"  Feature dim: {train_graphs[0].x.shape[1]}, Nodes avg: {np.mean([g.num_nodes for g in pretrain_graphs]):.0f}", flush=True)

        # Cache (sharded .pt format — fast to load)
        os.makedirs(cache_dir_shard, exist_ok=True)
        if use_unlabeled:
            for i, g in enumerate(pretrain_graphs): torch.save(g, os.path.join(cache_dir_shard, f"pretrain_{i}.pt"))
        for i, g in enumerate(train_graphs): torch.save(g, os.path.join(cache_dir_shard, f"train_{i}.pt"))
        for i, g in enumerate(test_graphs): torch.save(g, os.path.join(cache_dir_shard, f"test_{i}.pt"))
        meta = {"n_train": len(train_graphs), "n_test": len(test_graphs),
                "train_y": train_labels, "test_y": test_labels}
        if use_unlabeled:
            meta["n_pretrain"] = len(pretrain_graphs)
        torch.save(meta, os.path.join(cache_dir_shard, "meta.pt"))
        cache_size = sum(os.path.getsize(os.path.join(cache_dir_shard, f)) for f in os.listdir(cache_dir_shard))
        print(f"  Cached: {cache_dir_shard} ({cache_size/1e6:.0f} MB)", flush=True)

    # Step 4: Train IG-JEPA on pretrain split (unlabeled for STL-10, train for others)
    print(f"\nStep 4: IG-JEPA pretraining on {len(pretrain_graphs)} graphs ({args.epochs} epochs)...", flush=True)
    in_dim = pretrain_graphs[0].x.shape[1]
    pretrain_loader = DataLoader(pretrain_graphs, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_graphs, batch_size=args.bs, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=args.bs, num_workers=0)

    model = IGJEPA(in_dim, args.hid, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    for ep in range(1, args.epochs+1):
        model.train(); tl, n = 0, 0
        for b in pretrain_loader:
            b = b.to(device); opt.zero_grad()
            loss, info = model(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); model.ema_update()
            tl += loss.item(); n += 1
        sch.step()
        avg_loss = tl / n
        if ep % 10 == 0 or ep == 1:
            byol_str = f" | byol: {info.get('byol',0):.4f}" if 'byol' in info else ""
            print(f"  Ep {ep:3d} | Loss: {avg_loss:.4f} | std: {info['std']:.4f}{byol_str}", flush=True)

    print(f"  Training complete. Using final model from epoch {args.epochs}.", flush=True)

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

    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
    clf_j = LogisticRegression(max_iter=2000); clf_j.fit(Xj_tr, y_tr)
    clf_r = LogisticRegression(max_iter=2000); clf_r.fit(Xr_tr, y_tr)
    pred_j, pred_r = clf_j.predict(Xj_te), clf_r.predict(Xr_te)
    acc_j = accuracy_score(y_te, pred_j)
    acc_r = accuracy_score(y_te, pred_r)

    # MLP probe (proper training with scheduler)
    n_cls = len(np.unique(y_tr))
    mlp = nn.Sequential(nn.Linear(args.hid, args.hid), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(args.hid, args.hid), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(args.hid, n_cls)).to(device)
    mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    mlp_sch = CosineAnnealingLR(mlp_opt, T_max=200, eta_min=1e-5)
    Xt = torch.tensor(Xj_tr, dtype=torch.float).to(device)
    yt = torch.tensor(y_tr, dtype=torch.long).to(device)
    mlp_bs = 256
    mlp.train()
    for ep_mlp in range(200):
        perm = torch.randperm(Xt.size(0), device=device)
        for i in range(0, Xt.size(0), mlp_bs):
            idx = perm[i:i+mlp_bs]
            mlp_opt.zero_grad(); F.cross_entropy(mlp(Xt[idx]), yt[idx]).backward(); mlp_opt.step()
        mlp_sch.step()
    mlp.eval()
    with torch.no_grad(): acc_mlp = accuracy_score(y_te, mlp(torch.tensor(Xj_te, dtype=torch.float).to(device)).argmax(1).cpu().numpy())

    print(f"\nRESULTS — {args.dataset}", flush=True)
    print(f"  Raw ({N_FEAT}-dim) + LogReg:  {acc_r*100:.2f}%", flush=True)
    print(f"  JEPA + LogReg:           {acc_j*100:.2f}%", flush=True)
    print(f"  JEPA + MLP:              {acc_mlp*100:.2f}%", flush=True)

    # Detailed metrics
    avg = 'weighted' if len(np.unique(y_te)) > 2 else 'binary'
    f1_j = f1_score(y_te, pred_j, average=avg)
    f1_r = f1_score(y_te, pred_r, average=avg)
    prec_j = precision_score(y_te, pred_j, average=avg)
    prec_r = precision_score(y_te, pred_r, average=avg)
    rec_j = recall_score(y_te, pred_j, average=avg)
    rec_r = recall_score(y_te, pred_r, average=avg)
    print(f"\nDETAILED METRICS — {args.dataset}", flush=True)
    print(f"  {'Metric':>12} | {'Raw+LR':>8} | {'JEPA+LR':>8}", flush=True)
    print(f"  {'Accuracy':>12} | {acc_r*100:>7.2f}% | {acc_j*100:>7.2f}%", flush=True)
    print(f"  {'F1 (wtd)':>12} | {f1_r*100:>7.2f}% | {f1_j*100:>7.2f}%", flush=True)
    print(f"  {'Precision':>12} | {prec_r*100:>7.2f}% | {prec_j*100:>7.2f}%", flush=True)
    print(f"  {'Recall':>12} | {rec_r*100:>7.2f}% | {rec_j*100:>7.2f}%", flush=True)

    # Per-class report
    print(f"\nPER-CLASS REPORT (JEPA + LogReg) — {args.dataset}", flush=True)
    print(classification_report(y_te, pred_j, digits=4), flush=True)

    # Confusion matrix
    cm = confusion_matrix(y_te, pred_j)
    print(f"CONFUSION MATRIX (JEPA + LogReg) — {args.dataset}", flush=True)
    print(cm, flush=True)

    # Label efficiency evaluation
    print(f"\nLABEL EFFICIENCY — {args.dataset}", flush=True)
    print(f"  {'Frac':>6} | {'N_train':>7} | {'Raw+LR':>8} | {'JEPA+LR':>8} | {'Gap':>7}", flush=True)
    print(f"  {'-'*6} | {'-'*7} | {'-'*8} | {'-'*8} | {'-'*7}", flush=True)
    label_fracs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    label_eff = {}
    from sklearn.model_selection import StratifiedShuffleSplit
    for frac in label_fracs:
        n_sub = max(10, int(len(y_tr) * frac))
        if n_sub >= len(y_tr):
            sub_idx = np.arange(len(y_tr))
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sub, random_state=42)
            sub_idx = list(sss.split(Xj_tr, y_tr))[0][0]
        clf_j_sub = LogisticRegression(max_iter=2000); clf_j_sub.fit(Xj_tr[sub_idx], y_tr[sub_idx])
        clf_r_sub = LogisticRegression(max_iter=2000); clf_r_sub.fit(Xr_tr[sub_idx], y_tr[sub_idx])
        acc_j_sub = accuracy_score(y_te, clf_j_sub.predict(Xj_te))
        acc_r_sub = accuracy_score(y_te, clf_r_sub.predict(Xr_te))
        gap = acc_j_sub - acc_r_sub
        label_eff[frac] = {"raw": acc_r_sub, "jepa": acc_j_sub, "n": n_sub}
        print(f"  {frac:>5.0%} | {n_sub:>7} | {acc_r_sub*100:>7.2f}% | {acc_j_sub*100:>7.2f}% | {gap*100:>+6.2f}%", flush=True)

    # Signal
    import json
    sig = {"acc_raw": acc_r, "acc_jepa_lr": acc_j, "acc_jepa_mlp": acc_mlp,
           "f1_raw": f1_r, "f1_jepa": f1_j,
           "precision_raw": prec_r, "precision_jepa": prec_j,
           "recall_raw": rec_r, "recall_jepa": rec_j,
           "confusion_matrix": cm.tolist(),
           "label_efficiency": label_eff,
           "dataset": args.dataset, "params": sum(p.numel() for p in model.parameters())}
    with open(f"/home/ud3d4/Desktop/Projects/NIPS 26/signals/done_{args.dataset}_dino.json", "w") as f:
        json.dump(sig, f, indent=2)
    print(f"Signal: done_{args.dataset}_dino.json", flush=True)


if __name__ == "__main__":
    main()
