"""
IG-JEPA v3: CNN backbone + Graph JEPA (end-to-end)

Same ResNet-18 backbone as SimCLR/BYOL benchmarks, but instead of
global average pool, we route through superpixel graph + JEPA:

Benchmark:  Image → ResNet-18 → global pool → 512d → linear probe
Ours:       Image → ResNet-18 → feature map → pool per superpixel → graph → JEPA → probe

Same backbone, same params — only difference is graph structure on top.
If this beats vanilla ResNet-18 SSL, the graph structure adds value.

Cached: superpixel graph structure (edges + pixel-to-node maps).
On-the-fly: CNN features extracted each forward pass (end-to-end trainable).
"""

import os, sys, copy, argparse, random, time, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms
import fastloops

EDGE_MERGED = 0b0001_0000
CACHE_DIR = "/scratch/ud3d4/igjepa_cache"

KERNELS = {
    "cifar10": {"merge_distance": 5, "cut_distance": 80,
                "delete_small_node_max_size": 0, "delete_large_node_min_size": 1024},
    "stl10": {"merge_distance": 8, "cut_distance": 80,
              "delete_small_node_max_size": 2, "delete_large_node_min_size": 9216},
    "tinyimagenet": {"merge_distance": 6, "cut_distance": 80,
                     "delete_small_node_max_size": 1, "delete_large_node_min_size": 4096},
}

IMG_SIZES = {"cifar10": 32, "stl10": 96, "tinyimagenet": 64}


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


def build_graph_structure(img_np, kernel):
    """Build graph structure only: edges + pixel-to-node mapping. No features."""
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
    snode_map = cc2s[labels]  # (H, W) → node index per pixel

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

    return {
        "edge_index": np.stack([s, d]).astype(np.int64),
        "snode_map": snode_map.astype(np.int32),  # (H, W) pixel→node
        "num_nodes": int(N),
    }


class ImageGraphDataset(Dataset):
    """Dataset that returns raw images + precomputed graph structures."""
    def __init__(self, images, labels, graph_structs, transform=None):
        self.images = images       # list of numpy (H,W,3)
        self.labels = labels       # list of int
        self.structs = graph_structs  # list of dicts (edge_index, snode_map, num_nodes)
        self.transform = transform
        # Filter out None structs
        valid = [(i, img, lbl, gs) for i, (img, lbl, gs) in
                 enumerate(zip(images, labels, graph_structs)) if gs is not None]
        self.indices, self.images, self.labels, self.structs = zip(*valid) if valid else ([], [], [], [])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 3) uint8
        if self.transform:
            img_t = self.transform(img)  # (3, H, W) float tensor
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, self.labels[idx], idx


def collate_graph_batch(batch_data, structs, cnn_feat_maps, feat_dim):
    """Build a PyG Batch from CNN feature maps + precomputed graph structures.

    Args:
        batch_data: list of (img_tensor, label, dataset_idx)
        structs: full dataset's graph structures
        cnn_feat_maps: (B, C, H', W') CNN output feature maps
        feat_dim: CNN output channels
    Returns:
        PyG Batch with CNN-pooled node features
    """
    graphs = []
    for i, (_, label, ds_idx) in enumerate(batch_data):
        gs = structs[ds_idx]
        smap = gs["snode_map"]  # (H, W)
        N = gs["num_nodes"]
        ei = torch.from_numpy(gs["edge_index"]).long()

        # Pool CNN features per superpixel
        feat_map = cnn_feat_maps[i]  # (C, H', W')
        C, fH, fW = feat_map.shape
        H, W = smap.shape

        # Resize snode_map to match feature map spatial dims
        dev = feat_map.device
        smap_resized = torch.from_numpy(smap).unsqueeze(0).unsqueeze(0).float()
        smap_resized = F.interpolate(smap_resized, size=(fH, fW), mode='nearest')[0, 0].long().to(dev)

        # Scatter mean: pool CNN features per node (float32 for index_add_ compatibility)
        flat_map = smap_resized.reshape(-1)  # (fH*fW,)
        flat_feat = feat_map.float().reshape(C, -1).T  # (fH*fW, C) in float32
        valid = flat_map >= 0
        node_feat = torch.zeros(N, C, device=dev)
        node_cnt = torch.zeros(N, 1, device=dev)
        if valid.any():
            node_feat.index_add_(0, flat_map[valid], flat_feat[valid])
            node_cnt.index_add_(0, flat_map[valid], torch.ones(valid.sum(), 1, device=dev))
        node_feat = node_feat / node_cnt.clamp(min=1)

        g = Data(x=node_feat, edge_index=ei.to(feat_map.device),
                 num_nodes=N, y=torch.tensor([label], dtype=torch.long))
        graphs.append(g)

    return Batch.from_data_list(graphs)


# === Model ===

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
        return h + h0


class IGJEPA(nn.Module):
    def __init__(self, cnn_dim, hid=512, n_layers=4, n_heads=4, mask_ratio=0.4, ema=0.996):
        super().__init__()
        self.mask_ratio, self.ema_m, self.hid = mask_ratio, ema, hid
        self.enc = GraphTransformerEncoder(cnn_dim, hid, n_layers, n_heads)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters(): p.requires_grad = False
        self.pred = nn.Sequential(nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,hid))
        self.graph_pred = nn.Sequential(nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,hid))

    @torch.no_grad()
    def ema_update(self):
        for a, b in zip(self.enc.parameters(), self.tgt.parameters()):
            b.data.mul_(self.ema_m).add_(a.data, alpha=1-self.ema_m)

    def forward(self, batch):
        x, ei = batch.x, batch.edge_index; N = x.size(0)
        dev = ei.device

        # Per-graph BFS subgraph masking
        masked_flag = torch.zeros(N, dtype=torch.bool, device=dev)
        graph_ids = batch.batch
        unique_graphs = graph_ids.unique()
        ei_np_src, ei_np_dst = ei[0].cpu().numpy(), ei[1].cpu().numpy()
        for gid in unique_graphs:
            node_mask = (graph_ids == gid)
            g_nodes = node_mask.nonzero(as_tuple=True)[0]
            g_N = g_nodes.size(0)
            if g_N < 2: continue
            g_start = g_nodes[0].item()
            g_edge_mask = node_mask[ei[0]] & node_mask[ei[1]]
            g_src = ei_np_src[g_edge_mask.cpu().numpy()] - g_start
            g_dst = ei_np_dst[g_edge_mask.cpu().numpy()] - g_start
            nm = max(1, int(g_N * self.mask_ratio))
            mi_local, _, _ = fastloops.subgraph_mask(g_src, g_dst, g_N, nm, random.randint(0, g_N-1))
            masked_flag[g_nodes[mi_local]] = True
        mi = masked_flag.nonzero(as_tuple=True)[0]
        edge_keep = ~(masked_flag[ei[0]] | masked_flag[ei[1]])
        ctx_ei = ei[:, edge_keep]

        # Teacher: clean full graph
        with torch.no_grad():
            tgt_all = self.tgt(x, ei)
            target = F.layer_norm(tgt_all[mi], [self.hid])
            tgt_graph = global_mean_pool(tgt_all, batch.batch)

        # Student: context-only
        xc = x.clone(); xc[mi] = 0
        ei_aug = ctx_ei[:, torch.rand(ctx_ei.size(1), device=dev) > 0.15]
        c = self.enc(xc, ei_aug)

        # Loss 1: Node-level JEPA
        c2m = masked_flag[ei[1]] & ~masked_flag[ei[0]]
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

        # Loss 2: Graph-level BYOL
        stu_graph = global_mean_pool(c, batch.batch)
        stu_pred = self.graph_pred(stu_graph)
        byol_loss = 2 - 2 * F.cosine_similarity(stu_pred, tgt_graph, dim=-1).mean()

        # Loss 3: VICReg (float32 to avoid AMP overflow)
        ctx_emb = c[~masked_flag].float(); Nc = ctx_emb.size(0)
        std = torch.sqrt(ctx_emb.var(dim=0)+1e-4)
        vl = F.relu(1.0-std).mean()
        cm = ctx_emb-ctx_emb.mean(0); cov = (cm.T@cm)/max(Nc-1,1)
        od = cov.flatten()[1:].view(self.hid-1,self.hid+1)[:,:-1].flatten()
        cl = (od**2).sum()/self.hid

        total = pred_loss + 1.0*byol_loss + 25*vl + 1*cl
        return total, {"pred":pred_loss.item(), "byol":byol_loss.item(), "std":std.mean().item()}

    def encode_graph(self, batch):
        return global_mean_pool(self.enc(batch.x, batch.edge_index), batch.batch)


class FullModel(nn.Module):
    """CNN backbone + Graph JEPA, end-to-end."""
    def __init__(self, cnn_dim=512, hid=512, n_layers=4, n_heads=4):
        super().__init__()
        # ResNet-18 backbone (same as SimCLR/BYOL benchmarks) minus final FC
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool + fc
        self.cnn_dim = cnn_dim  # ResNet-18 outputs 512 channels
        # Graph JEPA
        self.jepa = IGJEPA(cnn_dim, hid, n_layers, n_heads)

    def extract_features(self, images):
        """Run CNN backbone on images → spatial feature maps (AMP-safe)."""
        with torch.amp.autocast('cuda'):
            feat = self.cnn(images)
        return feat.float()  # Return float32 for graph operations

    def forward(self, batch_data, structs):
        """End-to-end forward: images → CNN → pool per superpixel → graph JEPA."""
        imgs = torch.stack([d[0] for d in batch_data])  # (B, 3, H, W)
        device = next(self.parameters()).device
        imgs = imgs.to(device)

        # CNN feature extraction
        feat_maps = self.extract_features(imgs)  # (B, 512, H', W')

        # Build graph batch with CNN features pooled per superpixel
        graph_batch = collate_graph_batch(batch_data, structs, feat_maps, self.cnn_dim)

        # Graph JEPA forward
        return self.jepa(graph_batch)

    @torch.no_grad()
    def encode(self, batch_data, structs):
        """Extract graph-level embeddings for evaluation."""
        imgs = torch.stack([d[0] for d in batch_data])
        device = next(self.parameters()).device
        imgs = imgs.to(device)
        feat_maps = self.extract_features(imgs)
        graph_batch = collate_graph_batch(batch_data, structs, feat_maps, self.cnn_dim)
        return self.jepa.encode_graph(graph_batch)

    def ema_update(self):
        self.jepa.ema_update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "stl10", "tinyimagenet"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hid", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlabeled", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    kernel = KERNELS[args.dataset]
    img_size = IMG_SIZES[args.dataset]
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset} ({img_size}x{img_size})", flush=True)

    # === Step 1: Load images ===
    print("Step 1: Loading images...", flush=True)
    from torchvision import datasets
    data_dir = f"/home/ud3d4/datasets/{args.dataset}"
    use_unlabeled = args.unlabeled and (args.dataset == "stl10")

    if args.dataset == "cifar10":
        tr_ds = datasets.CIFAR10(data_dir, train=True, download=True)
        te_ds = datasets.CIFAR10(data_dir, train=False, download=True)
    elif args.dataset == "stl10":
        tr_ds = datasets.STL10(data_dir, split='train', download=True)
        te_ds = datasets.STL10(data_dir, split='test', download=True)
    elif args.dataset == "tinyimagenet":
        from datasets import load_dataset as hf_load
        hf_tr = hf_load('Maysee/tiny-imagenet', split='train', cache_dir='/home/ud3d4/datasets/tinyimagenet')
        hf_te = hf_load('Maysee/tiny-imagenet', split='valid', cache_dir='/home/ud3d4/datasets/tinyimagenet')
        class HFW:
            def __init__(self, d): self.d = d
            def __len__(self): return len(self.d)
            def __getitem__(self, i): return self.d[i]['image'], self.d[i]['label']
        tr_ds, te_ds = HFW(hf_tr), HFW(hf_te)

    t0 = time.time()
    tr_imgs = [np.array(tr_ds[i][0]) for i in range(len(tr_ds))]
    tr_labels = [tr_ds[i][1] for i in range(len(tr_ds))]
    te_imgs = [np.array(te_ds[i][0]) for i in range(len(te_ds))]
    te_labels = [te_ds[i][1] for i in range(len(te_ds))]

    if use_unlabeled:
        un_ds = datasets.STL10(data_dir, split='unlabeled', download=True)
        un_imgs = [np.array(un_ds[i][0]) for i in range(len(un_ds))]
        un_labels = [-1] * len(un_imgs)
        print(f"  Pretrain: {len(un_imgs)}, Train: {len(tr_imgs)}, Test: {len(te_imgs)}", flush=True)
    else:
        print(f"  Train: {len(tr_imgs)}, Test: {len(te_imgs)}", flush=True)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # === Step 2: Build/cache graph structures ===
    cache_tag = f"{args.dataset}_structs_v3" + ("_unlabeled" if use_unlabeled else "")
    ck = hashlib.md5(cache_tag.encode()).hexdigest()[:12]
    cache_path = os.path.join(CACHE_DIR, f"structs_{ck}.pt")

    if os.path.exists(cache_path):
        print(f"Step 2: Loading cached graph structures: {cache_path}", flush=True)
        cached = torch.load(cache_path, weights_only=False)
        tr_structs = cached["train"]
        te_structs = cached["test"]
        if use_unlabeled:
            un_structs = cached["pretrain"]
        print(f"  Loaded {len(tr_structs)} train, {len(te_structs)} test" +
              (f", {len(un_structs)} pretrain" if use_unlabeled else ""), flush=True)
    else:
        print("Step 2: Building graph structures...", flush=True)
        t0 = time.time()

        def build_structs(images):
            structs = [None] * len(images)
            def proc(i):
                return i, build_graph_structure(images[i], kernel)
            with ThreadPoolExecutor(max_workers=16) as ex:
                futures = {ex.submit(proc, i): i for i in range(len(images))}
                done = 0
                for f in as_completed(futures):
                    idx, gs = f.result()
                    structs[idx] = gs
                    done += 1
                    if done % 10000 == 0:
                        print(f"    {done}/{len(images)}", flush=True)
            return structs

        if use_unlabeled:
            un_structs = build_structs(un_imgs)
            print(f"  Pretrain structs: {sum(1 for s in un_structs if s is not None)}/{len(un_structs)}", flush=True)
        tr_structs = build_structs(tr_imgs)
        te_structs = build_structs(te_imgs)
        print(f"  Built in {time.time()-t0:.1f}s", flush=True)

        os.makedirs(CACHE_DIR, exist_ok=True)
        save_data = {"train": tr_structs, "test": te_structs}
        if use_unlabeled: save_data["pretrain"] = un_structs
        torch.save(save_data, cache_path)
        print(f"  Cached: {cache_path}", flush=True)

    # Set up pretrain data
    if use_unlabeled:
        pt_imgs, pt_labels, pt_structs = un_imgs, un_labels, un_structs
    else:
        pt_imgs, pt_labels, pt_structs = tr_imgs, tr_labels, tr_structs

    # Image transform (standard SSL augmentation)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ])

    pt_dataset = ImageGraphDataset(pt_imgs, pt_labels, pt_structs, transform=train_transform)
    tr_dataset = ImageGraphDataset(tr_imgs, tr_labels, tr_structs, transform=eval_transform)
    te_dataset = ImageGraphDataset(te_imgs, te_labels, te_structs, transform=eval_transform)

    n_workers = min(16, os.cpu_count() or 4)
    pt_loader = TorchDataLoader(pt_dataset, batch_size=args.bs, shuffle=True,
                                num_workers=n_workers, pin_memory=True, collate_fn=lambda x: x,
                                persistent_workers=True, prefetch_factor=4)
    tr_loader = TorchDataLoader(tr_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=n_workers // 2, pin_memory=True, collate_fn=lambda x: x)
    te_loader = TorchDataLoader(te_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=n_workers // 2, pin_memory=True, collate_fn=lambda x: x)

    print(f"  Pretrain: {len(pt_dataset)}, Train: {len(tr_dataset)}, Test: {len(te_dataset)}", flush=True)

    # === Validation: Graph structure sanity checks ===
    print("\n--- Validation: Graph structures ---", flush=True)
    valid_count = sum(1 for s in pt_dataset.structs if s is not None)
    print(f"  Valid graphs: {valid_count}/{len(pt_dataset)}", flush=True)
    sample_gs = pt_dataset.structs[0]
    assert sample_gs is not None, "First graph structure is None"
    assert sample_gs["num_nodes"] >= 2, f"Too few nodes: {sample_gs['num_nodes']}"
    assert sample_gs["edge_index"].shape[0] == 2, f"Bad edge_index shape: {sample_gs['edge_index'].shape}"
    assert sample_gs["edge_index"].max() < sample_gs["num_nodes"], "Edge index exceeds num_nodes"
    assert (sample_gs["snode_map"] >= -1).all(), "snode_map has invalid values"
    node_counts = [s["num_nodes"] for s in pt_dataset.structs if s is not None]
    print(f"  Nodes: min={min(node_counts)}, avg={np.mean(node_counts):.0f}, max={max(node_counts)}", flush=True)
    print(f"  Graph structure validation: PASSED", flush=True)

    # === Validation: CNN + superpixel pooling sanity check ===
    print("\n--- Validation: CNN + superpixel pooling ---", flush=True)
    model_test = FullModel(cnn_dim=512, hid=args.hid, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    model_test.eval()
    with torch.no_grad():
        test_batch = [pt_dataset[i] for i in range(min(4, len(pt_dataset)))]
        test_imgs = torch.stack([d[0] for d in test_batch]).to(device)
        test_feats = model_test.extract_features(test_imgs)
        print(f"  CNN output: {test_feats.shape}, dtype={test_feats.dtype}", flush=True)
        assert not test_feats.isnan().any(), "CNN output contains NaN"
        assert not test_feats.isinf().any(), "CNN output contains Inf"
        test_graph = collate_graph_batch(test_batch, pt_dataset.structs, test_feats, 512)
        print(f"  Graph batch: {test_graph.num_nodes} nodes, {test_graph.edge_index.shape[1]} edges, features={test_graph.x.shape}", flush=True)
        assert not test_graph.x.isnan().any(), "Pooled node features contain NaN"
        assert not test_graph.x.isinf().any(), "Pooled node features contain Inf"
        assert test_graph.x.shape[1] == 512, f"Expected 512-dim features, got {test_graph.x.shape[1]}"
        # Check feature variance (not collapsed)
        feat_std = test_graph.x.std(dim=0).mean().item()
        print(f"  Pooled feature std: {feat_std:.4f} (should be > 0.01)", flush=True)
        assert feat_std > 0.001, f"Features appear collapsed: std={feat_std}"
        # Test full forward pass
        loss, info = model_test.jepa(test_graph)
        print(f"  JEPA forward: loss={loss.item():.4f}, pred={info['pred']:.4f}, byol={info['byol']:.4f}, std={info['std']:.4f}", flush=True)
        assert not np.isnan(loss.item()), "JEPA loss is NaN"
        assert not np.isinf(loss.item()), "JEPA loss is Inf"
        # Test encode
        emb = model_test.jepa.encode_graph(test_graph)
        print(f"  Encode: {emb.shape}, std={emb.std(dim=0).mean().item():.4f}", flush=True)
        assert emb.shape == (len(test_batch), args.hid), f"Bad embedding shape: {emb.shape}"
    # Test backward (separate scope to not pollute training model)
    model_test.train()
    test_batch2 = [pt_dataset[i] for i in range(min(4, len(pt_dataset)))]
    loss2, _ = model_test(test_batch2, pt_dataset.structs)
    loss2.backward()
    grad_norm = sum(p.grad.norm().item() for p in model_test.parameters() if p.grad is not None)
    print(f"  Backward: grad_norm={grad_norm:.4f}", flush=True)
    assert grad_norm > 0, "Gradients are all zero"
    assert not np.isnan(grad_norm), "Gradients contain NaN"
    del model_test, loss2; torch.cuda.empty_cache()
    print(f"  CNN + pooling + JEPA validation: PASSED", flush=True)

    # === Step 3: Train ===
    print(f"\nStep 3: Training CNN + Graph JEPA ({args.epochs} epochs)...", flush=True)
    model = FullModel(cnn_dim=512, hid=args.hid, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    for ep in range(1, args.epochs+1):
        model.train(); tl, n = 0, 0
        t_ep = time.time()
        for batch_data in pt_loader:
            opt.zero_grad()
            try:
                loss, info = model(batch_data, pt_dataset.structs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model.ema_update()
                tl += loss.item(); n += 1
            except Exception as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise
        sch.step()
        avg = tl / max(n, 1)
        # Runtime validation: catch NaN/Inf early
        if np.isnan(avg) or np.isinf(avg):
            print(f"  ERROR: Loss is {avg} at epoch {ep}. Aborting.", flush=True)
            sys.exit(1)
        if ep % 10 == 0 or ep == 1:
            eps = time.time() - t_ep
            print(f"  Ep {ep:3d} | Loss: {avg:.4f} | std: {info.get('std',0):.4f} | byol: {info.get('byol',0):.4f} | {eps:.0f}s", flush=True)
        # Periodic embedding health check (every 50 epochs)
        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                check_batch = [pt_dataset[i] for i in range(min(8, len(pt_dataset)))]
                check_emb = model.encode(check_batch, pt_dataset.structs)
                emb_std = check_emb.std(dim=0).mean().item()
                emb_nan = check_emb.isnan().any().item()
                print(f"    Health: emb_std={emb_std:.4f}, nan={emb_nan}", flush=True)
                if emb_nan or emb_std < 0.001:
                    print(f"    WARNING: Embeddings unhealthy (collapsed or NaN)", flush=True)
            model.train()

    print(f"  Training complete.", flush=True)

    # === Step 4: Evaluate ===
    print("\nStep 4: Evaluation...", flush=True)
    model.eval()

    @torch.no_grad()
    def extract_embeddings(loader, dataset):
        feats, labels = [], []
        for batch_data in loader:
            emb = model.encode(batch_data, dataset.structs)
            feats.append(emb.cpu())
            labels.extend([d[1] for d in batch_data])
        return torch.cat(feats).numpy(), np.array(labels)

    Xj_tr, y_tr = extract_embeddings(tr_loader, tr_dataset)
    Xj_te, y_te = extract_embeddings(te_loader, te_dataset)
    print(f"  Embeddings: train={Xj_tr.shape}, test={Xj_te.shape}", flush=True)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xj_tr, y_tr)
    pred_j = clf.predict(Xj_te)
    acc_j = accuracy_score(y_te, pred_j)

    # MLP probe
    n_cls = len(np.unique(y_tr))
    mlp = nn.Sequential(nn.Linear(args.hid, args.hid), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(args.hid, args.hid), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(args.hid, n_cls)).to(device)
    mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    mlp_sch = CosineAnnealingLR(mlp_opt, T_max=200, eta_min=1e-5)
    Xt = torch.tensor(Xj_tr, dtype=torch.float).to(device)
    yt = torch.tensor(y_tr, dtype=torch.long).to(device)
    mlp.train()
    for ep_mlp in range(200):
        perm = torch.randperm(Xt.size(0), device=device)
        for i in range(0, Xt.size(0), 256):
            idx = perm[i:i+256]
            mlp_opt.zero_grad(); F.cross_entropy(mlp(Xt[idx]), yt[idx]).backward(); mlp_opt.step()
        mlp_sch.step()
    mlp.eval()
    with torch.no_grad():
        acc_mlp = accuracy_score(y_te, mlp(torch.tensor(Xj_te, dtype=torch.float).to(device)).argmax(1).cpu().numpy())

    avg_m = 'weighted'
    f1_j = f1_score(y_te, pred_j, average=avg_m)
    prec_j = precision_score(y_te, pred_j, average=avg_m)
    rec_j = recall_score(y_te, pred_j, average=avg_m)

    print(f"\nRESULTS — {args.dataset}", flush=True)
    print(f"  JEPA + LogReg:           {acc_j*100:.2f}%", flush=True)
    print(f"  JEPA + MLP:              {acc_mlp*100:.2f}%", flush=True)
    print(f"  F1 (wtd): {f1_j*100:.2f}%  Precision: {prec_j*100:.2f}%  Recall: {rec_j*100:.2f}%", flush=True)
    print(f"  Params: {n_params:,}", flush=True)

    # Label efficiency
    print(f"\nLABEL EFFICIENCY — {args.dataset}", flush=True)
    from sklearn.model_selection import StratifiedShuffleSplit
    label_fracs = [0.01, 0.05, 0.10, 0.50, 1.00]
    for frac in label_fracs:
        n_sub = max(n_cls * 2, int(len(y_tr) * frac))
        if n_sub >= len(y_tr):
            sub_idx = np.arange(len(y_tr))
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sub, random_state=42)
            sub_idx = list(sss.split(Xj_tr, y_tr))[0][0]
        clf_sub = LogisticRegression(max_iter=2000); clf_sub.fit(Xj_tr[sub_idx], y_tr[sub_idx])
        acc_sub = accuracy_score(y_te, clf_sub.predict(Xj_te))
        print(f"  {frac:>5.0%} | {n_sub:>7} | {acc_sub*100:>7.2f}%", flush=True)

    # Signal
    import json
    sig = {"acc_jepa_lr": acc_j, "acc_jepa_mlp": acc_mlp,
           "f1_jepa": f1_j, "precision_jepa": prec_j, "recall_jepa": rec_j,
           "dataset": args.dataset, "params": n_params, "backbone": "resnet18"}
    with open(f"/home/ud3d4/Desktop/Projects/NIPS 26/signals/done_{args.dataset}_cnn.json", "w") as f:
        json.dump(sig, f, indent=2)
    print(f"Signal: done_{args.dataset}_cnn.json", flush=True)


if __name__ == "__main__":
    main()
