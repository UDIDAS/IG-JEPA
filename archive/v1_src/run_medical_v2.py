"""
IG-JEPA Medical Segmentation v2 — aggressive improvements to match supervised SOTA.

Changes from v1:
  - Finer supernodes (merge_distance=3, more nodes per slice)
  - Liver-specific CT windowing (WL=60, WW=200)
  - Multi-slice context (z-1, z, z+1 features per node)
  - MLP probe (3 layers) instead of linear
  - Dice loss for node classifier
  - More epochs (100 pretrain, 100 classify)
  - Post-processing: largest connected component

Target: LiTS Liver 99.4%, Pancreas 94.0%
"""

import os, sys, copy, argparse, random, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import label as scipy_label
import fastloops

EDGE_MERGED = 0b0001_0000

# Finer kernel for medical CT — merge=3 works (129s for 3000 slices)
MEDICAL_KERNEL = {
    "merge_distance": 3,
    "cut_distance": 40,
    "delete_small_node_max_size": 2,
    "delete_large_node_min_size": 100000,
}


def region_labels(adj):
    H2, W2 = adj.shape; H, W = (H2+1)//2, (W2+1)//2; n = H*W
    rr, rc = np.where((adj[::2, 1::2] & EDGE_MERGED) != 0)
    dr, dc = np.where((adj[1::2, ::2] & EDGE_MERGED) != 0)
    src = np.concatenate([rr*W+rc, dr*W+dc])
    dst = np.concatenate([rr*W+rc+1, dr*W+dc+W])
    if src.size == 0: return np.arange(n, dtype=np.int32).reshape(H,W)
    g = coo_matrix((np.ones(src.size, dtype=bool), (src, dst)), shape=(n,n)).tocsr()
    _, labels = connected_components(g, directed=False)
    return labels.astype(np.int32).reshape(H,W)


def windowed_to_uint8(ct_slice, wl=60, ww=200):
    """Liver-specific CT windowing."""
    lo, hi = wl - ww/2, wl + ww/2
    img = np.clip((ct_slice - lo) / (hi - lo), 0, 1)
    img = (img * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def build_graph_from_slice(img_uint8, seg_slice, prev_slice=None, next_slice=None):
    """Build supergraph with multi-slice context features."""
    H, W = img_uint8.shape[:2]
    adj, feat = fastloops.merge_and_cut(img_uint8, **MEDICAL_KERNEL)
    N = feat.shape[0]
    if N < 5:
        return None

    labels = region_labels(adj)
    canon = feat[:, 14:16].astype(np.int64)
    max_label = int(labels.max()) + 1
    cc_arr = np.full(max_label, -1, dtype=np.int32)
    for i in range(N):
        cc_arr[labels[int(canon[i,0]), int(canon[i,1])]] = i
    snode_map = cc_arr[labels]

    # Build edges (vectorized)
    h_nm = (adj[::2, 1::2] & EDGE_MERGED) == 0
    ll, lr = labels[:, :-1], labels[:, 1:]
    hd = (ll != lr) & h_nm
    hs1, hs2 = cc_arr[ll[hd]], cc_arr[lr[hd]]
    hv = (hs1 >= 0) & (hs2 >= 0) & (hs1 != hs2)
    hp = np.stack([hs1[hv], hs2[hv]], axis=1) if hv.any() else np.zeros((0,2), dtype=np.int32)

    v_nm = (adj[1::2, ::2] & EDGE_MERGED) == 0
    lt, lb = labels[:-1,:], labels[1:,:]
    vd = (lt != lb) & v_nm
    vs1, vs2 = cc_arr[lt[vd]], cc_arr[lb[vd]]
    vv = (vs1 >= 0) & (vs2 >= 0) & (vs1 != vs2)
    vp = np.stack([vs1[vv], vs2[vv]], axis=1) if vv.any() else np.zeros((0,2), dtype=np.int32)

    all_p = np.concatenate([hp, vp], axis=0)
    if all_p.shape[0] == 0:
        return None
    all_p = np.sort(all_p, axis=1)
    up = np.unique(all_p, axis=0)
    src = np.concatenate([up[:,0], up[:,1]])
    dst = np.concatenate([up[:,1], up[:,0]])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Node features: 14 base + 6 multi-slice context = 20 dim
    feat_f = feat.astype(np.float64)
    area = feat_f[:, 0].clip(min=1)

    # Base features (14-dim)
    nf = np.zeros((N, 20), dtype=np.float32)
    nf[:, 0] = np.log1p(area)
    nf[:, 1] = feat_f[:, 1] / area / W
    nf[:, 2] = feat_f[:, 2] / area / H
    nf[:, 3] = feat_f[:, 6] / area / 255.0
    nf[:, 4] = feat_f[:, 7] / area / 255.0
    nf[:, 5] = feat_f[:, 8] / area / 255.0
    nf[:, 6] = (feat_f[:, 10] - feat_f[:, 9]) / W
    nf[:, 7] = (feat_f[:, 12] - feat_f[:, 11]) / H
    nf[:, 8] = feat_f[:, 13] / area.clip(min=1)
    nf[:, 9] = (feat_f[:, 3] / area - (feat_f[:, 1]/area)**2).clip(min=0) / (W*W)
    nf[:, 10] = (feat_f[:, 4] / area - (feat_f[:, 2]/area)**2).clip(min=0) / (H*H)
    nf[:, 11] = np.sqrt(area) / W
    deg = np.bincount(src, minlength=N)[:N]
    nf[:, 12] = deg / max(deg.max(), 1)
    nf[:, 13] = (nf[:, 3] + nf[:, 4] + nf[:, 5]) / 3

    # Multi-slice context: mean intensity of same region in adjacent slices
    valid = snode_map >= 0
    if prev_slice is not None:
        prev_img = windowed_to_uint8(prev_slice)[:,:,0].astype(np.float32) / 255.0
        prev_sums = np.bincount(snode_map[valid], weights=prev_img.ravel()[valid.ravel()], minlength=N)[:N]
        counts = np.bincount(snode_map[valid], minlength=N)[:N].clip(min=1)
        nf[:, 14] = prev_sums / counts
        nf[:, 15] = nf[:, 3] - nf[:, 14]  # difference from prev slice
    if next_slice is not None:
        next_img = windowed_to_uint8(next_slice)[:,:,0].astype(np.float32) / 255.0
        next_sums = np.bincount(snode_map[valid], weights=next_img.ravel()[valid.ravel()], minlength=N)[:N]
        counts = np.bincount(snode_map[valid], minlength=N)[:N].clip(min=1)
        nf[:, 16] = next_sums / counts
        nf[:, 17] = nf[:, 3] - nf[:, 16]  # difference from next slice

    # Texture features: local intensity variance within each supernode
    img_gray = img_uint8[:,:,0].astype(np.float32) / 255.0
    img_sq = img_gray ** 2
    sum_sq = np.bincount(snode_map[valid], weights=img_sq.ravel()[valid.ravel()], minlength=N)[:N]
    sum_val = np.bincount(snode_map[valid], weights=img_gray.ravel()[valid.ravel()], minlength=N)[:N]
    counts = np.bincount(snode_map[valid], minlength=N)[:N].clip(min=1)
    nf[:, 18] = (sum_sq / counts - (sum_val / counts)**2).clip(min=0)  # intensity variance
    nf[:, 19] = nf[:, 8] * nf[:, 18]  # boundary_ratio * texture_variance

    # Per-node segmentation labels (majority vote)
    node_labels = np.zeros(N, dtype=np.int64)
    for nid in range(N):
        m = snode_map == nid
        if m.any():
            vals, counts_v = np.unique(seg_slice[m], return_counts=True)
            node_labels[nid] = vals[counts_v.argmax()]

    return Data(
        x=torch.tensor(nf, dtype=torch.float),
        edge_index=edge_index,
        node_y=torch.tensor(node_labels, dtype=torch.long),
        snode_map=torch.tensor(snode_map, dtype=torch.int32),
        mask_shape=torch.tensor([H, W], dtype=torch.int32),
        num_nodes=N,
    )


class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim, hid, layers=6, heads=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.convs = nn.ModuleList([TransformerConv(hid, hid//heads, heads=heads) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.drop = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x = self.proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + self.drop(F.gelu(norm(conv(x, edge_index))))
        return x


class IGJEPA(nn.Module):
    def __init__(self, in_dim, hid=256, layers=6, heads=4, mask_ratio=0.4, mom=0.996):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mom = mom
        self.hid = hid
        self.enc = GraphTransformerEncoder(in_dim, hid, layers, heads)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters(): p.requires_grad = False
        self.pred = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, hid))
        self.lam_var = 25.0
        self.lam_cov = 1.0

    @torch.no_grad()
    def ema_update(self):
        for p, t in zip(self.enc.parameters(), self.tgt.parameters()):
            t.data.mul_(self.mom).add_(p.data, alpha=1-self.mom)

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
        x, ei = batch.x, batch.edge_index
        N = x.size(0)
        mi, ctx_ei = self.subgraph_mask(ei, N)

        with torch.no_grad():
            tz = self.tgt(x, ei)
            tgt = F.layer_norm(tz[mi], [self.hid])

        xc = x.clone(); xc[mi] = 0
        cz = self.enc(xc, ctx_ei)
        pred = F.layer_norm(self.pred(cz[mi]), [self.hid])

        loss_pred = F.smooth_l1_loss(pred, tgt)
        ctx_mask = torch.ones(N, dtype=torch.bool, device=x.device); ctx_mask[mi] = False
        ctx_emb = cz[ctx_mask]; Nc = ctx_emb.size(0)
        std = ctx_emb.std(dim=0)
        loss_var = F.relu(1.0 - std).mean()
        z_c = ctx_emb - ctx_emb.mean(dim=0)
        cov = (z_c.T @ z_c) / max(Nc - 1, 1)
        loss_cov = (cov.fill_diagonal_(0) ** 2).sum() / self.hid

        return loss_pred + self.lam_var * loss_var + self.lam_cov * loss_cov, {"std": std.mean().item()}

    def encode_nodes(self, x, edge_index):
        return self.enc(x, edge_index)


class DiceLoss(nn.Module):
    def __init__(self, n_classes, weights=None):
        super().__init__()
        self.n_classes = n_classes
        self.weights = weights

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.n_classes).float()
        loss = 0
        for c in range(self.n_classes):
            p = probs[:, c]
            g = targets_oh[:, c]
            intersection = (p * g).sum()
            dice = (2 * intersection + 1) / (p.sum() + g.sum() + 1)
            w = self.weights[c] if self.weights is not None else 1.0
            loss += w * (1 - dice)
        return loss / self.n_classes


def compute_dice(pred_mask, gt_mask, class_id):
    p = (pred_mask == class_id)
    g = (gt_mask == class_id)
    intersection = (p & g).sum()
    return (2.0 * intersection) / max(p.sum() + g.sum(), 1)


def largest_connected_component(mask, class_id):
    """Keep only the largest connected component of a given class."""
    binary = (mask == class_id).astype(np.int32)
    labeled, n_components = scipy_label(binary)
    if n_components == 0:
        return mask
    # Find largest component
    sizes = np.bincount(labeled.ravel())[1:]  # skip background (0)
    if len(sizes) == 0:
        return mask
    largest = sizes.argmax() + 1
    # Zero out everything except largest
    result = mask.copy()
    result[(mask == class_id) & (labeled != largest)] = 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lits", "pancreas"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--classify_epochs", type=int, default=100)
    parser.add_argument("--max_slices", type=int, default=3000)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Config: hid={args.hid}, layers={args.layers}, pretrain={args.pretrain_epochs}, classify={args.classify_epochs}", flush=True)

    # ═══ Step 1: Load and extract slices with context ═══
    print("\nStep 1: Loading volumes...", flush=True)

    if args.dataset == "lits":
        ct_dir = "/dev/shm/medical_data/LiTS/ct"
        seg_dir = "/dev/shm/medical_data/LiTS/seg"
        ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.npy')])
        seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.npy')])
        n_train = int(len(ct_files) * 0.8)
        n_classes = 3

        def load_slices_with_context(vol_list, max_slices):
            slices = []
            for vf in vol_list:
                sf = vf.replace("volume", "segmentation")
                vol = np.load(os.path.join(ct_dir, vf))
                seg = np.load(os.path.join(seg_dir, sf))
                for s in range(vol.shape[0]):
                    if seg[s].max() > 0:
                        prev_s = vol[s-1] if s > 0 else None
                        next_s = vol[s+1] if s < vol.shape[0]-1 else None
                        slices.append((vol[s], seg[s], prev_s, next_s))
                    if len(slices) >= max_slices:
                        break
                if len(slices) >= max_slices:
                    break
            return slices

        train_slices = load_slices_with_context(ct_files[:n_train], args.max_slices)
        test_slices = load_slices_with_context(ct_files[n_train:], args.max_slices // 4)

    elif args.dataset == "pancreas":
        import nibabel
        img_dir = "/dev/shm/medical_data/Pancreas/imagesTr"
        lbl_dir = "/dev/shm/medical_data/Pancreas/labelsTr"
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz') and not f.startswith('._')])
        n_train = int(len(img_files) * 0.8)
        n_classes = 3

        def load_slices_with_context(file_list, max_slices):
            slices = []
            for f in file_list:
                if not os.path.exists(os.path.join(lbl_dir, f)):
                    continue
                vol = nibabel.load(os.path.join(img_dir, f)).get_fdata()
                seg = nibabel.load(os.path.join(lbl_dir, f)).get_fdata().astype(np.uint8)
                # Downsample 512→256 for speed
                if vol.shape[0] > 256:
                    vol = vol[::2, ::2, :]
                    seg = seg[::2, ::2, :]
                for s in range(vol.shape[2]):
                    if seg[:,:,s].max() > 0:
                        prev_s = vol[:,:,s-1] if s > 0 else None
                        next_s = vol[:,:,s+1] if s < vol.shape[2]-1 else None
                        slices.append((vol[:,:,s], seg[:,:,s], prev_s, next_s))
                    if len(slices) >= max_slices:
                        break
                if len(slices) >= max_slices:
                    break
            return slices

        train_slices = load_slices_with_context(img_files[:n_train], args.max_slices)
        test_slices = load_slices_with_context(img_files[n_train:], args.max_slices // 4)

    print(f"  Train slices: {len(train_slices)}, Test slices: {len(test_slices)}", flush=True)

    # ═══ Step 2: Build graphs ═══
    print("\nStep 2: Building fine supergraphs...", flush=True)
    t0 = time.time()

    def build_graphs(slice_list):
        graphs = []
        for i, (ct_s, seg_s, prev_s, next_s) in enumerate(slice_list):
            img = windowed_to_uint8(ct_s)
            g = build_graph_from_slice(img, seg_s.astype(np.uint8), prev_s, next_s)
            if g is not None and g.num_nodes <= 5000:
                graphs.append(g)
            if (i+1) % 500 == 0:
                print(f"    {i+1}/{len(slice_list)}", flush=True)
        return graphs

    train_graphs = build_graphs(train_slices)
    test_graphs = build_graphs(test_slices)
    del train_slices, test_slices

    print(f"  Built in {time.time()-t0:.1f}s: {len(train_graphs)} train, {len(test_graphs)} test", flush=True)
    if train_graphs:
        avg_nodes = np.mean([g.num_nodes for g in train_graphs])
        print(f"  Nodes avg: {avg_nodes:.0f}, Feature dim: {train_graphs[0].x.shape[1]}", flush=True)

    # ═══ Step 3: JEPA Pretraining ═══
    print(f"\nStep 3: IG-JEPA Pretraining ({args.pretrain_epochs} epochs)...", flush=True)
    train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.bs)
    in_dim = train_graphs[0].x.shape[1]

    model = IGJEPA(in_dim, args.hid, args.layers).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=args.pretrain_epochs, eta_min=1e-6)

    for ep in range(1, args.pretrain_epochs+1):
        model.train(); tl, n = 0, 0
        for b in train_loader:
            b = b.to(device); opt.zero_grad()
            loss, info = model(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); model.ema_update()
            tl += loss.item(); n += 1
        sch.step()
        if ep % 20 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | Loss: {tl/n:.4f} | std: {info['std']:.4f}", flush=True)

    # ═══ Step 4: MLP Node Classifier with Dice Loss ═══
    print(f"\nStep 4: MLP Node Classifier ({args.classify_epochs} epochs)...", flush=True)
    model.eval()
    for p in model.enc.parameters(): p.requires_grad = False

    # MLP classifier (3 layers instead of linear)
    node_clf = nn.Sequential(
        nn.Linear(args.hid, args.hid),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(args.hid, args.hid // 2),
        nn.GELU(),
        nn.Linear(args.hid // 2, n_classes),
    ).to(device)

    clf_opt = torch.optim.Adam(node_clf.parameters(), lr=1e-3)
    clf_sch = CosineAnnealingLR(clf_opt, T_max=args.classify_epochs, eta_min=1e-5)
    dice_loss = DiceLoss(n_classes, weights=torch.tensor([0.1, 1.0, 2.0], device=device))
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 5.0], device=device))

    for ep in range(1, args.classify_epochs+1):
        node_clf.train(); tl, n = 0, 0
        for b in train_loader:
            b = b.to(device)
            with torch.no_grad():
                emb = model.encode_nodes(b.x, b.edge_index)
            logits = node_clf(emb)
            loss = ce_loss(logits, b.node_y) + dice_loss(logits, b.node_y)
            clf_opt.zero_grad(); loss.backward(); clf_opt.step()
            tl += loss.item(); n += 1
        clf_sch.step()
        if ep % 20 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | Loss: {tl/n:.4f}", flush=True)

    # ═══ Step 5: Evaluate with post-processing ═══
    print(f"\nStep 5: Evaluation + post-processing...", flush=True)
    node_clf.eval()
    dice_scores = {c: [] for c in range(1, n_classes)}
    dice_scores_pp = {c: [] for c in range(1, n_classes)}  # with post-processing

    for data in test_graphs:
        data_d = data.to(device)
        with torch.no_grad():
            emb = model.encode_nodes(data_d.x, data_d.edge_index)
            preds = node_clf(emb).argmax(dim=1).cpu().numpy()

        snode_map = data.snode_map.cpu().numpy()
        H, W = data.mask_shape[0].item(), data.mask_shape[1].item()
        pred_mask = np.zeros((H, W), dtype=np.int64)
        gt_mask = np.zeros((H, W), dtype=np.int64)

        for nid in range(data.num_nodes):
            m = snode_map == nid
            pred_mask[m] = preds[nid]
            gt_mask[m] = data.node_y[nid].item()

        # Raw Dice
        for c in range(1, n_classes):
            if (gt_mask == c).any():
                dice_scores[c].append(compute_dice(pred_mask, gt_mask, c))

        # Post-processed: largest connected component for class 1 (organ)
        pred_pp = largest_connected_component(pred_mask, 1)
        for c in range(1, n_classes):
            if (gt_mask == c).any():
                dice_scores_pp[c].append(compute_dice(pred_pp, gt_mask, c))

    # ═══ Results ═══
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS — {args.dataset}", flush=True)
    print(f"{'='*60}", flush=True)

    labels = {1: "Liver" if args.dataset == "lits" else "Pancreas", 2: "Tumor"}
    sig = {"dataset": args.dataset, "n_train": len(train_graphs), "n_test": len(test_graphs)}

    for c, name in labels.items():
        raw = dice_scores[c]
        pp = dice_scores_pp[c]
        if raw:
            raw_mean = np.mean(raw) * 100
            pp_mean = np.mean(pp) * 100
            print(f"  {name:12s} Dice (raw): {raw_mean:.2f}%  Dice (post-proc): {pp_mean:.2f}%  (n={len(raw)})", flush=True)
            sig[f"dice_{name.lower()}_raw"] = float(np.mean(raw))
            sig[f"dice_{name.lower()}_pp"] = float(np.mean(pp))
        else:
            print(f"  {name:12s} Dice: N/A", flush=True)

    sig_path = f"/home/ud3d4/Desktop/Projects/NIPS 26/signals/done_{args.dataset}_medical_v2.json"
    with open(sig_path, "w") as f:
        json.dump(sig, f, indent=2)
    print(f"\nSignal: {sig_path}", flush=True)

    print(f"\nSOTA targets:", flush=True)
    if args.dataset == "lits":
        print(f"  Liver: 99.4% (UNet++-ViT, 2025)", flush=True)
        print(f"  Tumor: 93.12% (DynTransNet, 2025)", flush=True)
    else:
        print(f"  Pancreas: 94.0% (3D nnU-Net, 2025)", flush=True)


if __name__ == "__main__":
    main()
