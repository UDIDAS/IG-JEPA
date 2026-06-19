"""
KG-JEPA: Joint Embedding Predictive Architecture on Clinical Knowledge Graphs.

Converts medical KGs (LiTS, Pancreas) into PyG graphs and applies JEPA
for self-supervised clinical reasoning.

Tasks:
  1. Tumor burden classification (3-class: minimal/moderate/extensive)
  2. Tumor size regression (predict volume)
  3. Node property prediction from masked graph context

KG Schema: Patient → CTScan → Tumor → connectedToOrgan → Organ
            CTScan → hasTumor → Tumor
            Tumor → hasVisualization → Image → hasFeature → ImageFeature
"""

import os, sys, copy, argparse, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import TransformerConv, global_mean_pool, SAGEConv
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from rdflib import Graph as RDFGraph, Namespace, RDF, Literal
import warnings
warnings.filterwarnings('ignore')

EX = Namespace('http://example.org/pancreas-kg#')


def load_kg_as_pyg(kg_path, dataset_name):
    """
    Convert an RDF KG into a PyG homogeneous graph.

    Nodes: Tumors (primary entities with rich features)
    Node features: [volume, coverage%, distance, 512-dim CLIP embedding of associated image,
                    8 zero-shot feature confidences]
    Edges: tumors from same patient/scan are connected
    Labels: tumor burden class (from zero-shot feature)
    """
    g = RDFGraph()
    g.parse(kg_path, format='turtle')
    print(f"  Loaded {kg_path}: {len(g)} triples", flush=True)

    # Get all tumors with their properties
    tumors = list(g.subjects(RDF.type, EX.Tumor))
    tumor_data = {}

    for t in tumors:
        tid = str(t).split('#')[-1]
        vol = list(g.objects(t, EX.tumorVolume))
        cov = list(g.objects(t, EX.tumorCoveragePercent))
        dist = list(g.objects(t, EX.distanceToConnectedOrgan))

        # Get associated image via scan
        # Tumor → (part of scan) → scan → hasVisualization → Image
        # Actually: scan hasTumor tumor, and tumor hasVisualization image
        img = list(g.objects(t, EX.hasVisualization))

        tumor_data[tid] = {
            'volume': float(str(vol[0])) if vol else 0,
            'coverage': float(str(cov[0])) if cov else 0,
            'distance': float(str(dist[0])) if dist else 0,
            'image_uri': img[0] if img else None,
        }

    # Get CLIP embeddings and zero-shot features per image
    image_features = {}
    for img_uri in g.subjects(RDF.type, EX.Image):
        emb_str = list(g.objects(img_uri, EX.visualEmbedding))
        if emb_str:
            emb = np.array([float(x) for x in str(emb_str[0]).split(',')], dtype=np.float32)
        else:
            emb = np.zeros(512, dtype=np.float32)

        # Get zero-shot feature confidences
        zs_features = {}
        for feat_uri in g.objects(img_uri, EX.hasFeature):
            fname = list(g.objects(feat_uri, EX.featureName))
            fconf = list(g.objects(feat_uri, EX.confidenceScore))
            fval = list(g.objects(feat_uri, EX.featureValue))
            if fname and fconf:
                zs_features[str(fname[0])] = {
                    'confidence': float(str(fconf[0])),
                    'value': str(fval[0]) if fval else ''
                }

        image_features[str(img_uri)] = {'embedding': emb, 'zs_features': zs_features}

    # Build node features for each tumor
    feat_names = ['tumorBurden', 'tumorShape', 'tumorPosition', 'tumorOrganRelation',
                  'organCount', 'structuralComplexity', 'tumorVisibility', 'spatialExtent']

    node_features = []
    node_labels = []  # tumor burden class
    tumor_ids = []

    for tid, tdata in tumor_data.items():
        # Base features: volume, coverage, distance (3-dim)
        base = np.array([
            np.log1p(tdata['volume']),
            tdata['coverage'],
            tdata['distance']
        ], dtype=np.float32)

        # CLIP embedding (512-dim) + zero-shot confidences (8-dim)
        img_uri = tdata['image_uri']
        if img_uri and str(img_uri) in image_features:
            img_data = image_features[str(img_uri)]
            clip_emb = img_data['embedding']
            zs_conf = np.array([
                img_data['zs_features'].get(fn, {}).get('confidence', 0.0)
                for fn in feat_names
            ], dtype=np.float32)

            # Label will be set later based on volume percentile (more reliable)
            label = 0  # placeholder
        else:
            clip_emb = np.zeros(512, dtype=np.float32)
            zs_conf = np.zeros(8, dtype=np.float32)
            label = 0

        # Full feature: 3 + 512 + 8 = 523 dim
        feat = np.concatenate([base, clip_emb, zs_conf])
        node_features.append(feat)
        node_labels.append(label)
        tumor_ids.append(tid)

    node_features = np.stack(node_features)
    node_labels = np.array(node_labels)

    # Build edges: tumors from same scan are connected
    scan_to_tumors = {}
    for scan_uri in g.subjects(RDF.type, EX.CTScan):
        tumor_uris = list(g.objects(scan_uri, EX.hasTumor))
        scan_tumors = []
        for tu in tumor_uris:
            tid = str(tu).split('#')[-1]
            if tid in tumor_data:
                idx = tumor_ids.index(tid)
                scan_tumors.append(idx)
        if len(scan_tumors) > 1:
            scan_to_tumors[str(scan_uri)] = scan_tumors

    # Create edges (fully connected within each scan)
    src, dst = [], []
    for scan_id, tidxs in scan_to_tumors.items():
        for i in range(len(tidxs)):
            for j in range(i + 1, len(tidxs)):
                src.extend([tidxs[i], tidxs[j]])
                dst.extend([tidxs[j], tidxs[i]])

    # Also connect tumors to their spatial neighbors (by distance in feature space)
    # KNN graph on tumor positions for better connectivity
    from sklearn.neighbors import kneighbors_graph
    if len(node_features) > 10:
        pos_features = node_features[:, :3]  # volume, coverage, distance
        k = min(5, len(node_features) - 1)
        knn = kneighbors_graph(pos_features, k, mode='connectivity', include_self=False)
        knn_src, knn_dst = knn.nonzero()
        src.extend(knn_src.tolist())
        dst.extend(knn_dst.tolist())

    if len(src) == 0:
        # Fallback: chain all tumors
        for i in range(len(tumor_ids) - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Create labels based on tumor volume percentiles (3-class: small/medium/large)
    volumes = np.array([tumor_data[tid]['volume'] for tid in tumor_ids])
    p33 = np.percentile(volumes, 33)
    p66 = np.percentile(volumes, 66)
    node_labels = np.where(volumes < p33, 0, np.where(volumes < p66, 1, 2))

    # Create one big graph
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(node_labels, dtype=torch.long),
        num_nodes=len(tumor_ids),
    )

    # Also store regression target (tumor volume)
    data.volume = torch.tensor(volumes, dtype=torch.float)

    print(f"  Graph: {data.num_nodes} nodes, {edge_index.shape[1]} edges, {node_features.shape[1]}-dim features", flush=True)
    print(f"  Labels: {np.bincount(node_labels)}", flush=True)

    return data


class KGJEPA(nn.Module):
    """JEPA on knowledge graph structure."""
    def __init__(self, in_dim, hid=256, layers=4, heads=4, mask_ratio=0.3):
        super().__init__()
        self.hid = hid
        self.mask_ratio = mask_ratio
        self.enc = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.proj = nn.Linear(in_dim, hid)
        for _ in range(layers):
            self.enc.append(SAGEConv(hid, hid))
            self.norms.append(nn.LayerNorm(hid))
        self.tgt_proj = nn.Linear(in_dim, hid)
        self.tgt_enc = nn.ModuleList()
        self.tgt_norms = nn.ModuleList()
        for _ in range(layers):
            self.tgt_enc.append(SAGEConv(hid, hid))
            self.tgt_norms.append(nn.LayerNorm(hid))
        # Copy weights
        self.tgt_proj.load_state_dict(self.proj.state_dict())
        for te, e in zip(self.tgt_enc, self.enc):
            te.load_state_dict(e.state_dict())
        for tn, n in zip(self.tgt_norms, self.norms):
            tn.load_state_dict(n.state_dict())
        # Freeze target
        for p in self.tgt_proj.parameters(): p.requires_grad = False
        for m in self.tgt_enc:
            for p in m.parameters(): p.requires_grad = False
        for m in self.tgt_norms:
            for p in m.parameters(): p.requires_grad = False

        self.predictor = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, hid))
        self.mom = 0.996

    def encode(self, x, edge_index):
        x = self.proj(x)
        for conv, norm in zip(self.enc, self.norms):
            x = F.gelu(norm(conv(x, edge_index))) + x
        return x

    @torch.no_grad()
    def encode_target(self, x, edge_index):
        x = self.tgt_proj(x)
        for conv, norm in zip(self.tgt_enc, self.tgt_norms):
            x = F.gelu(norm(conv(x, edge_index))) + x
        return x

    @torch.no_grad()
    def ema_update(self):
        for p, t in zip(self.proj.parameters(), self.tgt_proj.parameters()):
            t.data.mul_(self.mom).add_(p.data, alpha=1 - self.mom)
        for (ec, tc) in zip(self.enc, self.tgt_enc):
            for p, t in zip(ec.parameters(), tc.parameters()):
                t.data.mul_(self.mom).add_(p.data, alpha=1 - self.mom)
        for (en, tn) in zip(self.norms, self.tgt_norms):
            for p, t in zip(en.parameters(), tn.parameters()):
                t.data.mul_(self.mom).add_(p.data, alpha=1 - self.mom)

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

    def forward(self, data):
        x, ei = data.x, data.edge_index
        N = x.size(0)
        mi, ctx_ei = self.subgraph_mask(ei, N)

        # Target sees full graph
        tz = self.encode_target(x, ei)
        tgt = F.layer_norm(tz[mi], [self.hid])

        # Context: masked features + edges to masked nodes removed
        xc = x.clone(); xc[mi] = 0
        cz = self.encode(xc, ctx_ei)
        pred = F.layer_norm(self.predictor(cz[mi]), [self.hid])

        loss_pred = F.smooth_l1_loss(pred, tgt)

        # VICReg on context nodes only
        ctx_mask = torch.ones(N, dtype=torch.bool, device=x.device); ctx_mask[mi] = False
        ctx_emb = cz[ctx_mask]
        std = ctx_emb.std(dim=0)
        loss_var = F.relu(1.0 - std).mean()

        return loss_pred + 25 * loss_var, {'std': std.mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lits", "pancreas", "both"], default="both")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hid", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load KGs
    kg_paths = {
        'lits': '/home/ud3d4/Desktop/Projects/MICCAI_26/kg/kgLiTSv2.ttl',
        'pancreas': '/home/ud3d4/Desktop/Projects/MICCAI_26/kg/kg_niiv2.ttl',
    }

    datasets_to_run = ['lits', 'pancreas'] if args.dataset == 'both' else [args.dataset]

    for ds_name in datasets_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"KG-JEPA — {ds_name}", flush=True)
        print(f"{'='*60}", flush=True)

        data = load_kg_as_pyg(kg_paths[ds_name], ds_name).to(device)

        # Train/test split (80/20 by node)
        N = data.num_nodes
        perm = torch.randperm(N)
        n_train = int(0.8 * N)
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        test_mask = ~train_mask

        # JEPA Pretraining
        print(f"\nPhase 1: KG-JEPA Pretraining ({args.epochs} epochs)...", flush=True)
        model = KGJEPA(data.x.shape[1], args.hid, layers=4).to(device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        for ep in range(1, args.epochs + 1):
            model.train()
            opt.zero_grad()
            loss, info = model(data)
            loss.backward()
            opt.step()
            model.ema_update()
            if ep % 50 == 0 or ep == 1:
                print(f"  Ep {ep:3d} | Loss: {loss.item():.4f} | std: {info['std']:.4f}", flush=True)

        # Evaluate: tumor burden classification
        print(f"\nPhase 2: Evaluation...", flush=True)
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(data.x, data.edge_index).cpu().numpy()

        y = data.y.cpu().numpy()
        X_train, y_train = embeddings[train_mask.numpy()], y[train_mask.numpy()]
        X_test, y_test = embeddings[test_mask.numpy()], y[test_mask.numpy()]

        # Classification: tumor burden
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Regression: tumor volume
        volumes = data.volume.cpu().numpy()
        v_train, v_test = volumes[train_mask.numpy()], volumes[test_mask.numpy()]
        from sklearn.linear_model import Ridge
        reg = Ridge()
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        reg.fit(X_train_s, np.log1p(v_train))
        v_pred = np.expm1(reg.predict(X_test_s))
        mae = mean_absolute_error(v_test, v_pred)
        # Relative MAE
        rel_mae = mae / max(np.mean(v_test), 1)

        # Raw feature baseline
        X_raw_train = data.x[train_mask].cpu().numpy()
        X_raw_test = data.x[test_mask].cpu().numpy()
        clf_raw = LogisticRegression(max_iter=500)
        clf_raw.fit(X_raw_train, y_train)
        acc_raw = accuracy_score(y_test, clf_raw.predict(X_raw_test))

        print(f"\n  RESULTS — {ds_name}", flush=True)
        print(f"  Task: Tumor Burden Classification (3-class)", flush=True)
        print(f"    KG-JEPA + LogReg:    Acc={acc*100:.2f}%  F1={f1:.4f}", flush=True)
        print(f"    Raw features + LogReg: Acc={acc_raw*100:.2f}%", flush=True)
        print(f"    Delta: {(acc-acc_raw)*100:+.2f}%", flush=True)
        print(f"  Task: Tumor Volume Prediction", flush=True)
        print(f"    MAE: {mae:.0f} voxels  (relative: {rel_mae:.2f})", flush=True)
        print(f"    Mean test volume: {np.mean(v_test):.0f}", flush=True)

        # Save signal
        sig = {
            'dataset': ds_name,
            'task': 'tumor_burden_classification',
            'acc_jepa': float(acc),
            'f1_jepa': float(f1),
            'acc_raw': float(acc_raw),
            'volume_mae': float(mae),
            'volume_rel_mae': float(rel_mae),
            'n_train': int(n_train),
            'n_test': int(N - n_train),
        }
        sig_path = f"/home/ud3d4/Desktop/Projects/NIPS 26/signals/done_{ds_name}_kg_jepa.json"
        with open(sig_path, 'w') as f:
            json.dump(sig, f, indent=2)
        print(f"  Signal: {sig_path}", flush=True)


if __name__ == "__main__":
    main()
