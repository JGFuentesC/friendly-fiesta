"""Reentrena los mejores genomas del GA en el train grande y evalúa en test.

Elige un ÚNICO modelo FF (sin ensamble): top-3 genomas × 5 semillas, selección
por recall@FPR<=0.5% en validación, umbral en val, evaluación final en test.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)

MAX_FPR = 0.005
SEEDS = [42, 43, 44, 45, 46]

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


def build_mlp(n_in, width, depth, dropout, activation, norm):
    act = ACTIVATIONS[activation]
    layers = []
    for i in range(depth):
        in_f = n_in if i == 0 else width
        layers.append(nn.Linear(in_f, width))
        if norm == "batch":
            layers.append(nn.BatchNorm1d(width))
        elif norm == "layer":
            layers.append(nn.LayerNorm(width))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def recall_at_fpr(y_true, probs, max_fpr):
    fpr, tpr, _ = roc_curve(y_true, probs)
    idx = np.where(fpr <= max_fpr)[0]
    return float(tpr[idx[-1]]) if len(idx) else 0.0


def threshold_at_fpr(y_true, probs, max_fpr):
    fpr, tpr, th = roc_curve(y_true, probs)
    idx = np.where(fpr <= max_fpr)[0]
    if len(idx) == 0:
        return 0.5
    i = idx[-1]
    return float(th[i]) if i < len(th) else float(th[-1])


def train_full(genome, Xtr, ytr, Xv, yv_np, pos_weight, seed, epochs=50,
               batch=16384, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_mlp(Xtr.shape[1], genome["width"], genome["depth"],
                      genome["dropout"], genome["activation"], genome["norm"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=genome["lr"],
                           weight_decay=genome["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    scaler = torch.cuda.amp.GradScaler()
    n = Xtr.shape[0]

    best_pr, best_state, no_improve = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = loss_fn(model(Xtr[idx]).squeeze(1), ytr[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv).squeeze(1)).cpu().numpy()
        pr = average_precision_score(yv_np, pv)
        if pr > best_pr:
            best_pr, no_improve = pr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model


def top_genomes(log_path, k=3):
    recs = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("phase") == "phase2"]
    if not recs:
        recs = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
    key = lambda r: (r["lr"], r["width"], r["depth"], r["dropout"],
                     r["weight_decay"], r["activation"], r["norm"])
    best = {}
    for r in recs:
        kk = key(r)
        if kk not in best or r["recall_at_fpr"] > best[kk]["recall_at_fpr"]:
            best[kk] = r
    ranked = sorted(best.values(),
                    key=lambda r: (-r["recall_at_fpr"], -r["pr_auc"]))
    return ranked[:k]


def main():
    root = Path("/workspace/friendly-fiesta")
    if not (root / "data").exists():
        root = Path.cwd()
    results_dir = root / "results"

    genomes = top_genomes(results_dir / "ga_log.jsonl", k=3)
    print("top genomas:")
    for g in genomes:
        print(" ", {k: g[k] for k in
                    ("lr", "width", "depth", "dropout", "weight_decay",
                     "activation", "norm", "recall_at_fpr", "pr_auc")})

    # ---- datos ----
    tr = pd.read_csv(root / "data/processed/fraud_train_big.csv")
    val = pd.read_csv(root / "data/processed/fraud_val.csv")
    test = pd.read_csv(root / "data/processed/fraud_test.csv")
    feats = [c for c in tr.columns if c not in ("id", "is_fraud")]
    scaler = StandardScaler().fit(tr[feats])

    Xtr = torch.tensor(scaler.transform(tr[feats]).astype(np.float32), device=DEVICE)
    ytr = torch.tensor(tr["is_fraud"].to_numpy(np.float32), device=DEVICE)
    Xv = torch.tensor(scaler.transform(val[feats]).astype(np.float32), device=DEVICE)
    yv_np = val["is_fraud"].to_numpy(np.float32)
    Xt = torch.tensor(scaler.transform(test[feats]).astype(np.float32), device=DEVICE)
    yt_np = test["is_fraud"].to_numpy(np.float32)
    pos_weight = (1 - ytr.mean().item()) / ytr.mean().item()
    print(f"train {Xtr.shape} val {Xv.shape} test {Xt.shape} "
          f"pos_weight={pos_weight:.2f}", flush=True)

    best_overall = None
    for gi, genome in enumerate(genomes):
        for seed in SEEDS:
            t0 = time.time()
            model = train_full(genome, Xtr, ytr, Xv, yv_np, pos_weight, seed)
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xv).squeeze(1)).cpu().numpy()
            r = recall_at_fpr(yv_np, pv, MAX_FPR)
            pr = average_precision_score(yv_np, pv)
            cand = {"genome": genome, "seed": seed, "val_recall": r,
                    "val_pr": pr, "model": model}
            print(f"  genoma {gi} seed {seed}: val recall@0.5%={r:.4f} "
                  f"pr={pr:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if best_overall is None or (r > best_overall["val_recall"] or
                                        (r == best_overall["val_recall"] and
                                         pr > best_overall["val_pr"])):
                best_overall = cand

    # ---- evaluación final en test ----
    model = best_overall["model"]
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xv).squeeze(1)).cpu().numpy()
        pt = torch.sigmoid(model(Xt).squeeze(1)).cpu().numpy()

    thr = threshold_at_fpr(yv_np, pv, MAX_FPR)
    pred = (pt >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt_np, pred).ravel()

    metrics = {
        "best_genome": {k: best_overall["genome"][k] for k in
                        ("lr", "width", "depth", "dropout", "weight_decay",
                         "activation", "norm")},
        "seed": best_overall["seed"],
        "threshold": thr,
        "val_recall_at_fpr": best_overall["val_recall"],
        "test": {
            "fpr": fp / (fp + tn),
            "recall": tp / (tp + fn),
            "precision": tp / (tp + fp),
            "roc_auc": roc_auc_score(yt_np, pt),
            "pr_auc": average_precision_score(yt_np, pt),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "n_test": int(len(yt_np)),
        },
    }
    (results_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), results_dir / "final_model.pt")

    print("\n===== RESULTADOS FINALES (test) =====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
