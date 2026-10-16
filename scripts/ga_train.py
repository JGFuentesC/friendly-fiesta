"""Búsqueda de hiperparámetros de una red FF con algoritmo genético (GPU).

Corre en platypy (lab-pytorch, RTX 3070). Usa AMP, datos precargados en VRAM y
checkpointing continuo en `results/` por si el proceso se interrumpe.

Dos fases: (1) gruesa sobre todo el espacio, (2) fina alrededor del mejor de (1).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score

SEED = 42
MAX_FPR = 0.005  # fitness: recall a FPR <= 0.5 % (margen para test < 1 %)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}

P1 = {
    "lr": (1e-4, 1e-1),
    "width": (16, 512),
    "depth": (1, 5),
    "dropout": (0.0, 0.5),
    "weight_decay": (1e-6, 1e-2),
    "activation": ["relu", "gelu", "tanh", "silu"],
    "norm": ["batch", "layer", "none"],
}


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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def clip(v, rng_range):
    lo, hi = rng_range
    return max(lo, min(hi, v))


def sample_genome(rng, R):
    return {
        "lr": float(10 ** rng.uniform(np.log10(R["lr"][0]), np.log10(R["lr"][1]))),
        "width": int(round(2 ** rng.uniform(np.log2(R["width"][0]), np.log2(R["width"][1])))),
        "depth": int(rng.integers(R["depth"][0], R["depth"][1] + 1)),
        "dropout": float(rng.uniform(R["dropout"][0], R["dropout"][1])),
        "weight_decay": float(10 ** rng.uniform(np.log10(R["weight_decay"][0]), np.log10(R["weight_decay"][1]))),
        "activation": R["activation"][int(rng.integers(len(R["activation"])))],
        "norm": R["norm"][int(rng.integers(len(R["norm"])))],
    }


def mutate(g, rng, R, sigma=0.35):
    g = dict(g)
    if rng.random() < 0.3:
        g["lr"] = clip(g["lr"] * 10 ** rng.normal(0, sigma), R["lr"])
    if rng.random() < 0.3:
        g["width"] = int(clip(round(g["width"] * 2 ** rng.normal(0, sigma)), R["width"]))
    if rng.random() < 0.3:
        g["depth"] = int(rng.integers(R["depth"][0], R["depth"][1] + 1))
    if rng.random() < 0.3:
        g["dropout"] = clip(g["dropout"] + rng.normal(0, 0.1), R["dropout"])
    if rng.random() < 0.3:
        g["weight_decay"] = clip(g["weight_decay"] * 10 ** rng.normal(0, sigma), R["weight_decay"])
    if rng.random() < 0.2:
        g["activation"] = R["activation"][int(rng.integers(len(R["activation"])))]
    if rng.random() < 0.2:
        g["norm"] = R["norm"][int(rng.integers(len(R["norm"])))]
    return g


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def recall_at_fpr(y_true, probs, max_fpr):
    fpr, tpr, _ = roc_curve(y_true, probs)
    idx = np.where(fpr <= max_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


class Evaluator:
    def __init__(self, Xtr, ytr, Xv, yv_np, n_in, pos_weight,
                 epochs=30, batch=16384, patience=5):
        self.Xtr, self.ytr = Xtr, ytr
        self.Xv, self.yv_np = Xv, yv_np
        self.n_in = n_in
        self.pos_weight = pos_weight
        self.epochs, self.batch, self.patience = epochs, batch, patience

    def __call__(self, g):
        model = build_mlp(self.n_in, g["width"], g["depth"],
                          g["dropout"], g["activation"], g["norm"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=g["lr"],
                               weight_decay=g["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], device=DEVICE))
        scaler = torch.cuda.amp.GradScaler()
        n = self.Xtr.shape[0]

        best_pr, best_state, no_improve = -1.0, None, 0
        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = loss_fn(model(self.Xtr[idx]).squeeze(1), self.ytr[idx])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(self.Xv).squeeze(1)).cpu().numpy()
            pr = average_precision_score(self.yv_np, pv)
            if pr > best_pr:
                best_pr, no_improve = pr, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(self.Xv).squeeze(1)).cpu().numpy()
        return recall_at_fpr(self.yv_np, pv, MAX_FPR), best_pr


def tournament(ranked_indices, fitness, rng, k=3):
    cand = rng.choice(ranked_indices, size=k, replace=False)
    return int(cand[np.argmax([fitness[i][0] for i in cand])])


def run_phase(R, evaluator, pop_size, gens, elite, seed, phase, log_fh,
              results_dir, best_path):
    rng = np.random.default_rng(seed)
    pop = [sample_genome(rng, R) for _ in range(pop_size)]
    fit = []
    for g in pop:
        f = evaluator(g)
        fit.append(f)
        rec = {"phase": phase, "gen": 0, "recall_at_fpr": f[0],
               "pr_auc": f[1], **g}
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()
        print(f"[{phase}] gen 0 init: recall@0.5%={f[0]:.4f} pr={f[1]:.4f} "
              f"w={g['width']} d={g['depth']} lr={g['lr']:.2e}", flush=True)

    best_global = None
    for gen in range(1, gens + 1):
        ranked = sorted(range(pop_size), key=lambda i: -fit[i][0])
        elite_idx = ranked[:elite]
        new_pop = [pop[i] for i in elite_idx]
        new_fit = [fit[i] for i in elite_idx]
        while len(new_pop) < pop_size:
            pa = pop[tournament(ranked, fit, rng)]
            pb = pop[tournament(ranked, fit, rng)]
            child = mutate(crossover(pa, pb, rng), rng, R)
            new_pop.append(child)
            f = evaluator(child)
            new_fit.append(f)
            rec = {"phase": phase, "gen": gen, "recall_at_fpr": f[0],
                   "pr_auc": f[1], **child}
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
        pop, fit = new_pop, new_fit

        bi = int(np.argmax([f[0] for f in fit]))
        best_global = pop[bi]
        (results_dir / best_path).write_text(json.dumps(best_global, indent=2))
        print(f"[{phase}] gen {gen}: best recall@0.5%={fit[bi][0]:.4f} "
              f"pr={fit[bi][1]:.4f} w={pop[bi]['width']} d={pop[bi]['depth']} "
              f"lr={pop[bi]['lr']:.2e}", flush=True)
    return best_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "both"], default="both")
    ap.add_argument("--pop1", type=int, default=24)
    ap.add_argument("--gens1", type=int, default=12)
    ap.add_argument("--pop2", type=int, default=12)
    ap.add_argument("--gens2", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    root = Path("/workspace/friendly-fiesta")
    if not (root / "data").exists():
        root = Path.cwd()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "ga_log.jsonl"
    log_fh = open(log_path, "a")

    print("DEVICE:", DEVICE, flush=True)
    t0 = time.time()

    # ---- datos ----
    train = pd.read_csv(root / "data/processed/fraud_train.csv")
    val = pd.read_csv(root / "data/processed/fraud_val.csv")
    feats = [c for c in train.columns if c not in ("id", "is_fraud")]
    scaler = StandardScaler().fit(train[feats])
    Xtr = torch.tensor(scaler.transform(train[feats]).astype(np.float32), device=DEVICE)
    ytr = torch.tensor(train["is_fraud"].to_numpy(np.float32), device=DEVICE)
    Xv = torch.tensor(scaler.transform(val[feats]).astype(np.float32), device=DEVICE)
    yv_np = val["is_fraud"].to_numpy(np.float32)
    pos_weight = (1 - ytr.mean().item()) / ytr.mean().item()
    n_in = len(feats)
    print(f"train {Xtr.shape}  val {Xv.shape}  pos_weight={pos_weight:.2f} "
          f"features={n_in}", flush=True)

    evaluator = Evaluator(Xtr, ytr, Xv, yv_np, n_in, pos_weight,
                          epochs=args.epochs)

    best1_path = results_dir / "phase1_best.json"
    best2_path = results_dir / "phase2_best.json"

    if args.phase in ("1", "both"):
        best1 = run_phase(P1, evaluator, args.pop1, args.gens1, elite=2,
                          seed=SEED, phase="phase1", log_fh=log_fh,
                          results_dir=results_dir, best_path="phase1_best.json")
    else:
        best1 = json.loads(best1_path.read_text())

    if args.phase in ("2", "both"):
        R2 = {
            "lr": (best1["lr"] / 3, best1["lr"] * 3),
            "width": (max(16, best1["width"] // 2), best1["width"] * 2),
            "depth": (max(1, best1["depth"] - 1), min(5, best1["depth"] + 1)),
            "dropout": (max(0.0, best1["dropout"] - 0.15), min(0.5, best1["dropout"] + 0.15)),
            "weight_decay": (best1["weight_decay"] / 3, best1["weight_decay"] * 3),
            "activation": P1["activation"],
            "norm": P1["norm"],
        }
        best2 = run_phase(R2, evaluator, args.pop2, args.gens2, elite=2,
                          seed=SEED + 1, phase="phase2", log_fh=log_fh,
                          results_dir=results_dir, best_path="phase2_best.json")
    else:
        best2 = best1

    log_fh.close()
    print(f"\nTOTAL {time.time() - t0:.1f}s", flush=True)
    print("best:", best2, flush=True)


if __name__ == "__main__":
    main()
