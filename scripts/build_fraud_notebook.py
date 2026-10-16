"""Genera notebooks/03_fraude_ff.ipynb a partir de celdas definidas aquí.

Usa nbformat (mismo patrón que build_notebook.py). El cuaderno se ejecuta en
platypy (lab-pytorch, RTX 3070) y se copia de vuelta con salidas.
"""

import nbformat as nbf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "03_fraude_ff.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

# ---------------------------------------------------------------------------
cells.append(md(
"""# Red feed-forward para detección de fraude: hiperparámetros vs parámetros

En este cuaderno entrenamos una red neuronal feed-forward (FF) para **detectar fraude en tarjetas de crédito** con una **tasa de falsos positivos (FPR) muy baja**.

Los datos son transacciones sintéticas de IBM (SD254 v2), ya transformadas a un vector de 43 features numéricas por el script `scripts/build_fraud_dataset.py`. El objetivo `is_fraud` es 1 si la transacción es fraudulenta y 0 si no.

## La idea central

Una red neuronal tiene **dos tipos de "números"** que es fácil confundir:

- **Parámetros** ($\\theta$): los pesos $W$ y los sesgos $b$. Los **aprende el optimizador** (descenso de gradiente + backpropagation, como vimos en el notebook `02_mlp_backprop`). Sus valores cambian en cada paso del entrenamiento.
- **Hiperparámetros**: la tasa de aprendizaje, el número de capas, las neuronas por capa, el tamaño de lote, las épocas, la activación, la semilla. Los **fija el humano antes de entrenar** y el optimizador **no los toca**.

A lo largo del cuaderno distinguimos ambos y mostramos que **buscar buenos hiperparámetros es otra tarea**, distinta de entrenar: cada combinación de hiperparámetros exige reentrenar la red desde cero, y el espacio de combinaciones **crece de forma combinatoria**."""
))

# ---------------------------------------------------------------------------
cells.append(code(
"""import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve, average_precision_score)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


def find_repo_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / "data" / "processed" / "train.csv").exists():
            return parent
    for c in ["/workspace/friendly-fiesta", "/workspace"]:
        if (Path(c) / "data" / "processed" / "train.csv").exists():
            return Path(c)
    return p


ROOT = find_repo_root()
print("ROOT =", ROOT)"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Datos: desbalance y sobremuestreo

El fraude es raro: en los datos completos la tasa real es **0.12 %**. Para que la red aprenda la clase minoritaria, el **conjunto de entrenamiento** está sobremuestreado a un **10 % de fraude** (10 000 fraudes + 90 000 legítimas). El **conjunto de test conserva la tasa real** (0.12 %), de modo que el FPR que reportamos sea el honesto.

- `fraud_train.csv` → ~100 000 filas, fraude 10 % (para aprender).
- `fraud_test.csv` → ~300 000 filas, fraude 0.12 % (para medir el FPR real)."""
))

cells.append(code(
"""train = pd.read_csv(ROOT / "data" / "processed" / "fraud_train.csv")
test = pd.read_csv(ROOT / "data" / "processed" / "fraud_test.csv")

FEATURES = [c for c in train.columns if c not in ("id", "is_fraud")]
print("features:", len(FEATURES))
print("train:", train.shape, "| fraude:", train["is_fraud"].sum(),
      f"({train['is_fraud'].mean():.2%})")
print("test :", test.shape, "| fraude:", test["is_fraud"].sum(),
      f"({test['is_fraud'].mean():.3%})")"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Preparación: split estratificado y escalado

Separamos una parte del entrenamiento como validación (para elegir hiperparámetros y vigilar el sobreajuste) y estandarizamos las features a media 0 y desviación 1. El escalador se ajusta **solo con el entrenamiento** para no filtrar información."""
))

cells.append(code(
"""train_df, val_df = train_test_split(
    train, test_size=0.2, stratify=train["is_fraud"], random_state=SEED
)

scaler = StandardScaler().fit(train_df[FEATURES])
X_train = scaler.transform(train_df[FEATURES]).astype(np.float32)
X_val = scaler.transform(val_df[FEATURES]).astype(np.float32)
X_test = scaler.transform(test[FEATURES]).astype(np.float32)

y_train = train_df["is_fraud"].to_numpy().astype(np.float32)
y_val = val_df["is_fraud"].to_numpy().astype(np.float32)
y_test = test["is_fraud"].to_numpy().astype(np.float32)

print("X_train", X_train.shape, " X_val", X_val.shape, " X_test", X_test.shape)"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Parámetros vs hiperparámetros (la distinción)

Antes de entrenar, dejamos clara la diferencia.

**Parámetros** son los valores que aprende la red: pesos $W$ y sesgos $b$. Su **cantidad** la determina la arquitectura, pero su **valor** lo fija el optimizador al minimizar la pérdida.

**Hiperparámetros** son las decisiones que tomamos nosotros: tasa de aprendizaje, ancho (neuronas por capa), profundidad (número de capas), tamaño de lote, épocas, activación y semilla. No se actualizan durante el entrenamiento.

La cantidad de parámetros de una capa lineal de $n_{in}$ entradas a $n_{out}$ salidas es:

$$n_{in} \\cdot n_{out} + n_{out}$$

Fíjate: **los hiperparámetros (ancho, profundidad) deciden cuántos parámetros hay**, pero no sus valores."""
))

cells.append(code(
"""n_in = len(FEATURES)


def build_mlp(n_in, width, depth, seed=SEED):
    torch.manual_seed(seed)
    layers = [nn.Linear(n_in, width), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# el número de parámetros depende de los hiperparámetros
for width, depth in [(32, 2), (64, 2), (128, 3)]:
    m = build_mlp(n_in, width, depth)
    print(f"ancho={width:3d} profundidad={depth}: {count_params(m):>8,} parámetros")

baseline = build_mlp(n_in, width=64, depth=2)
print("\\nbaseline (ancho=64, profundidad=2):",
      f"{count_params(baseline):,} parámetros")"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Entrenamiento de una red (los parámetros cambian, los hiperparámetros no)

Definimos el bucle de entrenamiento. La pérdida es `BCEWithLogitsLoss` (entropía cruzada binaria; la salida es un logit) con un **peso de clase** que compensa el desbalance. El optimizador actualiza solo los parámetros: los hiperparámetros (`lr`, `epochs`, `batch`, `width`, `depth`) permanecen congelados durante todo el bucle."""
))

cells.append(code(
"""def train_model(model, X, y, Xv, yv, lr=1e-3, epochs=30, batch=1024,
                pos_weight=None, verbose=True):
    model = model.to(DEVICE)
    Xt = torch.tensor(X, device=DEVICE)
    yt = torch.tensor(y, device=DEVICE)
    Xvt = torch.tensor(Xv, device=DEVICE)
    yvt = torch.tensor(yv, device=DEVICE)

    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    pw = torch.tensor([pos_weight], device=DEVICE) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"epoch": [], "loss": [], "val_auc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xvt).squeeze(1)).cpu().numpy()
        hist["epoch"].append(epoch)
        hist["loss"].append(tot / len(ds))
        hist["val_auc"].append(roc_auc_score(yv, pv))
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"época {epoch:2d}/{epochs}: loss={hist['loss'][-1]:.4f} "
                  f"val AUC={hist['val_auc'][-1]:.4f}")
    return model, hist


# peso de clase: negativos / positivos (compensa el desbalance)
pos_weight = (1.0 - y_train.mean()) / y_train.mean()
print(f"pos_weight = {pos_weight:.2f} (tasa de fraude en train: {y_train.mean():.2%})")"""
))

cells.append(md(
"""### Los parámetros cambian con el entrenamiento

Congelamos los pesos iniciales, entrenamos y comprobamos que **solo los parámetros se movieron**. Los hiperparámetros (constantes del código) siguen siendo exactamente los mismos."""
))

cells.append(code(
"""# pesos iniciales (antes de entrenar), copiados a CPU para compararlos después
init = {k: v.detach().cpu().clone() for k, v in baseline.named_parameters()}
w_before = init["0.weight"]

model, hist = train_model(
    baseline, X_train, y_train, X_val, y_val,
    lr=1e-3, epochs=30, batch=1024, pos_weight=pos_weight, verbose=True,
)

w_after = dict(model.named_parameters())["0.weight"].detach().cpu()
moved = sum(
    1 for k in init
    if not torch.equal(init[k], dict(model.named_parameters())[k].detach().cpu())
)

print(f"\\npesos/sesgos que cambiaron: {moved}/{len(init)}")
print(f"norma de W[0]: antes {w_before.norm().item():.4f} "
      f"→ después {w_after.norm().item():.4f}")
print("hiperparámetros usados: lr=1e-3, epochs=30, batch=1024, width=64, depth=2 "
      "(no cambiaron)")"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Curva de aprendizaje

Seguimos la pérdida de entrenamiento y el AUC de validación por época. El AUC mide la capacidad de separar fraude de no-fraude sin depender del umbral (1.0 es perfecto, 0.5 es azar)."""
))

cells.append(code(
"""import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["loss"], mode="lines",
                         name="loss train", line=dict(color="#3B82F6", width=2),
                         yaxis="y1"))
fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_auc"], mode="lines",
                         name="AUC validación", line=dict(color="#F97316", width=2),
                         yaxis="y2"))
fig.update_layout(
    title="Pérdida (train) y AUC (validación) por época",
    xaxis_title="época",
    yaxis=dict(title=dict(text="loss", font=dict(color="#3B82F6"))),
    yaxis2=dict(title=dict(text="AUC", font=dict(color="#F97316")),
                overlaying="y", side="right"),
    width=720, height=420)
fig.show()"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Evaluación: la matriz de confusión, precisión, recall y FPR

La red devuelve una probabilidad $\\hat{p} = \\sigma(z)$. Para decidir "fraude" hay que **elegir un umbral**: si $\\hat{p} \\ge$ umbral, marcamos fraude. Cada umbral produce una matriz de confusión:

|  | Predice **no fraude** | Predice **fraude** |
|---|---|---|
| Real: **no fraude** | TN (aciertos) | **FP (falsos positivos)** |
| Real: **fraude** | FN (falsos negativos) | TP (aciertos) |

De ahí salen tres números clave:

- **FPR** (tasa de falsos positivos) $= \\frac{FP}{FP + TN}$: qué fracción de las transacciones **legítimas** marcamos mal. Queremos que sea **bajísima** para no molestar a clientes buenos.
- **Precisión** $= \\frac{TP}{TP + FP}$: de las que marcamos como fraude, cuántas lo son de verdad.
- **Recall** $= \\frac{TP}{TP + FN}$: qué fracción del fraude real capturamos.

Con fraude tan raro (0.12 %), aunque el FPR sea bajísimo la precisión es baja: al marcar pocas operaciones, la mayoría de las marcadas siguen siendo legítimas. Es el precio de un umbral conservador."""
))

cells.append(code(
"""with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor(X_test, device=DEVICE))
                          .squeeze(1)).cpu().numpy()

roc_auc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)
print(f"ROC-AUC test: {roc_auc:.4f}")
print(f"PR-AUC  test: {pr_auc:.4f}  (baseline azar = {y_test.mean():.4f})")"""
))

cells.append(md(
"""### Elegir el umbral: fijamos un piso de recall y minimizamos el FPR

No hay un umbral "correcto": es un **trade-off** entre capturar fraude (recall) y no molestar legítimos (FPR). Nuestra regla de decisión es:

> **Capturar al menos el 50 % del fraude (recall ≥ 0.5) con el menor FPR posible.**

Es decir, tomamos el **umbral más alto** que aún cumpla recall ≥ 0.5. Así minimizamos los falsos positivos sin dejar pasar más de la mitad del fraude."""
))

cells.append(code(
"""def operating_points(y_true, probs, targets=(0.001, 0.005, 0.01)):
    fpr, tpr, th = roc_curve(y_true, probs)
    rows = []
    for t in targets:
        idx = np.where(fpr <= t)[0]
        if len(idx) == 0:
            continue
        i = idx[-1]
        thr = th[i] if i < len(th) else th[-1]
        pred = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "FPR objetivo": f"{t:.2%}",
            "umbral": round(float(thr), 3),
            "FPR real": round(fp / (fp + tn), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        })
    return pd.DataFrame(rows)


print("Tabla de puntos de operación (para contexto):")
print(operating_points(y_test, probs).to_string(index=False))


# regla: umbral más alto con recall >= 0.5 (menor FPR)
RECALL_MIN = 0.5
fpr, tpr, th = roc_curve(y_test, probs)
i_umbral = int(np.where(tpr >= RECALL_MIN)[0][0])
THR = float(th[i_umbral])
print(f"\\nUmbral elegido: {THR:.3f}  (recall >= {RECALL_MIN:.0%} con el menor FPR)")"""
))

cells.append(md(
"""### Matriz de confusión del primer modelo

Aplicamos el umbral elegido a las predicciones del modelo sobre el conjunto de test."""
))

cells.append(code(
"""def show_confusion(y_true, probs, thr, title):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"{title}  (umbral = {thr:.3f})")
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr_r = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"  precision={prec:.3f}  recall={rec:.3f}  FPR={fpr_r:.4%}")

    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["No fraude", "Fraude"], y=["No fraude", "Fraude"],
        text=[["TN", "FP"], ["FN", "TP"]],
        texttemplate="%{text}<br>%{z}",
        colorscale=[[0.0, "#E8F1FB"], [1.0, "#1E3A8A"]],
        showscale=False))
    fig.update_layout(title=title, width=420, height=360)
    fig.show()
    return {"precision": prec, "recall": rec, "fpr": fpr_r,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


m1 = show_confusion(y_test, probs, THR, "Matriz de confusión: modelo baseline")"""
))

cells.append(md(
"""### Barrido de umbral: curva precision-recall y FPR vs recall

A la izquierda, la curva precision-recall (la línea punteada es el azar, igual a la tasa de fraude). A la derecha, cómo crece el FPR a medida que bajamos el umbral para capturar más fraude. El punto marcado es el umbral que elegimos."""
))

cells.append(code(
"""precision, recall, _ = precision_recall_curve(y_test, probs)
fpr, tpr, _ = roc_curve(y_test, probs)

fig = go.Figure()
fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                         name="precision-recall", line=dict(color="#3B82F6", width=2)))
fig.add_trace(go.Scatter(x=[0, 1], y=[y_test.mean(), y_test.mean()],
                         mode="lines", name="azar",
                         line=dict(color="#F97316", width=2, dash="dash")))
fig.add_trace(go.Scatter(x=[m1["recall"]], y=[m1["precision"]],
                         mode="markers", name="umbral elegido",
                         marker=dict(color="#EF4444", size=11)))
fig.update_layout(title="Curva precision-recall (test)",
                  xaxis_title="recall", yaxis_title="precision",
                  width=560, height=420)
fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=recall, y=fpr, mode="lines",
                          name="FPR vs recall", line=dict(color="#22C55E", width=2)))
fig2.add_trace(go.Scatter(x=[m1["recall"]], y=[m1["fpr"]],
                          mode="markers", name="umbral elegido",
                          marker=dict(color="#EF4444", size=11)))
fig2.update_layout(title="FPR frente a recall (test)",
                   xaxis_title="recall", yaxis_title="FPR",
                   width=560, height=420)
fig2.show()"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Búsqueda de hiperparámetros: el espacio crece de forma combinatoria

Hasta aquí entrenamos **una** red con hiperparámetros fijos. Pero ¿cómo elegir `lr`, `ancho` y `profundidad`? Hay que **buscar**: probar combinaciones y quedarse con la que mejor valide.

El problema es que el espacio es un **producto cartesiano**: con 3 tasas de aprendizaje, 3 anchos y 2 profundidades hay

$$3 \\times 3 \\times 2 = 18 \\ \\text{combinaciones}$$

y **cada una exige reentrenar la red desde cero** (los parámetros de una config no sirven para otra). Si además barriéramos la activación (2 opciones), el lote (3) y la semilla (5):

$$18 \\times 2 \\times 3 \\times 5 = 540 \\ \\text{entrenamientos}$$

El coste crece **multiplicativamente** con cada eje nuevo: es la explosión combinatoria. Por eso en vivo solo **muestreamos unas pocas combinaciones**. Esta búsqueda **no aprende hiperparámetros**: somos nosotros quienes exploramos el espacio; la red solo aprende parámetros dentro de cada config."""
))

cells.append(code(
"""grid = {"lr": [1e-3, 3e-3, 1e-2],
        "width": [32, 64, 128],
        "depth": [2, 3]}

combos = list(product(grid["lr"], grid["width"], grid["depth"]))
print(f"combinaciones totales: {len(combos)} = "
      f"{len(grid['lr'])}·{len(grid['width'])}·{len(grid['depth'])}")
print("primeras 5:", combos[:5])"""
))

cells.append(code(
"""# muestreamos 6 configuraciones (no las 18) para no eternizar la sesión
rng = np.random.default_rng(SEED)
subset = rng.choice(len(combos), size=6, replace=False)

results = []
for i in subset:
    lr, width, depth = combos[i]
    m = build_mlp(n_in, width, depth)
    _, h = train_model(m, X_train, y_train, X_val, y_val,
                       lr=lr, epochs=15, batch=1024, pos_weight=pos_weight,
                       verbose=False)
    results.append({
        "lr": lr, "width": width, "depth": depth,
        "params": count_params(m),
        "val_auc": round(h["val_auc"][-1], 4),
    })
    print(f"config (lr={lr}, width={width}, depth={depth}) -> "
          f"val AUC {h['val_auc'][-1]:.4f}")

res_df = pd.DataFrame(results).sort_values("val_auc", ascending=False)
print("\\nResultados ordenados por AUC de validación:")
print(res_df.to_string(index=False))

best = res_df.iloc[0]
BEST_HP = {"lr": float(best["lr"]), "width": int(best["width"]),
           "depth": int(best["depth"])}
print(f"\\nmejor config: lr={BEST_HP['lr']}, width={BEST_HP['width']}, "
      f"depth={BEST_HP['depth']} (val AUC {best['val_auc']:.4f})")"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Reentrenamos el mejor modelo y lo evaluamos

La búsqueda usó **validación** para elegir la config ganadora. Ahora reentrenamos esa config con más épocas (30) y la evaluamos en el **test** con el mismo umbral, para medir su rendimiento final sin optimismo."""
))

cells.append(code(
"""best_model = build_mlp(n_in, BEST_HP["width"], BEST_HP["depth"])
best_model, hist_best = train_model(
    best_model, X_train, y_train, X_val, y_val,
    lr=BEST_HP["lr"], epochs=30, batch=1024, pos_weight=pos_weight, verbose=True,
)

with torch.no_grad():
    probs_best = torch.sigmoid(
        best_model(torch.tensor(X_test, device=DEVICE)).squeeze(1)
    ).cpu().numpy()

m2 = show_confusion(y_test, probs_best, THR, "Matriz de confusión: mejor modelo")"""
))

cells.append(md(
"""### Comparación final

Reunimos las métricas de los dos modelos al mismo umbral elegido (`THR`)."""
))

cells.append(code(
"""comparacion = pd.DataFrame([
    {"modelo": "baseline (64×2, lr=1e-3)",
     "params": count_params(model),
     "ROC-AUC": roc_auc_score(y_test, probs),
     "precision": m1["precision"], "recall": m1["recall"], "FPR": m1["fpr"]},
    {"modelo": f"mejor ({BEST_HP['width']}×{BEST_HP['depth']}, lr={BEST_HP['lr']})",
     "params": count_params(best_model),
     "ROC-AUC": roc_auc_score(y_test, probs_best),
     "precision": m2["precision"], "recall": m2["recall"], "FPR": m2["fpr"]},
]).round(4)
print(comparacion.to_string(index=False))"""
))

# ---------------------------------------------------------------------------
cells.append(md(
"""## Resumen

- **Parámetros** (pesos y sesgos): los aprende el optimizador; su número lo fija la arquitectura, su valor lo fija el entrenamiento.
- **Hiperparámetros** (lr, ancho, profundidad, lote, épocas, activación, semilla): los elegimos nosotros y no cambian durante el entrenamiento.
- **Umbral de decisión**: es otro parámetro de operación. Lo fijamos con una regla de negocio (recall ≥ 50 % con el menor FPR posible).
- **Búsqueda de hiperparámetros**: reentrenar por cada combinación del espacio; el coste crece de forma **combinatoria**, por lo que en la práctica se muestrea.
- Para un **FPR bajísimo**, elevamos el umbral de decisión: capturamos menos fraude pero apenas marcamos transacciones legítimas."""))

nb["cells"] = cells
nbf.write(nb, OUT)
print("escrito", OUT)
