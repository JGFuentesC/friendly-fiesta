"""Genera notebooks/02_mlp_backprop.ipynb a partir de celdas definidas aquí.

Usa nbformat para evitar errores de escape con el LaTeX y el código.
"""

import nbformat as nbf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "02_mlp_backprop.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md(
"""# Backpropagation de un perceptrón multicapa (MLP)

En este cuaderno construimos, desde cero y paso a paso, un perceptrón multicapa (MLP) para regresión y derivamos el algoritmo de retropropagación del error (backpropagation).

La tarea es predecir la variable continua $y \\in \\mathbb{R}$ a partir de tres predictores $\\mathbf{x} = (x_1, x_2, x_3)$.

## Arquitectura

$$3 \\to 16 \\to 8 \\to 1$$

Dos capas ocultas (16 y 8 neuronas) con activación tangente hiperbólica y una neurona de salida lineal.

## Qué veremos

1. Explorar los datos con visualizaciones 3D.
2. Derivar el paso hacia delante (forward) en forma matricial.
3. Derivar el backpropagation con la regla de la cadena, capa por capa.
4. Implementar el MLP a mano con NumPy y verificar los gradientes por diferencias finitas.
5. Repetir el entrenamiento con PyTorch y con TensorFlow.
6. Comparar las curvas de pérdida de los tres marcos.
7. Visualizar la superficie que aprende la red, con curvas de nivel proyectadas."""
))

cells.append(code(
"""import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)


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

cells.append(code(
"""train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
val = pd.read_csv(ROOT / "data" / "processed" / "validation.csv")

print("train:", train.shape, " validation:", val.shape)
print(train.describe().round(3))"""
))

cells.append(md(
"""## Escalado de los predictores

Las capas ocultas usan la tangente hiperbólica, que satura para entradas grandes. Para que el gradiente fluya bien, estandarizamos cada predictor a media $0$ y desviación $1$. El escalador se ajusta **solo con el conjunto de entrenamiento** y se aplica a ambos conjuntos, para no filtrar información del conjunto de validación.

$$x'_j = \\frac{x_j - \\mu_j}{\\sigma_j}$$"""
))

cells.append(code(
"""scaler = StandardScaler().fit(train[["x1", "x2", "x3"]])

X_train = scaler.transform(train[["x1", "x2", "x3"]])
X_val = scaler.transform(val[["x1", "x2", "x3"]])

y_train = train["y"].to_numpy().reshape(-1, 1)
y_val = val["y"].to_numpy().reshape(-1, 1)

print("X_train", X_train.shape, " y_train", y_train.shape)
print("media (escalada):", X_train.mean(axis=0).round(3))
print("desv. (escalada):", X_train.std(axis=0).round(3))"""
))

cells.append(md(
"""## ¿Por qué miramos los datos en 3D?

Antes de entrenar conviene ver la relación entre cada pareja de predictores y el objetivo. Como $y$ depende de tres variables, cada gráfico 3D muestra solo dos predictores a la vez (la tercera queda fuera de la vista). Los ejes horizontales son los dos predictores, la altura y el color son el objetivo $y$.

La superficie dibujada es una interpolación suave de los datos (ajuste por spline sobre los puntos observados). Las curvas de nivel proyectadas en el suelo muestran hacia dónde crece o decrece $y$.""" ))

cells.append(code(
"""import plotly.graph_objects as go
from scipy.interpolate import griddata


def save_fig(fig, name):
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(ROOT / "figures" / name), scale=2)
        print("guardado figures/" + name)
    except Exception:
        pass


def pair_figure(df, cols, grid_n=45, title=""):
    \"\"\"Dispersión 3D de y sobre dos predictores, con superficie interpolada y contornos.\"\"\"
    x = df[cols[0]].to_numpy()
    y = df[cols[1]].to_numpy()
    z = df["y"].to_numpy()

    xi = np.linspace(x.min(), x.max(), grid_n)
    yi = np.linspace(y.min(), y.max(), grid_n)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(np.column_stack([x, y]), z, (Xi, Yi), method="linear")

    zfloor = z.min() - 0.15 * (z.max() - z.min())

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=2.5, color=z, colorscale="Viridis", opacity=0.85),
        name="datos",
    ))

    fig.add_trace(go.Surface(
        x=xi, y=yi, z=Zi, colorscale="Viridis", opacity=0.55,
        contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=False))),
        name="superficie", showscale=False,
    ))

    fig.add_trace(go.Surface(
        x=xi, y=yi, z=np.full_like(Zi, zfloor), surfacecolor=Zi,
        colorscale="Viridis", showscale=False, opacity=0.9,
        contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=False))),
        name="curvas de nivel",
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=cols[0], yaxis_title=cols[1], zaxis_title="y",
        ),
        width=620, height=560,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


for cols in [("x1", "x2"), ("x1", "x3"), ("x2", "x3")]:
    fig = pair_figure(train, cols, title=f"y en función de ({cols[0]}, {cols[1]})")
    fig.show()
    save_fig(fig, f"datos_3d_{cols[0]}_{cols[1]}.png")"""
))

cells.append(md(
"""**Interpretación:** en las tres parejas se observa que $y$ crece de forma marcada con $x_1$ (los puntos de la derecha son más altos) y decrece con $x_3$. La variable $x_2$ tiene un efecto más débil. La superficie interpolada y las curvas de nivel muestran una pendiente dominante a lo largo de $x_1$ y, en menor medida, un descenso con $x_3$. Esto sugiere que una red con capas no lineales podrá ajustar esta superficie, y que $x_1$ será el predictor con mayor peso."""
))

cells.append(md(
"""## Paso hacia delante (forward pass)

Fijamos la notación. Para una muestra $\\mathbf{x} \\in \\mathbb{R}^{3}$ (fila):

**Capa 1** (oculta, 16 neuronas, tanh):

$$\\mathbf{z}^{(1)} = \\mathbf{x} W^{(1)} + \\mathbf{b}^{(1)}, \\qquad \\mathbf{a}^{(1)} = \\tanh\\!(\\mathbf{z}^{(1)})$$

con $W^{(1)} \\in \\mathbb{R}^{3 \\times 16}$ y $\\mathbf{b}^{(1)} \\in \\mathbb{R}^{16}$.

**Capa 2** (oculta, 8 neuronas, tanh):

$$\\mathbf{z}^{(2)} = \\mathbf{a}^{(1)} W^{(2)} + \\mathbf{b}^{(2)}, \\qquad \\mathbf{a}^{(2)} = \\tanh\\!(\\mathbf{z}^{(2)})$$

con $W^{(2)} \\in \\mathbb{R}^{16 \\times 8}$ y $\\mathbf{b}^{(2)} \\in \\mathbb{R}^{8}$.

**Capa de salida** (1 neurona, activación lineal):

$$\\hat{y} = \\mathbf{z}^{(3)} = \\mathbf{a}^{(2)} W^{(3)} + b^{(3)}$$

con $W^{(3)} \\in \\mathbb{R}^{8 \\times 1}$ y $b^{(3)} \\in \\mathbb{R}$.

**Función de pérdida** (error cuadrático medio sobre un lote de $m$ muestras):

$$\\mathcal{L} = \\frac{1}{m} \\sum_{i=1}^{m} \\left( \\hat{y}_i - y_i \\right)^2$$"""
))

cells.append(md(
"""## Backpropagation: regla de la cadena capa por capa

Derivamos hacia atrás desde la salida. Escribimos $\\delta^{(\\ell)}$ para el gradiente de la pérdida respecto a $\\mathbf{z}^{(\\ell)}$.

**Salida.** La derivada del error cuadrático es directa:

$$\\delta^{(3)} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}^{(3)}} = \\frac{2}{m}\\left( \\hat{\\mathbf{y}} - \\mathbf{y} \\right)$$

de donde salen los gradientes de la última capa:

$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(3)}} = \\left(\\mathbf{a}^{(2)}\\right)^\\top \\delta^{(3)}, \\qquad
\\frac{\\partial \\mathcal{L}}{\\partial b^{(3)}} = \\sum_{i} \\delta^{(3)}_i$$

**Capa 2.** El error se propaga con la derivada de tanh:

$$\\tanh'(z) = 1 - \\tanh^2(z) = 1 - \\left(a^{(2)}\\right)^2$$

$$\\delta^{(2)} = \\delta^{(3)} \\left(W^{(3)}\\right)^\\top \\odot \\left( 1 - \\left(\\mathbf{a}^{(2)}\\right)^2 \\right)$$

$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(2)}} = \\left(\\mathbf{a}^{(1)}\\right)^\\top \\delta^{(2)}, \\qquad
\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}^{(2)}} = \\sum_{i} \\delta^{(2)}_i$$

**Capa 1.** El mismo patrón:

$$\\delta^{(1)} = \\delta^{(2)} \\left(W^{(2)}\\right)^\\top \\odot \\left( 1 - \\left(\\mathbf{a}^{(1)}\\right)^2 \\right)$$

$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(1)}} = \\mathbf{x}^\\top \\delta^{(1)}, \\qquad
\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}^{(1)}} = \\sum_{i} \\delta^{(1)}_i$$

El símbolo $\\odot$ es el producto elemento a elemento (Hadamard). Observa la estructura recurrente: cada $\\delta$ se obtiene multiplicando el anterior por la transpuesta de la matriz de pesos de la capa y escalando por la derivada de la activación."""
))

cells.append(code(
"""class MLP:
    \"\"\"MLP de dos capas ocultas (tanh) y salida lineal, con backprop a mano.\"\"\"

    def __init__(self, sizes=(3, 16, 8, 1), seed=SEED):
        self.sizes = sizes
        self.rng = np.random.default_rng(seed)
        self.params = {}
        self._init_params()

    def _init_params(self):
        for l in range(1, len(self.sizes)):
            fan_in, fan_out = self.sizes[l - 1], self.sizes[l]
            bound = np.sqrt(6.0 / (fan_in + fan_out))  # inicialización de Xavier (uniforme)
            self.params[f"W{l}"] = self.rng.uniform(-bound, bound, (fan_in, fan_out))
            self.params[f"b{l}"] = np.zeros((1, fan_out))

    def forward(self, X):
        cache = {"a0": X}
        A = X
        L = len(self.sizes) - 1
        for l in range(1, L):
            Z = A @ self.params[f"W{l}"] + self.params[f"b{l}"]
            A = np.tanh(Z)
            cache[f"z{l}"] = Z
            cache[f"a{l}"] = A
        Z = A @ self.params[f"W{L}"] + self.params[f"b{L}"]
        cache[f"z{L}"] = Z
        cache[f"a{L}"] = Z  # salida lineal
        return cache

    def predict(self, X):
        L = len(self.sizes) - 1
        return self.forward(X)[f"a{L}"]

    def backward(self, cache, y):
        m = y.shape[0]
        L = len(self.sizes) - 1
        grads = {}
        delta = 2.0 * (cache[f"a{L}"] - y) / m  # dL/dzL
        grads[f"W{L}"] = cache[f"a{L - 1}"].T @ delta
        grads[f"b{L}"] = delta.sum(axis=0, keepdims=True)
        for l in range(L - 1, 0, -1):
            delta = (delta @ self.params[f"W{l + 1}"].T) * (1.0 - cache[f"a{l}"] ** 2)
            grads[f"W{l}"] = cache[f"a{l - 1}"].T @ delta
            grads[f"b{l}"] = delta.sum(axis=0, keepdims=True)
        return grads

    def step(self, grads, lr):
        for k in self.params:
            self.params[k] -= lr * grads[k]


def mse(yhat, y):
    return float(np.mean((yhat - y) ** 2))


model = MLP()
print("Número de parámetros:", sum(v.size for v in model.params.values()))
for k, v in model.params.items():
    print(f"{k}: {v.shape}")"""
))

cells.append(md(
"""## ¿Por qué verificamos el gradiente con diferencias finitas?

El backpropagation es propenso a errores de signo o de dimensiones. Antes de entrenar, comparamos el gradiente analítico con una aproximación numérica por diferencias centrales:

$$\\frac{\\partial \\mathcal{L}}{\\partial \\theta} \\approx \\frac{\\mathcal{L}(\\theta + \\epsilon) - \\mathcal{L}(\\theta - \\epsilon)}{2\\epsilon}$$

Si ambas coinciden hasta un error del orden de $10^{-5}$ o menor, la implementación es correcta."""
))

cells.append(code(
"""def numerical_gradient(model, X, y, eps=1e-6):
    \"\"\"Gradiente por diferencias centrales para todos los parámetros.\"\"\"
    grads = {k: np.zeros_like(v) for k, v in model.params.items()}
    for k, v in model.params.items():
        it = np.nditer(v, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = v[idx]
            v[idx] = orig + eps
            Lp = mse(model.predict(X), y)
            v[idx] = orig - eps
            Lm = mse(model.predict(X), y)
            v[idx] = orig
            grads[k][idx] = (Lp - Lm) / (2 * eps)
            it.iternext()
    return grads


# Una sola muestra para que la comprobación sea barata
x0 = X_train[:1]
y0 = y_train[:1]
cache = model.forward(x0)
ana = model.backward(cache, y0)
num = numerical_gradient(model, x0, y0)

print(f"{'parámetro':>6} {'analítico':>16} {'numérico':>16} {'dif. abs':>12}")
for k in model.params:
    d = np.abs(ana[k] - num[k]).max()
    print(f"{k:>6} {ana[k].ravel()[0]:>16.8f} {num[k].ravel()[0]:>16.8f} {d:>12.2e}")"""
))

cells.append(code(
"""def train_numpy(model, X, y, Xv, yv, lr=0.1, epochs=300, batch=32):
    n = X.shape[0]
    history = {"epoch": [], "train": [], "val": []}
    for epoch in range(epochs):
        order = model.rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            Xb, yb = X[idx], y[idx]
            cache = model.forward(Xb)
            grads = model.backward(cache, yb)
            model.step(grads, lr)
        history["epoch"].append(epoch + 1)
        history["train"].append(mse(model.predict(X), y))
        history["val"].append(mse(model.predict(Xv), yv))
    return history


model = MLP()
hist_np = train_numpy(model, X_train, y_train, X_val, y_val)

print(f"train MSE final : {hist_np['train'][-1]:.5f}")
print(f"val   MSE final : {hist_np['val'][-1]:.5f}")"""
))

cells.append(md(
"""## Curva de pérdida (NumPy)

Seguimos el error cuadrático medio de entrenamiento y de validación en cada época. Una buena señal es que ambas curvas bajen juntas y se estabilicen sin separarse. Si la curva de validación sube mientras la de entrenamiento baja, hay sobreajuste."""
))

cells.append(code(
"""import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_np["epoch"], y=hist_np["train"], mode="lines",
    name="train", line=dict(color="#3B82F6", width=2)))
fig.add_trace(go.Scatter(
    x=hist_np["epoch"], y=hist_np["val"], mode="lines",
    name="validación", line=dict(color="#F97316", width=2)))
fig.update_yaxes(type="log")
fig.update_layout(
    title="Pérdida (MSE) por época, MLP en NumPy",
    xaxis_title="época", yaxis_title="MSE (escala log)",
    width=700, height=420)
fig.show()
save_fig(fig, "loss_numpy.png")"""
))

cells.append(md(
"""**Interpretación:** ambas curvas descienden de forma suave y se estabilizan cerca del mismo valor, lo que indica que la red aprende sin sobreajustar. La escala logarítmica del eje vertical permite ver la convergencia en las últimas épocas, donde la mejora es pequeña en valor absoluto."""
))

cells.append(md(
"""## PyTorch: autograd calcula el mismo gradiente

PyTorch construye el grafo de cómputo de forma automática y `backward()` aplica la regla de la cadena por nosotros. La arquitectura es idéntica a la de NumPy."""
))

cells.append(code(
"""import torch
import torch.nn as nn


def train_pytorch(X, y, Xv, yv, lr=0.1, epochs=300, batch=32, seed=SEED):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    Xvt = torch.tensor(Xv, dtype=torch.float32)
    yvt = torch.tensor(yv, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(3, 16), nn.Tanh(),
        nn.Linear(16, 8), nn.Tanh(),
        nn.Linear(8, 1),
    )
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    history = {"epoch": [], "train": [], "val": []}
    n = Xt.shape[0]
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            history["train"].append(float(loss_fn(model(Xt), yt)))
            history["val"].append(float(loss_fn(model(Xvt), yvt)))
        history["epoch"].append(epoch + 1)
    return model, history


torch_model, hist_pt = train_pytorch(X_train, y_train, X_val, y_val)
print("GPU disponible:", torch.cuda.is_available())
print(f"train MSE final : {hist_pt['train'][-1]:.5f}")
print(f"val   MSE final : {hist_pt['val'][-1]:.5f}")"""
))

cells.append(md(
"""## TensorFlow: GradientTape registra el grafo

En TensorFlow, `tf.GradientTape` graba las operaciones y luego `gradient()` calcula los gradientes con diferenciación automática, aplicando la misma regla de la cadena que derivamos a mano."""
))

cells.append(code(
"""import tensorflow as tf

tf.random.set_seed(SEED)


def train_tensorflow(X, y, Xv, yv, lr=0.1, epochs=300, batch=32, seed=SEED):
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    Xv = Xv.astype(np.float32)
    yv = yv.astype(np.float32)
    tf.random.set_seed(seed)

    W1 = tf.Variable(tf.random.uniform((3, 16), -0.5, 0.5, dtype=tf.float32))
    b1 = tf.Variable(tf.zeros((1, 16), dtype=tf.float32))
    W2 = tf.Variable(tf.random.uniform((16, 8), -0.5, 0.5, dtype=tf.float32))
    b2 = tf.Variable(tf.zeros((1, 8), dtype=tf.float32))
    W3 = tf.Variable(tf.random.uniform((8, 1), -0.5, 0.5, dtype=tf.float32))
    b3 = tf.Variable(tf.zeros((1, 1), dtype=tf.float32))
    variables = [W1, b1, W2, b2, W3, b3]

    def forward(x):
        a1 = tf.tanh(x @ W1 + b1)
        a2 = tf.tanh(a1 @ W2 + b2)
        return a2 @ W3 + b3

    def loss_fn(x, y):
        return tf.reduce_mean((forward(x) - y) ** 2)

    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    history = {"epoch": [], "train": [], "val": []}
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb = tf.constant(X[idx])
            yb = tf.constant(y[idx])
            with tf.GradientTape() as tape:
                loss = loss_fn(xb, yb)
            grads = tape.gradient(loss, variables)
            opt.apply_gradients(zip(grads, variables))
        history["train"].append(float(loss_fn(tf.constant(X), tf.constant(y))))
        history["val"].append(float(loss_fn(tf.constant(Xv), tf.constant(yv))))
        history["epoch"].append(epoch + 1)
    return forward, history


tf_forward, hist_tf = train_tensorflow(X_train, y_train, X_val, y_val)
print("TensorFlow:", tf.__version__)
print(f"train MSE final : {hist_tf['train'][-1]:.5f}")
print(f"val   MSE final : {hist_tf['val'][-1]:.5f}")"""
))

cells.append(md(
"""## Comparación de los tres marcos

Los tres entrenan la misma arquitectura con el mismo optimizador (SGD) y la misma tasa de aprendizaje. Las curvas deberían quedar muy cerca, porque todas aplican la misma regla de la cadena."""
))

cells.append(code(
"""fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_np["epoch"], y=hist_np["val"], mode="lines",
                         name="NumPy", line=dict(color="#EF4444", width=2)))
fig.add_trace(go.Scatter(x=hist_pt["epoch"], y=hist_pt["val"], mode="lines",
                         name="PyTorch", line=dict(color="#3B82F6", width=2)))
fig.add_trace(go.Scatter(x=hist_tf["epoch"], y=hist_tf["val"], mode="lines",
                         name="TensorFlow", line=dict(color="#22C55E", width=2)))
fig.update_yaxes(type="log")
fig.update_layout(
    title="Pérdida de validación (MSE) por marco",
    xaxis_title="época", yaxis_title="MSE (escala log)",
    width=700, height=420)
fig.show()
save_fig(fig, "loss_comparacion.png")

resumen = pd.DataFrame({
    "marco": ["NumPy", "PyTorch", "TensorFlow"],
    "train MSE": [hist_np["train"][-1], hist_pt["train"][-1], hist_tf["train"][-1]],
    "val MSE": [hist_np["val"][-1], hist_pt["val"][-1], hist_tf["val"][-1]],
}).round(5)
print(resumen.to_string(index=False))"""
))

cells.append(md(
"""## ¿Qué superficie aprendió la red?

Para ver el resultado del aprendizaje, evaluamos la red sobre una malla de cada pareja de predictores, manteniendo el tercero fijo en su media. La altura y el color son la predicción $\\hat{y}$, y las curvas de nivel proyectadas muestran la forma de la función aprendida."""
))

cells.append(code(
"""def model_surface_pair(cols, grid_n=45, title=""):
    \"\"\"Predicción del MLP (NumPy) sobre una malla de dos predictores, tercero en su media.\"\"\"
    i, j = {"x1": 0, "x2": 1, "x3": 2}[cols[0]], {"x1": 0, "x2": 1, "x3": 2}[cols[1]]
    rest = [k for k in range(3) if k not in (i, j)][0]

    raw = train[[cols[0], cols[1]]].to_numpy()
    xi = np.linspace(raw[:, 0].min(), raw[:, 0].max(), grid_n)
    yi = np.linspace(raw[:, 1].min(), raw[:, 1].max(), grid_n)
    Xi, Yi = np.meshgrid(xi, yi)

    grid = np.zeros((grid_n * grid_n, 3))
    grid[:, i] = Xi.ravel()
    grid[:, j] = Yi.ravel()
    grid[:, rest] = train[["x1", "x2", "x3"]].to_numpy()[:, rest].mean()

    Zp = model.predict(scaler.transform(grid)).reshape(Xi.shape)
    zfloor = y_train.min() - 0.15 * (y_train.max() - y_train.min())

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=train[cols[0]], y=train[cols[1]], z=y_train.ravel(), mode="markers",
        marker=dict(size=2, color=y_train.ravel(), colorscale="Viridis", opacity=0.6),
        name="train", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=val[cols[0]], y=val[cols[1]], z=y_val.ravel(), mode="markers",
        marker=dict(size=2, color="white", opacity=0.5, line=dict(color="black", width=0.5)),
        name="validación", showlegend=False))
    fig.add_trace(go.Surface(
        x=xi, y=yi, z=Zp, colorscale="Viridis", opacity=0.9,
        contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=False))),
        name="superficie aprendida", showscale=False))
    fig.add_trace(go.Surface(
        x=xi, y=yi, z=np.full_like(Zp, zfloor), surfacecolor=Zp,
        colorscale="Viridis", showscale=False,
        contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=False))),
        name="curvas de nivel"))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=cols[0], yaxis_title=cols[1], zaxis_title="y"),
        width=640, height=560, margin=dict(l=0, r=0, t=40, b=0))
    return fig


for cols in [("x1", "x2"), ("x1", "x3"), ("x2", "x3")]:
    fig = model_surface_pair(cols, title=f"Superficie aprendida: ({cols[0]}, {cols[1]})")
    fig.show()
    save_fig(fig, f"superficie_3d_{cols[0]}_{cols[1]}.png")"""
))

cells.append(md(
"""**Interpretación:** la superficie aprendida reproduce la tendencia observada en los datos: crece con $x_1$ y desciende con $x_3$. Las curvas de nivel proyectadas son casi rectas y paralelas, lo que refleja que la relación dominante es lineal en cada pareja, con una curvatura suave que aportan las capas ocultas. Los puntos de validación (blancos) caen sobre la superficie, señal de que la red generaliza bien a datos que no vio en el entrenamiento."""
))

cells.append(md(
"""## Lo que queda de la sesión

Derivamos el forward y el backpropagation de un MLP de dos capas ocultas con la regla de la cadena, lo verificamos por diferencias finitas y entrenamos la misma arquitectura con NumPy, PyTorch y TensorFlow. Los tres llegan a una pérdida de validación similar, porque los tres calculan el mismo gradiente. La visualización de la superficie aprendida muestra de forma concreta la función que la red ajustó sobre los datos."""
))

cells.append(md(
"""## ¿Qué tan bien predice la red?

Una forma directa de juzgar la calidad del ajuste es comparar el valor observado $y$ con la predicción $\\hat{y}$ de la red. Si el ajuste es bueno, los puntos caen cerca de la diagonal $y = \\hat{y}$. Tomamos una muestra aleatoria del conjunto de validación, que la red no usó para entrenar.

La raíz del error cuadrático medio (RMSE) resume el error en las unidades originales de $y$, y $R^2$ indica qué fracción de la varianza de $y$ explica el modelo. Un $R^2$ cercano a $1$ indica un ajuste casi perfecto."""
))

cells.append(code(
'''rng_plot = np.random.default_rng(SEED)
n_plot = 200
idx = rng_plot.choice(len(X_val), size=min(n_plot, len(X_val)), replace=False)

y_true = y_val[idx]
y_hat = model.predict(X_val[idx])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=y_true.ravel(), y=y_hat.ravel(), mode="markers",
    marker=dict(size=6, color="#3B82F6", opacity=0.6),
    name="validación"))
lo = float(min(y_true.min(), y_hat.min()))
hi = float(max(y_true.max(), y_hat.max()))
fig.add_trace(go.Scatter(
    x=[lo, hi], y=[lo, hi], mode="lines",
    line=dict(color="#F97316", width=2, dash="dash"),
    name="y = ŷ"))
fig.update_layout(
    title="y observada vs y estimada (muestra aleatoria de validación)",
    xaxis_title="y observada", yaxis_title="y estimada",
    width=560, height=480)
fig.show()
save_fig(fig, "y_vs_yhat.png")

comparacion = pd.DataFrame({
    "y observada": y_true.ravel(),
    "y estimada": y_hat.ravel(),
    "error": (y_true.ravel() - y_hat.ravel()),
}).round(4)
print(comparacion.head(20).to_string(index=False))

from sklearn.metrics import mean_squared_error, r2_score

yhat_all = model.predict(X_val)
rmse = float(mean_squared_error(y_val.ravel(), yhat_all.ravel()) ** 0.5)
r2 = float(r2_score(y_val.ravel(), yhat_all.ravel()))
print(f"RMSE validación: {rmse:.4f}")
print(f"R² validación : {r2:.4f}")'''
))

nb["cells"] = cells

nbf.write(nb, OUT)
print("escrito", OUT)
