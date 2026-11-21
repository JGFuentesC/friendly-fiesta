"""Genera notebooks/09_ruido_hotot.ipynb: ruido gaussiano añadido paso a paso a una imagen.

Corre en local (numpy, pillow y matplotlib). La animación usa FuncAnimation.to_jshtml,
así que los controles de reproducción quedan guardados en el notebook ejecutado.

Uso:
    uv run python scripts/build_ruido_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/09_ruido_hotot.ipynb
    uv run python scripts/verificar_redaccion_html.py notebooks/09_ruido_hotot.ipynb
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "09_ruido_hotot.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# Sesión 09. Ruido sobre una imagen, paso a paso

Una imagen a color es un tensor de forma $(H, W, 3)$. En este cuaderno le sumamos ruido gaussiano en $T$ pasos, igual que el proceso directo de un modelo de difusión:

$$\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\boldsymbol\epsilon_t,\qquad \boldsymbol\epsilon_t\sim\mathcal N(\mathbf 0,\mathbf I).$$

En cada paso la señal se encoge un poco y entra un poco de ruido. Al final el tensor tiene la misma forma, pero sus valores siguen una gaussiana estándar y la foto desaparece.

La foto va en `data/raw/hotot/hotot.jpg` y no se versiona; sirve cualquier JPG.

Iremos viendo tres cosas a la vez: la imagen, el histograma de sus valores y un parche de $6\times 6$ píxeles del canal rojo con sus números."""))

cells.append(code(r"""from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from PIL import Image

plt.rcParams["animation.embed_limit"] = 60  # MB

SEED = 42
rng = np.random.default_rng(SEED)
NARANJA, PIZARRA = "#FF5900", "#262C34"


def find_repo_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("No encuentro pyproject.toml")


ROOT = find_repo_root()
RUTA = ROOT / "data" / "raw" / "hotot" / "hotot.jpg"
if not RUTA.exists():
    raise FileNotFoundError(f"Falta {RUTA}: coloca ahí cualquier foto JPG (la original no se versiona)")
"""))

cells.append(md(r"""## 1. La imagen como tensor

Leemos el JPEG, lo pasamos a `float32` y lo reescalamos de $[0,255]$ a $[-1,1]$, la escala habitual en difusión. Así el ruido estándar y la imagen viven en el mismo rango."""))

cells.append(code(r"""img = np.asarray(Image.open(RUTA).convert("RGB"))
x0 = img.astype(np.float32) / 127.5 - 1.0

print("forma:", x0.shape, " dtype:", x0.dtype)
print(f"mínimo={x0.min():.3f}  máximo={x0.max():.3f}  media={x0.mean():.3f}  desviación={x0.std():.3f}")

# Parche de 6x6 que seguiremos en todo el cuaderno (borde del ojo, canal rojo)
FILA, COL, LADO = 183, 288, 6
parche = np.s_[FILA:FILA + LADO, COL:COL + LADO, 0]
print("\nParche x0[filas, columnas, canal R]:")
print(np.round(x0[parche], 2))"""))

cells.append(code(r"""def a_imagen(x):
    # De [-1, 1] a [0, 1] para mostrar; lo que sale del rango se recorta
    return np.clip((x + 1) / 2, 0, 1)


fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(a_imagen(x0))
ax.add_patch(plt.Rectangle((COL - 0.5, FILA - 0.5), LADO, LADO, fill=False, ec=NARANJA, lw=2))
ax.set_title("x0 (recuadro: parche que seguimos)")
ax.axis("off")
plt.show()"""))

cells.append(md(r"""## 2. Añadir ruido en $T$ pasos

Usamos un horario lineal $\beta_t$ de $10^{-4}$ a $0.02$ con $T=500$. Guardamos el tensor cada 20 pasos para la animación. Junto a cada instantánea registramos la media, la desviación y la correlación con $\mathbf{x}_0$, que mide cuánta foto queda.

La forma cerrada predice esa correlación: $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$, con $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$."""))

cells.append(code(r"""T = 500
CADA = 20
betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
alpha_barra = np.cumprod(1 - betas)

x = x0.copy()
pasos, tensores = [0], [x0.copy()]
for t in range(1, T + 1):
    eps = rng.standard_normal(x.shape).astype(np.float32)
    x = np.sqrt(1 - betas[t - 1]) * x + np.sqrt(betas[t - 1]) * eps
    if t % CADA == 0:
        pasos.append(t)
        tensores.append(x.copy())


def correlacion(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


stats = [
    dict(t=t, media=float(xt.mean()), desv=float(xt.std()), corr=correlacion(xt, x0),
         sqrt_ab=float(np.sqrt(alpha_barra[t - 1])) if t else 1.0)
    for t, xt in zip(pasos, tensores)
]
print(f"{len(tensores)} instantáneas, cada una con forma {tensores[-1].shape}")
print(f"en t={T}: media={stats[-1]['media']:.3f}  desviación={stats[-1]['desv']:.3f}  corr con x0={stats[-1]['corr']:.3f}")"""))

cells.append(md(r"""## 3. Animación: imagen, histograma y números del tensor

Pulsa reproducir o arrastra la barra. A la izquierda la imagen; al centro el histograma de todos los valores del tensor (en gris el de $\mathbf{x}_0$, la línea naranja es $\mathcal N(0,1)$); a la derecha el parche de $6\times 6$ con sus valores."""))

cells.append(code(r"""rejilla = np.linspace(-4, 4, 200)
normal = np.exp(-rejilla**2 / 2) / np.sqrt(2 * np.pi)
bins = np.linspace(-4, 4, 81)

fig, (ax_img, ax_hist, ax_par) = plt.subplots(1, 3, figsize=(13, 4.4), dpi=72, gridspec_kw=dict(width_ratios=[1, 1.2, 1]))

im = ax_img.imshow(a_imagen(x0))
ax_img.axis("off")

ax_hist.hist(x0.ravel(), bins=bins, density=True, color="#BBBBBB", label="x0")
_, _, barras = ax_hist.hist(x0.ravel(), bins=bins, density=True, color=PIZARRA, alpha=0.75, label="xt")
ax_hist.plot(rejilla, normal, color=NARANJA, lw=2, label="N(0,1)")
ax_hist.set_xlim(-4, 4)
ax_hist.set_ylim(0, 1.6)
ax_hist.set_xlabel("valor del tensor")
ax_hist.legend(loc="upper right", fontsize=8)

mapa = ax_par.imshow(x0[parche], cmap="RdBu_r", vmin=-2.5, vmax=2.5)
textos = [[ax_par.text(j, i, "", ha="center", va="center", fontsize=8) for j in range(LADO)] for i in range(LADO)]
ax_par.set_xticks([]); ax_par.set_yticks([])
ax_par.set_title("parche 6x6, canal R")
titulo = fig.suptitle("")


def dibujar(k):
    xt, s = tensores[k], stats[k]
    im.set_data(a_imagen(xt))
    alturas, _ = np.histogram(xt.ravel(), bins=bins, density=True)
    for b, h in zip(barras, alturas):
        b.set_height(h)
    p = xt[parche]
    mapa.set_data(p)
    for i in range(LADO):
        for j in range(LADO):
            textos[i][j].set_text(f"{p[i, j]:.2f}")
            textos[i][j].set_color("white" if abs(p[i, j]) > 1.4 else "black")
    ax_img.set_title(f"t = {s['t']}")
    titulo.set_text(f"media={s['media']:+.3f}   desviación={s['desv']:.3f}   "
                    f"corr(xt, x0)={s['corr']:.3f}   sqrt(alfa_barra)={s['sqrt_ab']:.3f}")
    return [im, mapa, titulo, *barras]


anim = animation.FuncAnimation(fig, dibujar, frames=len(tensores), interval=350)
plt.close(fig)
HTML(anim.to_jshtml())"""))

cells.append(md(r"""## 4. Instantáneas y estadísticas del tensor

La misma historia en una rejilla estática y en números. La desviación sube hacia 1 y la media cae hacia 0. De la forma cerrada sale también la correlación esperada con la imagen original, con $\sigma_0$ la desviación de $\mathbf{x}_0$:

$$\rho_t=\frac{\sqrt{\bar\alpha_t}\,\sigma_0}{\sqrt{\bar\alpha_t\,\sigma_0^2+1-\bar\alpha_t}}.$$

La gráfica final compara esa curva con la correlación medida en cada instantánea."""))

cells.append(code(r"""mostrar = [0, 2, 5, 8, 12, 16, 20, len(tensores) - 1]
fig, axes = plt.subplots(1, len(mostrar), figsize=(2 * len(mostrar), 2.4))
for ax, k in zip(axes, mostrar):
    ax.imshow(a_imagen(tensores[k]))
    ax.set_title(f"t={pasos[k]}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

print(f"{'t':>4} {'media':>8} {'desv':>7} {'corr x0':>8} {'sqrt(ab)':>9}")
for s in stats[::2] + ([stats[-1]] if len(stats) % 2 == 0 else []):
    print(f"{s['t']:>4} {s['media']:>8.3f} {s['desv']:>7.3f} {s['corr']:>8.3f} {s['sqrt_ab']:>9.3f}")"""))

cells.append(code(r"""fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(pasos, [s["corr"] for s in stats], "o", color=PIZARRA, label="corr(xt, x0) medida")
s0 = x0.std()
rho = np.sqrt(alpha_barra) * s0 / np.sqrt(alpha_barra * s0**2 + 1 - alpha_barra)
ax.plot(np.arange(1, T + 1), rho, color=NARANJA, label="correlación predicha")
ax.plot(np.arange(1, T + 1), np.sqrt(alpha_barra), color=NARANJA, ls="--", lw=1, label="sqrt(alfa_barra_t)")
ax.plot(pasos, [s["desv"] for s in stats], "s", color="#5B6B7F", ms=4, label="desviación de xt")
ax.set_xlabel("paso t")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))

nb.cells = cells

# Regla de redacción: todo título que empieza con interrogativo acentuado lleva ¿
INTERROGATIVO = re.compile(r"^#+\s*(qué|quién|cómo|dónde|cuándo|cuánto|cuál|por qué)\b", re.IGNORECASE | re.MULTILINE)
for c in cells:
    if c.cell_type == "markdown" and INTERROGATIVO.search(c.source):
        sys.exit(f"Título interrogativo sin ¿: {INTERROGATIVO.search(c.source).group(0)}")

nbf.write(nb, OUT)
print(f"Escrito {OUT.relative_to(REPO_ROOT)}")
