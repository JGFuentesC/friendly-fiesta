"""Genera las figuras con resultados reales del apunte de la Sesión 07 (difusión).

Corre en local (uv run python scripts/figuras_difusion.py). Lee data/processed/difusion.json,
que escribe scripts/train_ddpm.py en el servidor GPU, y escribe en
reports/apuntes/07_difusion/figuras/:
  directo_2d.png, inverso_2d.png, score_2d.png, paisajes_tira.png, guia.png,
  mnist_tira.png, perdida_detalle.png   figuras PNG
  horarios.dat, perdida_t.dat           tablas para pgfplots
  metricas.tex                          cuadro con las métricas de las corridas
"""

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON = REPO_ROOT / "data" / "processed" / "difusion.json"
FIG = REPO_ROOT / "reports" / "apuntes" / "07_difusion" / "figuras"
NARANJA, PIZARRA = "#FF5900", "#262C34"
SEED = 42


def i16(s, forma, escala=1000):
    return np.frombuffer(base64.b64decode(s), dtype="<i2").astype(np.float64).reshape(forma) / escala


def u8(s, forma):
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(forma)


def ejes_limpios(a, lim=3.4):
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_color("#cccccc")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = json.loads(JSON.read_text())
    T = d["meta"]["T"]
    ab_lin = np.array(d["horarios"]["lineal"]["alpha_bar"])
    ab_cos = np.array(d["horarios"]["coseno"]["alpha_bar"])

    # Tablas para pgfplots
    with open(FIG / "horarios.dat", "w") as f:
        f.write("t lineal coseno snrlin snrcos\n")
        for t in list(range(1, T + 1, 5)) + [T]:
            a, c = ab_lin[t - 1], ab_cos[t - 1]
            f.write(f"{t} {a:.6f} {c:.6f} {np.log10(a / (1 - a)):.4f} {np.log10(max(c, 1e-12) / (1 - c)):.4f}\n")
    pm = d["mnist"]["perdida_vs_t"]
    p2 = d["dos_d"]["montanas"]["perdida_vs_t"]
    with open(FIG / "perdida_t.dat", "w") as f:
        f.write("t mnist montanas\n")
        for t, a, b in zip(pm["t"], pm["mse"], p2["mse"]):
            f.write(f"{t} {a} {b}\n")

    # Proceso directo 2D con el mismo epsilon por punto
    mont = d["dos_d"]["montanas"]
    x0 = i16(mont["datos"], (mont["n_datos"], 2))
    eps = np.random.default_rng(SEED).standard_normal(x0.shape)
    ts = [0, 25, 75, 150, 300, 600, 1000]
    fig, axs = plt.subplots(1, len(ts), figsize=(14, 2.4))
    for a, t in zip(axs, ts):
        xt = x0 if t == 0 else np.sqrt(ab_lin[t - 1]) * x0 + np.sqrt(1 - ab_lin[t - 1]) * eps
        a.scatter(xt[:, 0], xt[:, 1], s=0.6, color=NARANJA if t == 0 else PIZARRA)
        ejes_limpios(a)
        a.set_title(f"t = {t}" + ("" if t == 0 else f"   $\\bar\\alpha_t$ = {ab_lin[t - 1]:.3f}"), fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "directo_2d.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Proceso inverso 2D
    objetivo = [1000, 600, 300, 150, 60, 20, 0]
    fig, axs = plt.subplots(2, len(objetivo), figsize=(14, 4.6))
    for r, nombre in enumerate(["montanas", "ua"]):
        dd = d["dos_d"][nombre]
        inst = dd["instantes"]
        tray = i16(dd["trayectorias"], (len(inst), dd["n_particulas"], 2))
        for j, t in enumerate(objetivo):
            k = int(np.argmin(np.abs(np.array(inst) - t)))
            a = axs[r, j]
            a.scatter(tray[k, :, 0], tray[k, :, 1], s=0.8, color=NARANJA if inst[k] == 0 else PIZARRA)
            ejes_limpios(a)
            if r == 0:
                a.set_title(f"t = {inst[k]}", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "inverso_2d.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Campo de score
    g = np.array(mont["rejilla"])
    gx, gy = np.meshgrid(g, g)
    tc = mont["t_campo"]
    campo = i16(mont["campo_eps"], (len(tc), len(g) * len(g), 2))
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.7))
    for a, t in zip(axs, [700, 200, 50, 10]):
        e = campo[tc.index(t)]
        a.scatter(x0[:, 0], x0[:, 1], s=0.4, color=NARANJA, alpha=0.45)
        a.quiver(gx, gy, -e[:, 0].reshape(gx.shape), -e[:, 1].reshape(gx.shape), color=PIZARRA, scale=38, width=0.0038)
        ejes_limpios(a)
        a.set_title(f"$-\\epsilon_\\theta(x, t)$ con t = {t}", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "score_2d.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Paisajes: tira x_T -> x_0
    p = d["paisajes"]
    tira = u8(p["tira"], p["forma_tira"])  # (9, n_inst, 24, 24, 3)
    inst = p["instantes"]
    cols = [0, 3, 6, 8, 10, 12, 14, 16, 17, 18, len(inst) - 1]
    filas = [0, 1, 3, 4, 6, 7]
    fig, axs = plt.subplots(len(filas), len(cols), figsize=(12, 7.2))
    for r, fi in enumerate(filas):
        for j, k in enumerate(cols):
            axs[r, j].imshow(tira[fi, k], interpolation="nearest")
            axs[r, j].axis("off")
            if r == 0:
                axs[r, j].set_title(f"t = {inst[k]}", fontsize=9)
        if fi % p["por_prompt"] == 0:
            axs[r, 0].text(-4, 24, p["prompts"][fi // p["por_prompt"]], ha="right", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "paisajes_tira.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Guía: paisajes y MNIST
    gp = u8(p["guia"], p["forma_guia"])
    m = d["mnist"]
    gm = u8(m["guia"], m["forma_guia"])
    fig = plt.figure(figsize=(14, 3.4))
    sub = fig.subfigures(1, 2, width_ratios=[1, 3.1])
    axl = sub[0].subplots(3, 3)
    axr = sub[1].subplots(3, 10)
    for r, w in enumerate(p["guia_escalas"]):
        for c in range(3):
            axl[r, c].imshow(gp[r, c], interpolation="nearest")
            axl[r, c].axis("off")
            if r == 0:
                axl[r, c].set_title(p["prompts"][c].replace(" ", "\n", 1), fontsize=7)
        axl[r, 0].text(-3, 12, f"w = {w:g}", ha="right", va="center", fontsize=9)
        for c in range(10):
            axr[r, c].imshow(gm[r, c], cmap="gray", interpolation="nearest")
            axr[r, c].axis("off")
            if r == 0:
                axr[r, c].set_title(str(c), fontsize=9)
        axr[r, 0].text(-3, 14, f"w = {w:g}", ha="right", va="center", fontsize=9)
    plt.savefig(FIG / "guia.png", dpi=200, bbox_inches="tight")
    plt.close()

    # MNIST: 16 trayectorias
    tm = u8(m["tira"], m["forma_tira"])  # (16, n_inst, 28, 28)
    n, ni = tm.shape[:2]
    mosaico = np.ones((n * 30, ni * 30), dtype=np.uint8) * 255
    for r in range(n):
        for k in range(ni):
            mosaico[r * 30 + 1:r * 30 + 29, k * 30 + 1:k * 30 + 29] = tm[r, k]
    fig, a = plt.subplots(figsize=(11, 9))
    a.imshow(mosaico, cmap="gray", interpolation="nearest")
    a.set_xticks([k * 30 + 14 for k in range(ni)])
    a.set_xticklabels([str(t) for t in m["instantes"]], fontsize=7)
    a.set_yticks([r * 30 + 14 for r in range(n)])
    a.set_yticklabels([str(e) for e in m["etiquetas"]], fontsize=8)
    a.set_xlabel("t")
    a.set_ylabel("clase pedida")
    for s in a.spines.values():
        s.set_visible(False)
    plt.savefig(FIG / "mnist_tira.png", dpi=200, bbox_inches="tight")
    plt.close()

    # La pérdida por dentro
    pd_ = m["perdida_detalle"]
    im = u8(pd_["imagenes"], pd_["forma"])
    titulos = ["$x_t$", "$\\epsilon$ verdadero", "$\\epsilon_\\theta$ predicho", "diferencia", "$\\hat{x}_0$"]
    fig, axs = plt.subplots(len(pd_["t"]), 5, figsize=(7.5, 9.2))
    for r, t in enumerate(pd_["t"]):
        for c in range(5):
            axs[r, c].imshow(im[r, c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axs[r, c].axis("off")
            if r == 0:
                axs[r, c].set_title(titulos[c], fontsize=10)
        axs[r, 0].text(-3, 14, f"t = {t}\nMSE = {pd_['mse'][r]:.3f}", ha="right", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "perdida_detalle.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Cuadro de métricas
    def fin(c):
        return np.mean(c["perdida"][-10:])

    filas_tab = [
        ("Montañas 2D", "MLP", "60\\,000 puntos", "16\\,000 pasos", fin(d["dos_d"]["montanas"]["curva"]), d["dos_d"]["montanas"]["segundos"]),
        ("Letras UA 2D", "MLP", "60\\,000 puntos", "16\\,000 pasos", fin(d["dos_d"]["ua"]["curva"]), d["dos_d"]["ua"]["segundos"]),
        ("Paisajes $24\\times24$", "U-Net condicional", f"{p['n_entrenamiento']:,}".replace(",", "\\,") + " imágenes", "14 épocas", fin(p["curva"]), p["segundos"]),
        ("MNIST $28\\times28$", "U-Net condicional", "60\\,000 imágenes", "12 épocas", fin(m["curva"]), m["segundos"]),
    ]
    with open(FIG / "metricas.tex", "w") as f:
        f.write("% Generado por scripts/figuras_difusion.py\n")
        f.write("\\begin{table}[H]\n\\centering\\small\n\\begin{tabular}{lllrrr}\n\\toprule\n")
        f.write("Modelo & Red & Datos & Entrenamiento & $L_{\\text{simple}}$ final & Tiempo (s)\\\\\n\\midrule\n")
        for fila in filas_tab:
            f.write(f"{fila[0]} & {fila[1]} & {fila[2]} & {fila[3]} & {fila[4]:.4f} & {fila[5]:.0f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        gpu = d["meta"].get("gpu", d["meta"]["dispositivo"])
        f.write(f"\\caption{{Corridas de \\texttt{{scripts/train\\_ddpm.py}} en una {gpu}. Tiempo total con muestreo y exportación: "
                f"{d['meta']['segundos_total'] / 60:.1f} minutos; pico de memoria de GPU: {d['meta'].get('vram_pico_mb', 0)} MB.}}\n")
        f.write("\\label{tab:metricas}\n\\end{table}\n")
    print("figuras escritas en", FIG.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
