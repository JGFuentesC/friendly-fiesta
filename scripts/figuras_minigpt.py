"""Genera las figuras y los números del apunte de la Sesión 06 a partir del entrenamiento real.

Corre en local (macOS) con el entorno uv; solo necesita numpy y matplotlib.

Lee:
    data/processed/minigpt_teresa.json      (scripts/train_minigpt_teresa.py)

Escribe:
    reports/apuntes/06_llm/figuras/curva_perdida.png
    reports/apuntes/06_llm/figuras/tasa_aprendizaje.png
    reports/apuntes/06_llm/figuras/atencion_capas.png
    reports/apuntes/06_llm/figuras/atencion_cabezas.png
    reports/apuntes/06_llm/figuras/temperatura.png
    reports/apuntes/06_llm/figuras/embeddings.png
    reports/apuntes/06_llm/resultados.tex   macros con las métricas y las muestras

Uso:
    uv run python scripts/figuras_minigpt.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "processed" / "minigpt_teresa.json"
DESTINO = RAIZ / "reports" / "apuntes" / "06_llm"
FIG = DESTINO / "figuras"
NARANJA, PIZARRA, GRIS = "#FF5900", "#262C34", "#8A9099"

plt.rcParams.update({
    "figure.dpi": 200, "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555",
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8.5,
})


def pieza(p: str) -> str:
    return p.replace("▁", "·") if p != "<nl>" else "↵"


def curva(d):
    h, m = d["historia"], d["meta"]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.plot(h["paso"], h["train"], color=PIZARRA, lw=1, label="entrenamiento (media móvil por lote)")
    ax.plot(h["eval_paso"], h["eval_train"], "s", ms=2.5, color=PIZARRA, alpha=.6, label="entrenamiento (evaluación)")
    ax.plot(h["eval_paso"], h["eval_val"], "o-", ms=3, lw=1.4, color=NARANJA, label="validación")
    ax.axhline(math.log(d["config"]["vocab_size"]), ls=":", color=GRIS, lw=1)
    ax.text(h["eval_paso"][-1] * 0.06, math.log(d["config"]["vocab_size"]) + .08, r"$\ln V$ (adivinar al azar)", ha="left", color=GRIS, fontsize=7.5)
    ax.axvline(m["mejor_paso"], ls="--", color=GRIS, lw=.8)
    ax.annotate(f"mínimo de validación\npaso {m['mejor_paso']}, pérdida {m['mejor_val']:.2f}",
                xy=(m["mejor_paso"], m["mejor_val"]), xytext=(h["eval_paso"][-1] * 0.16, m["mejor_val"] + 2.6),
                arrowprops=dict(arrowstyle="->", color=GRIS, lw=.8), fontsize=7.5)
    ax.set_xlabel("paso de optimización")
    ax.set_ylabel("entropía cruzada (nats)")
    ax.set_ylim(min(h["train"] + h["eval_train"]) - .3, math.log(d["config"]["vocab_size"]) + .4)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "curva_perdida.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 1.9))
    ax.plot(h["paso"], np.array(h["lr"]) * 1e4, color=NARANJA, lw=1.4)
    ax.set_xlabel("paso")
    ax.set_ylabel(r"$\eta_t\ (\times 10^{-4})$")
    fig.tight_layout()
    fig.savefig(FIG / "tasa_aprendizaje.png")
    plt.close(fig)


def atencion(d):
    a = d["atencion"][0]
    piezas = [pieza(p) for p in a["piezas"]]
    mapas = np.array(a["mapas"])  # capa, cabeza, T, T
    L, H, T, _ = mapas.shape
    # Panel 1: promedio de cabezas por capa.
    fig, axs = plt.subplots(1, L, figsize=(7.2, 1.85))
    for c in range(L):
        axs[c].imshow(mapas[c].mean(0), cmap="Oranges", vmin=0, vmax=1)
        axs[c].set_title(f"capa {c + 1}", fontsize=7.5)
        axs[c].set_xticks([]); axs[c].set_yticks([])
    axs[0].set_yticks(range(T)); axs[0].set_yticklabels(piezas, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(FIG / "atencion_capas.png")
    plt.close(fig)
    # Panel 2: todas las cabezas de la primera y la última capa.
    fig, axs = plt.subplots(2, H, figsize=(7.2, 2.9))
    for fila, c in enumerate((0, L - 1)):
        for k in range(H):
            ax = axs[fila, k]
            ax.imshow(mapas[c, k], cmap="Oranges", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"capa {c + 1} · cabeza {k + 1}", fontsize=6)
            if k == 0:
                ax.set_yticks(range(T)); ax.set_yticklabels(piezas, fontsize=4.8)
    fig.tight_layout()
    fig.savefig(FIG / "atencion_cabezas.png")
    plt.close(fig)


def temperatura(d):
    s = next(x for x in d["siguiente"] if x["prefijo"] == "yo te")
    z = np.array(s["logits"])
    piezas = [pieza(p) for p in s["piezas"]]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, tau in zip(axs, (0.5, 1.0, 1.6)):
        p = np.exp((z - z.max()) / tau)
        p = p / p.sum()
        ax.bar(range(12), p[:12], color=NARANJA if tau == 1 else PIZARRA)
        ax.set_xticks(range(12)); ax.set_xticklabels(piezas[:12], rotation=70, fontsize=6.5)
        ax.set_title(rf"$\tau = {tau}$", fontsize=8.5)
    axs[0].set_ylabel("probabilidad\n(entre las 50 mejores)")
    fig.tight_layout()
    fig.savefig(FIG / "temperatura.png")
    plt.close(fig)


def embeddings(d):
    e = d["embeddings"]
    xy = np.array(e["tsne"])
    f = np.array(e["frecuencia"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(xy[:, 0], xy[:, 1], s=4 + 40 * np.sqrt(f / f.max()), color=NARANJA, alpha=.35, lw=0)
    for i, p in enumerate(e["piezas"][:110]):
        ax.text(xy[i, 0], xy[i, 1], pieza(p), fontsize=5.8, color=PIZARRA, ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "embeddings.png")
    plt.close(fig)


def escapar(t: str) -> str:
    """Escapa una pieza o frase para LaTeX; el marcador de espacio se dibuja con \\esp."""
    t = t.replace("\\", "/")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ("<", r"\textless{}"), (">", r"\textgreater{}")):
        t = t.replace(a, b)
    return t.replace("▁", r"\esp{}")


def latex_seguro(t: str) -> str:
    return t.replace("\\", "/").replace("\n", " / ")


def resultados(d):
    m, c = d["meta"], d["config"]
    lineas = [
        "% GENERADO por scripts/figuras_minigpt.py a partir de data/processed/minigpt_teresa.json",
        rf"\newcommand{{\mgCorpusMB}}{{{m['corpus_mb']:.2f}}}",
        rf"\newcommand{{\mgCapitulos}}{{{m['capitulos']}}}",
        rf"\newcommand{{\mgCapitulosVal}}{{{m['capitulos_val']}}}",
        rf"\newcommand{{\mgTokensTrain}}{{{m['tokens_train']:,}}}".replace(",", "\\,"),
        rf"\newcommand{{\mgTokensVal}}{{{m['tokens_val']:,}}}".replace(",", "\\,"),
        rf"\newcommand{{\mgCharsToken}}{{{m['corpus_caracteres'] / (m['tokens_train'] + m['tokens_val']):.2f}}}",
        rf"\newcommand{{\mgParametros}}{{{m['parametros']:,}}}".replace(",", "\\,"),
        rf"\newcommand{{\mgParametrosM}}{{{m['parametros'] / 1e6:.1f}}}",
        rf"\newcommand{{\mgPasos}}{{{m['pasos']:,}}}".replace(",", "\\,"),
        rf"\newcommand{{\mgMejorPaso}}{{{m['mejor_paso']:,}}}".replace(",", "\\,"),
        rf"\newcommand{{\mgMejorVal}}{{{m['mejor_val']:.2f}}}",
        rf"\newcommand{{\mgMejorPPL}}{{{m['mejor_ppl']:.1f}}}",
        rf"\newcommand{{\mgMinutos}}{{{m['minutos']:.0f}}}",
        rf"\newcommand{{\mgVRAM}}{{{m['vram_pico_gb']:.1f}}}",
        rf"\newcommand{{\mgVocab}}{{{c['vocab_size']}}}",
        rf"\newcommand{{\mgLnV}}{{{math.log(c['vocab_size']):.2f}}}",
        rf"\newcommand{{\mgEpocas}}{{{m['pasos'] * m['tokens_por_paso'] / m['tokens_train']:.0f}}}",
        rf"\newcommand{{\mgEpocasMejor}}{{{m['mejor_paso'] * m['tokens_por_paso'] / m['tokens_train']:.0f}}}",
        rf"\newcommand{{\mgValFinal}}{{{d['historia']['eval_val'][-1]:.2f}}}",
        rf"\newcommand{{\mgTrainFinal}}{{{d['historia']['eval_train'][-1]:.2f}}}",
        rf"\newcommand{{\mgDescartado}}{{{100 * (1 - m['mejor_ppl'] / c['vocab_size']):.0f}}}",
    ]
    (DESTINO / "resultados.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # Tabla de tokenización real: cada pieza en una cajita, el espacio inicial en naranja.
    filas = [r"\begin{center}\small", r"\begin{tabular}{@{}p{0.34\textwidth}p{0.6\textwidth}@{}}", r"\toprule",
             r"Frase & Piezas del tokenizador entrenado \\", r"\midrule"]
    for t in d["tokenizacion"]:
        cajas = r"\,".join(r"\tok{" + escapar(p) + "}" for p in t["piezas"])
        filas.append(rf"\emph{{{escapar(t['frase'])}}} ({len(t['piezas'])} tokens) & {cajas} \\[2pt]")
    filas += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    (DESTINO / "tokenizacion.tex").write_text("% GENERADO por scripts/figuras_minigpt.py\n" + "\n".join(filas) + "\n", encoding="utf-8")

    # Muestras: van en un listado para que su texto (generado por el modelo) no se mezcle con la prosa.
    bloque = []
    elegidas = [mu for mu in d["muestras"] if mu["paso"] in (0, 150, 600, 1500) or mu.get("etiqueta") == "mejor"]
    for mu in elegidas:
        etiqueta = " (mejor validación)" if mu.get("etiqueta") == "mejor" else ""
        texto = latex_seguro(mu["texto"])[:300]
        bloque.append(f"[paso {mu['paso']}{etiqueta}]\n{texto}\n")
    (DESTINO / "muestras.txt").write_text("\n".join(bloque), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = json.loads(DATOS.read_text(encoding="utf-8"))
    curva(d)
    atencion(d)
    temperatura(d)
    embeddings(d)
    resultados(d)
    print("figuras en", FIG.relative_to(RAIZ), "y macros en", (DESTINO / "resultados.tex").relative_to(RAIZ))


if __name__ == "__main__":
    main()
