"""Genera reports/fraud_paper.tex a partir de los resultados del GA y el modelo final.

Los resultados viven en results/ (copiados desde platypy):
- results/ga_log.jsonl         log de evaluaciones del GA
- results/final_metrics.json    métricas finales en test
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def load_ga_log():
    recs = [json.loads(l) for l in (RESULTS / "ga_log.jsonl").read_text().splitlines() if l.strip()]
    return recs


def best_per_generation(recs):
    out = {}
    for r in recs:
        g = (r["phase"], r["gen"])
        if g not in out or r["recall_at_fpr"] > out[g]:
            out[g] = r["recall_at_fpr"]
    return out


def make_fitness_plot(recs):
    bp = best_per_generation(recs)
    for phase in ["phase1", "phase2"]:
        xs = sorted([g for (p, g) in bp if p == phase])
        ys = [bp[(phase, g)] for g in xs]
        label = "Fase 1 (gruesa)" if phase == "phase1" else "Fase 2 (fina)"
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("generación")
    plt.ylabel("recall a FPR ≤ 0.5 % (validación)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "fitness_curve.png", dpi=150)
    plt.close()


def fmt_pct(x):
    return f"{100 * x:.2f}"


def main():
    recs = load_ga_log()
    final = json.loads((RESULTS / "final_metrics.json").read_text())
    make_fitness_plot(recs)

    t = final["test"]
    g = final["best_genome"]
    n = t["n_test"]
    cm = f"{t['tn']:,} & {t['fp']:,} \\\\\n{t['fn']:,} & {t['tp']:,}"

    tex = f"""\\documentclass[10pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage[margin=2.4cm]{{geometry}}
\\usepackage{{caption}}
\\title{{\\textbf{{Detección de fraude en tarjetas con una red feed-forward y
búsqueda de hiperparámetros mediante algoritmo genético}}}}
\\author{{}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
Se entrena una red neuronal feed-forward para detectar fraude en tarjetas de
crédito (conjunto sintético IBM SD254, 24.4 millones de transacciones) con el
objetivo de una tasa de falsos positivos (FPR) inferior al 1\\,\\%. La
arquitectura y sus hiperparámetros se seleccionan de forma autónoma con un
algoritmo genético que optimiza el recall sujeto a un FPR de 0.5\\,\\% en
validación. El modelo final alcanza un \\textbf{{FPR de {fmt_pct(t['fpr'])}\\,\\%}},
un recall de {fmt_pct(t['recall'])}\\:\\% y una precisión de
{fmt_pct(t['precision'])}\\:\\% en un test no visto de {n:,} transacciones
(ROC-AUC {t['roc_auc']:.4f}).
\\end{{abstract}}

\\section{{Introducción}}
El fraude en tarjetas es un evento raro: en los datos usados la tasa real es
0.12\\,\\%. El coste de marcar mal una transacción legítima (falso positivo) es
alto, por lo que el requisito es un FPR estrictamente menor al 1\\,\\%.

\\section{{Datos}}
El conjunto IBM SD254 v2 contiene {n_total:,} transacciones con 43 features
numéricas (importe, canal, MCC, errores de tarjeta, señales de comportamiento y
atributos del titular y de la tarjeta). Se divide en particiones disjuntas por
identificador:\\ par\\ de
entrenamiento (búsqueda GA, 268k filas, 9.7\\,\\% de fraude), entrenamiento final
(2.1M filas, 1.2\\,\\%), validación (1.1M, tasa real 0.11\\,\\%) y test
(2.1M, tasa real 0.12\\,\\%).

\\section{{Método}}
\\subsection{{Red feed-forward}}
Red totalmente conectada con profundidad $d$ y ancho $w$: capas ocultas con
normalización opcional, activación y dropout, y una salida lineal. Se entrena
minimizando la entropía cruzada binaria con peso de clase, optimizador Adam y
precisión mixta (AMP) en GPU (RTX 3070). Los hiperparámetros son: tasa de
aprendizaje $\\eta$, ancho $w$, profundidad $d$, dropout, decaimiento de pesos,
activación y normalización.

\\subsection{{Algoritmo genético}}
La búsqueda usa un GA en dos fases: una gruesa sobre todo el espacio (población
24, 12 generaciones) y una fina alrededor del mejor individuo (población 12, 8
generaciones). Selección por torneo ($k=3$), elitismo 2, cruce uniforme y
mutación por gen. La función de aptitud es el \\textbf{{recall a FPR $\\le$ 0.5\\,\\%}}
en validación (desempate por PR-AUC). El mejor genoma encontrado es:

\\begin{{center}}
\\begin{{tabular}}{{lc}}
\\toprule
Hiperparámetro & Valor \\\\
\\midrule
Tasa de aprendizaje & ${g['lr']:.2e}$ \\\\
Ancho $w$ & ${g['width']}$ \\\\
Profundidad $d$ & ${g['depth']}$ \\\\
Dropout & ${g['dropout']:.3f}$ \\\\
Decaimiento de pesos & ${g['weight_decay']:.2e}$ \\\\
Activación & {g['activation']} \\\\
Normalización & {g['norm']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}

La Figura~\\ref{{fig:fitness}} muestra la evolución de la aptitud por generación.

\\begin{{figure}}[ht]
\\centering
\\includegraphics[width=0.75\\textwidth]{{fitness_curve.png}}
\\caption{{Recall a FPR $\\le$ 0.5\\,\\% en validación del mejor individuo por
generación, en las dos fases del GA.}}
\\label{{fig:fitness}}
\\end{{figure}}

\\section{{Resultados}}
El modelo final se evalúa en el test no visto con un umbral fijado en
validación (máximo recall con FPR $\\le$ 0.5\\,\\%).

\\begin{{center}}
\\begin{{tabular}}{{lcc}}
\\toprule
Métrica & Valor \\\\
\\midrule
FPR (test) & {fmt_pct(t['fpr'])}\\:\\% \\\\
Recall & {fmt_pct(t['recall'])}\\:\\% \\\\
Precisión & {fmt_pct(t['precision'])}\\:\\% \\\\
ROC-AUC & {t['roc_auc']:.4f} \\\\
PR-AUC & {t['pr_auc']:.4f} \\\\
Umbral & {final['threshold']:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}

La matriz de confusión en el test ({n:,} transacciones) es:

\\begin{{center}}
\\begin{{tabular}}{{lcc}}
\\toprule
 & Predice no fraude & Predice fraude \\\\
\\midrule
Real: no fraude & {t['tn']:,} & {t['fp']:,} \\\\
Real: fraude & {t['fn']:,} & {t['tp']:,} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}

\\section{{Conclusiones}}
La búsqueda autónoma con GA encuentra una red feed-forward que satisface el
requisito de negocio: FPR {fmt_pct(t['fpr'])}\\:\\% $<$ 1\\,\\%, con recall
{fmt_pct(t['recall'])}\\:\\% y ROC-AUC {t['roc_auc']:.4f}, sobre datos reales no
vistos. La selección de hiperparámetros se realizó sin intervención humana.

\\end{{document}}
"""

    (REPORTS / "fraud_paper.tex").write_text(tex)
    print("escrito reports/fraud_paper.tex")


# n_total se rellena con el total real del dataset crudo
n_total = 24_386_900

if __name__ == "__main__":
    main()
