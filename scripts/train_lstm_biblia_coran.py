"""Clasificador LSTM Biblia/Corán en PyTorch con valores de Shapley por palabra.

Corre en el contenedor lab-pytorch de platypy (GPU). Estima P(Corán | documento)
sobre ventanas de hasta 200 palabras y escribe:

  data/processed/lstm_resultados.json   métricas, curvas y SHAP para la SPA
  figures/lstm/anatomy_data.json        valores reales para el diagrama isométrico
  results/lstm_biblia_coran/modelo.pt   pesos y vocabulario

Tres decisiones evitan que el modelo aprenda atajos:
  1. Se eliminan los encabezados editoriales (SURA n, LIBRO, 1 SAMUEL 3, Parte #).
  2. El reparto es por grupos: suras completas y bloques de 25 ventanas contiguas
     de la Biblia, de modo que dos ventanas vecinas nunca caen en particiones distintas.
  3. Cada lote de entrenamiento recorta los documentos a una longitud aleatoria,
     porque sin eso la longitud delataría al Corán (solo sus ventanas son cortas).

Uso dentro del contenedor:
    python scripts/train_lstm_biblia_coran.py

La explicación usa valores de Shapley muestreados por permutaciones: cada palabra
distinta del documento es un jugador y quitarla significa cambiarla por <unk>.
shap.GradientExplainer se descartó porque no convergía en esta red: su error de
aditividad siguió en 2.6 logits al pasar de 100 a 2000 muestras.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.nn.utils.rnn import pack_padded_sequence

RAIZ = Path(__file__).resolve().parents[1]
TEXTOS = RAIZ / "data" / "raw" / "textos_nlp"
SALIDA_JSON = RAIZ / "data" / "processed" / "lstm_resultados.json"
SALIDA_ANATOMIA = RAIZ / "figures" / "lstm" / "anatomy_data.json"
SALIDA_MODELO = RAIZ / "results" / "lstm_biblia_coran" / "modelo.pt"
STOPWORDS = RAIZ / "data" / "processed" / "bow_nltk_biblia.json"

SEMILLA = 42
VENTANA = 200
LARGO_MINIMO = 10       # documentos más cortos se descartan
BLOQUE_BIBLIA = 25      # ventanas contiguas que viajan juntas en el reparto
FRACCIONES = (0.70, 0.15, 0.15)
VOCABULARIO = 8000
EMBEDDING = 64
OCULTA = 64
CAPAS = 2
DENSA = 32
DROPOUT = 0.3
LOTE = 64
LR = 1e-3
EPOCAS_MAX = 120
PACIENCIA = 15
TIEMPO_MAX_S = 120
LARGOS_EVAL = [10, 20, 50, 100, 200]
SHAP_POR_CLASE = 120
SHAP_PERMUTACIONES = 16  # cada una se evalúa también invertida (muestreo antitético)
SHAP_LOTE = 1024
SHAP_REMUESTREOS = 1000
SHAP_MIN_APARICIONES = 10
TOP = 20
PASOS_DIAGRAMA = 16

PAD, UNK = 0, 1
CLASES = {"biblia": 0, "coran": 1}

PATRON_PALABRAS = re.compile(r"[a-záéíóúüñ]+")
PATRON_SURA = re.compile(r"^SURA \d+:.*$", re.MULTILINE)
PATRON_MAYUSCULAS = re.compile(r"^(?=.*[A-ZÁÉÍÓÚÑ])[^a-záéíóúüñ]+$")
PATRON_METADATOS = re.compile(r"^(LA SANTA BIBLIA|Parte #)")


def tokenizar(texto: str) -> list[str]:
    return PATRON_PALABRAS.findall(texto.lower())


def en_ventanas(palabras: list[str]) -> list[list[str]]:
    trozos = [palabras[i:i + VENTANA] for i in range(0, len(palabras), VENTANA)]
    return [t for t in trozos if len(t) >= LARGO_MINIMO]


def documentos_biblia() -> list[dict]:
    """Quita encabezados en mayúsculas y metadatos, y corta en ventanas fijas."""
    lineas = (TEXTOS / "biblia_es.txt").read_text(encoding="utf-8").splitlines()
    limpias = [
        l for l in lineas
        if not PATRON_MAYUSCULAS.match(l.strip()) and not PATRON_METADATOS.match(l.strip())
    ]
    return [
        {"tokens": t, "corpus": "biblia", "grupo": f"biblia-{i // BLOQUE_BIBLIA}"}
        for i, t in enumerate(en_ventanas(tokenizar("\n".join(limpias))))
    ]


def documentos_coran() -> list[dict]:
    """Corta por sura sin cruzarlas; el encabezado de cada sura se descarta."""
    texto = (TEXTOS / "coran_es.txt").read_text(encoding="utf-8")
    marcas = list(PATRON_SURA.finditer(texto))
    docs = []
    for i, marca in enumerate(marcas):
        fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
        for t in en_ventanas(tokenizar(texto[marca.end():fin])):
            docs.append({"tokens": t, "corpus": "coran", "grupo": f"sura-{i + 1}"})
    return docs


def repartir(docs: list[dict], azar: random.Random) -> dict[str, list[dict]]:
    """Reparto 70/15/15 por grupos, por separado en cada corpus."""
    particiones: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for corpus in CLASES:
        grupos: dict[str, list[dict]] = defaultdict(list)
        for d in docs:
            if d["corpus"] == corpus:
                grupos[d["grupo"]].append(d)
        nombres = sorted(grupos)
        azar.shuffle(nombres)
        total = sum(len(g) for g in grupos.values())
        acumulado = 0
        for nombre in nombres:
            frac = acumulado / total
            destino = "train" if frac < FRACCIONES[0] else "val" if frac < FRACCIONES[0] + FRACCIONES[1] else "test"
            particiones[destino].extend(grupos[nombre])
            acumulado += len(grupos[nombre])
    return particiones


class ClasificadorLSTM(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, EMBEDDING, padding_idx=PAD)
        self.drop = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(EMBEDDING, OCULTA, num_layers=CAPAS, bidirectional=True, dropout=DROPOUT, batch_first=True)
        self.densa = nn.Linear(2 * OCULTA, DENSA)
        self.salida = nn.Linear(DENSA, 1)

    def desde_embedding(self, emb: torch.Tensor, largos: torch.Tensor | None = None) -> torch.Tensor:
        entrada = emb if largos is None else pack_padded_sequence(emb, largos.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(entrada)
        z = self.drop(torch.cat([h[-2], h[-1]], dim=1))
        return self.salida(self.drop(torch.relu(self.densa(z))))

    def forward(self, ids: torch.Tensor, largos: torch.Tensor) -> torch.Tensor:
        return self.desde_embedding(self.drop(self.embedding(ids)), largos).squeeze(1)


def a_ids(tokens: list[str], palabra_id: dict[str, int]) -> list[int]:
    return [palabra_id.get(t, UNK) for t in tokens]


def lote_tensores(secuencias: list[list[int]], dispositivo) -> tuple[torch.Tensor, torch.Tensor]:
    largos = torch.tensor([len(s) for s in secuencias])
    ids = torch.full((len(secuencias), int(largos.max())), PAD, dtype=torch.long)
    for i, s in enumerate(secuencias):
        ids[i, :len(s)] = torch.tensor(s)
    return ids.to(dispositivo), largos


def recorte(seq: list[int], azar: random.Random) -> list[int]:
    """Subsecuencia contigua de longitud uniforme entre LARGO_MINIMO y la del documento."""
    largo = azar.randint(min(LARGO_MINIMO, len(seq)), len(seq))
    inicio = azar.randint(0, len(seq) - largo)
    return seq[inicio:inicio + largo]


@torch.no_grad()
def logits_de(modelo, secuencias: list[list[int]], dispositivo, lote: int = 128) -> np.ndarray:
    modelo.eval()
    orden = np.argsort([len(s) for s in secuencias])
    salida = np.zeros(len(secuencias))
    for i in range(0, len(orden), lote):
        idx = orden[i:i + lote]
        ids, largos = lote_tensores([secuencias[j] for j in idx], dispositivo)
        salida[idx] = modelo(ids, largos).cpu().numpy()
    return salida


def sigmoide(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_biblia": int((y == 0).sum()),
        "n_coran": int((y == 1).sum()),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_coran": float(precision_score(y, pred, zero_division=0)),
        "recall_coran": float(recall_score(y, pred, zero_division=0)),
        "recall_biblia": float(tn / max(tn + fp, 1)),
        "f1_coran": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibracion(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    cajas, ece = [], 0.0
    for b in range(bins):
        mascara = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1)
        if mascara.sum():
            confianza, tasa = float(p[mascara].mean()), float(y[mascara].mean())
            ece += mascara.mean() * abs(confianza - tasa)
            cajas.append({"p_media": confianza, "tasa_coran": tasa, "n": int(mascara.sum())})
    return {"ece": float(ece), "cajas": cajas}


@torch.inference_mode()
def shapley_por_palabra(modelo, seq: list[int], rng: np.random.Generator, dispositivo) -> tuple[dict[int, float], dict[int, float], float, float]:
    """Valores de Shapley del logit por palabra distinta, con referencia en <unk>.

    Cada permutación es una suma telescópica de f(todo) - f(nada), de modo que la
    aditividad es exacta salvo redondeo. Devuelve también las dos mitades de las
    permutaciones por separado, para medir cuánto ha convergido la estimación.
    """
    modelo.eval()
    arreglo = np.array(seq)
    jugadores = np.array(sorted(set(seq) - {UNK}))
    n = len(jugadores)
    jugador_de = {int(w): j for j, w in enumerate(jugadores)}
    pos_jugador = np.array([jugador_de.get(int(w), -1) for w in arreglo])
    ordenes = []
    for _ in range(SHAP_PERMUTACIONES):
        orden = rng.permutation(n)
        ordenes += [orden, orden[::-1]]
    filas = []
    for orden in ordenes:
        rango = np.empty(n + 1, dtype=int)
        rango[orden] = np.arange(n)
        rango[-1] = n + 1  # los tokens <unk> originales nunca se activan
        filas.append(np.stack([np.where(rango[pos_jugador] < k, arreglo, UNK) for k in range(n + 1)]))
    entradas = torch.from_numpy(np.concatenate(filas))
    largos = torch.full((len(entradas),), len(seq))
    logits = torch.cat([
        modelo(entradas[i:i + SHAP_LOTE].to(dispositivo), largos[i:i + SHAP_LOTE]).cpu()
        for i in range(0, len(entradas), SHAP_LOTE)
    ]).numpy().reshape(len(ordenes), n + 1)
    delta = np.zeros((len(ordenes), n))
    for m, orden in enumerate(ordenes):
        delta[m, orden] = np.diff(logits[m])
    fi = delta.mean(axis=0)
    mitad_a = delta[:SHAP_PERMUTACIONES].mean(axis=0)
    mitad_b = delta[SHAP_PERMUTACIONES:].mean(axis=0)
    claves = [int(w) for w in jugadores]
    return dict(zip(claves, fi)), {"a": mitad_a, "b": mitad_b}, float(logits[0, 0]), float(logits[0, -1])


def pasos_lstm(modelo: ClasificadorLSTM, emb: torch.Tensor) -> tuple[dict, float]:
    """Recorre la LSTM a mano con las ecuaciones de la teoría y devuelve compuertas y estados.

    Compara el resultado con nn.LSTM: el error devuelto demuestra que las ecuaciones
    de la sección de teoría son las que el modelo entrenado calcula.
    """
    lstm = modelo.lstm
    entrada = emb
    celdas = {}
    for capa in range(CAPAS):
        salidas = []
        for sufijo, nombre in (("", "adelante"), ("_reverse", "atras")):
            w_x = getattr(lstm, f"weight_ih_l{capa}{sufijo}")
            w_h = getattr(lstm, f"weight_hh_l{capa}{sufijo}")
            b_x = getattr(lstm, f"bias_ih_l{capa}{sufijo}")
            b_h = getattr(lstm, f"bias_hh_l{capa}{sufijo}")
            h = torch.zeros(OCULTA, device=emb.device)
            c = torch.zeros(OCULTA, device=emb.device)
            registro = {"f": [], "i": [], "o": [], "memoria_previa": [], "memoria_retenida": [], "memoria_nueva": [], "memoria": [], "salida": []}
            pasos = range(entrada.shape[0]) if sufijo == "" else reversed(range(entrada.shape[0]))
            por_paso = {}
            for t in pasos:
                gi, gf, gg, go = (w_x @ entrada[t] + b_x + w_h @ h + b_h).chunk(4)
                i, f, o = torch.sigmoid(gi), torch.sigmoid(gf), torch.sigmoid(go)
                previa, retenida, nueva = c.abs().mean(), (f * c).abs().mean(), (i * torch.tanh(gg)).abs().mean()
                c = f * c + i * torch.tanh(gg)
                h = o * torch.tanh(c)
                por_paso[t] = (f, i, o, c, h, previa, retenida, nueva)
            for t in range(entrada.shape[0]):
                f, i, o, c_t, h_t, previa, retenida, nueva = por_paso[t]
                registro["f"].append([round(v, 3) for v in f.tolist()])
                registro["i"].append([round(v, 3) for v in i.tolist()])
                registro["o"].append([round(v, 3) for v in o.tolist()])
                registro["memoria_previa"].append(round(float(previa), 4))
                registro["memoria_retenida"].append(round(float(retenida), 4))
                registro["memoria_nueva"].append(round(float(nueva), 4))
                registro["memoria"].append(round(float(c_t.abs().mean()), 4))
                registro["salida"].append(round(float(h_t.abs().mean()), 4))
            celdas[f"capa{capa + 1}_{nombre}"] = registro
            salidas.append(torch.stack([por_paso[t][4] for t in range(entrada.shape[0])]))
        entrada = torch.cat(salidas, dim=1)
    referencia, _ = lstm(emb.unsqueeze(0))
    return celdas, float((entrada - referencia.squeeze(0)).abs().max())


def main() -> None:
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    azar = random.Random(SEMILLA)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("dispositivo:", dispositivo, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    # 1. Datos
    docs = documentos_biblia() + documentos_coran()
    part = repartir(docs, azar)
    frecuencia = Counter(t for d in part["train"] for t in d["tokens"])
    comunes = [p for p, n in frecuencia.most_common(VOCABULARIO - 2) if n >= 2]
    palabra_id = {p: i + 2 for i, p in enumerate(comunes)}
    id_palabra = {i: p for p, i in palabra_id.items()}
    id_palabra[UNK] = "<unk>"

    seqs = {k: [a_ids(d["tokens"], palabra_id) for d in v] for k, v in part.items()}
    ys = {k: np.array([CLASES[d["corpus"]] for d in v]) for k, v in part.items()}
    reparto = {k: {c: int((ys[k] == i).sum()) for c, i in CLASES.items()} for k in part}
    grupos = {k: len({d["grupo"] for d in v}) for k, v in part.items()}
    print("reparto:", reparto, "grupos:", grupos, "vocabulario:", len(palabra_id) + 2)
    assert not (
        {d["grupo"] for d in part["train"]} & {d["grupo"] for d in part["test"]}
    ), "un grupo cayó en train y test"

    # Validación con recortes fijos, comparables con los de entrenamiento en cada época
    azar_val = random.Random(SEMILLA + 1)
    val_recortes = [recorte(s, azar_val) for s in seqs["val"]]

    # 2. Entrenamiento
    modelo = ClasificadorLSTM(len(palabra_id) + 2).to(dispositivo)
    optim = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4)
    perdida = nn.BCEWithLogitsLoss()
    historia = []
    mejor, mejor_estado, mejor_epoca, sin_mejora = float("inf"), None, 0, 0
    inicio = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoca in range(1, EPOCAS_MAX + 1):
        modelo.train()
        orden = list(range(len(seqs["train"])))
        azar.shuffle(orden)
        suma, aciertos = 0.0, 0
        for i in range(0, len(orden), LOTE):
            idx = orden[i:i + LOTE]
            ids, largos = lote_tensores([recorte(seqs["train"][j], azar) for j in idx], dispositivo)
            y = torch.tensor(ys["train"][idx], dtype=torch.float, device=dispositivo)
            optim.zero_grad(set_to_none=True)
            logit = modelo(ids, largos)
            valor = perdida(logit, y)
            valor.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optim.step()
            suma += float(valor) * len(idx)
            aciertos += int(((logit > 0).float() == y).sum())
        p_val = sigmoide(logits_de(modelo, val_recortes, dispositivo))
        registro = {
            "epoca": epoca,
            "train_loss": suma / len(orden),
            "train_acc": aciertos / len(orden),
            "val_loss": float(log_loss(ys["val"], np.clip(p_val, 1e-7, 1 - 1e-7), labels=[0, 1])),
            "val_acc": float(((p_val >= 0.5).astype(int) == ys["val"]).mean()),
            "segundos": time.perf_counter() - inicio,
        }
        historia.append(registro)
        print("época {epoca:2d} | train {train_loss:.4f} {train_acc:.3f} | val {val_loss:.4f} {val_acc:.3f} | {segundos:.1f} s".format(**registro))
        if registro["val_loss"] < mejor - 1e-4:
            mejor, mejor_epoca, sin_mejora = registro["val_loss"], epoca, 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
        if sin_mejora >= PACIENCIA or registro["segundos"] > TIEMPO_MAX_S:
            break
    tiempo_entrenamiento = time.perf_counter() - inicio
    memoria_pico = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    modelo.load_state_dict(mejor_estado)

    # 3. Evaluación en prueba: documento completo y prefijos de distinta longitud
    y_test = ys["test"]
    p_test = sigmoide(logits_de(modelo, seqs["test"], dispositivo))
    resultado_test = metricas(y_test, p_test)
    print("prueba:", {k: round(v, 4) for k, v in resultado_test.items() if isinstance(v, float)})
    fpr, tpr, _ = roc_curve(y_test, p_test)

    # Línea base: TF-IDF y regresión logística sobre los mismos recortes
    azar_base = random.Random(SEMILLA + 2)
    textos_base, y_base = [], []
    for d, y in zip(part["train"], ys["train"]):
        for _ in range(3):
            largo = azar_base.randint(min(LARGO_MINIMO, len(d["tokens"])), len(d["tokens"]))
            ini = azar_base.randint(0, len(d["tokens"]) - largo)
            textos_base.append(" ".join(d["tokens"][ini:ini + largo]))
            y_base.append(y)
    tfidf = TfidfVectorizer(max_features=VOCABULARIO, sublinear_tf=True, token_pattern=r"\S+")
    base = LogisticRegression(max_iter=2000, random_state=SEMILLA).fit(tfidf.fit_transform(textos_base), y_base)

    por_largo = []
    for largo in LARGOS_EVAL:
        idx = [i for i, s in enumerate(seqs["test"]) if len(s) >= largo]
        y_l = y_test[idx]
        p_l = sigmoide(logits_de(modelo, [seqs["test"][i][:largo] for i in idx], dispositivo))
        tx_l = [" ".join(part["test"][i]["tokens"][:largo]) for i in idx]
        p_base = base.predict_proba(tfidf.transform(tx_l))[:, 1]
        fila = {"largo": largo, "lstm": metricas(y_l, p_l), "base": metricas(y_l, p_base)}
        por_largo.append(fila)
        print(f"L={largo:3d} n={len(idx)} LSTM bal_acc {fila['lstm']['balanced_accuracy']:.3f} logloss {fila['lstm']['log_loss']:.3f} | base bal_acc {fila['base']['balanced_accuracy']:.3f} logloss {fila['base']['log_loss']:.3f}")

    # Calibración con prefijos de 20 palabras: en 200 casi todas las p son 0 o 1
    idx_20 = [i for i, s in enumerate(seqs["test"]) if len(s) >= 20]
    p_20 = sigmoide(logits_de(modelo, [seqs["test"][i][:20] for i in idx_20], dispositivo))
    calib = {"documento": calibracion(y_test, p_test), "prefijo_20": calibracion(y_test[idx_20], p_20)}

    # 4. Shapley por permutaciones sobre documentos completos, balanceados por clase
    azar_shap = random.Random(SEMILLA + 3)
    rng_shap = np.random.default_rng(SEMILLA + 3)
    completos = {c: [i for i, s in enumerate(seqs["test"]) if len(s) == VENTANA and y_test[i] == v] for c, v in CLASES.items()}
    elegidos = []
    for c in CLASES:
        elegidos += azar_shap.sample(completos[c], min(SHAP_POR_CLASE, len(completos[c])))
    t0 = time.perf_counter()
    fi = np.zeros((len(elegidos), VENTANA))
    f_x = np.zeros(len(elegidos))
    errores, mitades_a, mitades_b = [], [], []
    f_base = 0.0
    for fila, i in enumerate(elegidos):
        seq = seqs["test"][i]
        por_palabra, mitades, f_base, f_x[fila] = shapley_por_palabra(modelo, seq, rng_shap, dispositivo)
        cuenta = Counter(seq)
        # Para pintar el texto, el valor de la palabra se reparte entre sus apariciones
        for t, tid in enumerate(seq):
            if tid != UNK:
                fi[fila, t] = por_palabra[tid] / cuenta[tid]
        errores.append(abs(sum(por_palabra.values()) - (f_x[fila] - f_base)))
        mitades_a.append(mitades["a"])
        mitades_b.append(mitades["b"])
    tiempo_shap = time.perf_counter() - t0
    error_aditividad = np.array(errores)
    correlacion_mitades = float(np.corrcoef(np.concatenate(mitades_a), np.concatenate(mitades_b))[0, 1])
    print(f"Shapley: {len(elegidos)} documentos en {tiempo_shap:.1f} s; aditividad máx {error_aditividad.max():.1e} logit; correlación entre mitades {correlacion_mitades:.3f}; f(base) {f_base:.2f}")

    # Importancia global: contribución neta por documento, con clases balanceadas.
    # Se suma con signo antes de promediar porque el ruido de Monte Carlo se cancela así;
    # la media de |phi| crece con la frecuencia y premia a "que" o "la" aunque su efecto sea nulo.
    neto: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(elegidos)))
    apariciones = Counter()
    apariciones_clase: dict[str, Counter] = {c: Counter() for c in CLASES}
    for fila, i in enumerate(elegidos):
        clase = "coran" if y_test[i] == 1 else "biblia"
        for t, tid in enumerate(seqs["test"][i]):
            if tid == UNK:
                continue
            palabra = id_palabra[tid]
            neto[palabra][fila] += fi[fila, t]
            apariciones[palabra] += 1
            apariciones_clase[clase][palabra] += 1

    candidatas = sorted(p for p, n in apariciones.items() if n >= SHAP_MIN_APARICIONES)
    matriz = np.stack([neto[p] for p in candidatas])
    medias = matriz.mean(axis=1)
    azar_np = np.random.default_rng(SEMILLA)
    remuestras = np.stack([
        matriz[:, azar_np.integers(0, len(elegidos), len(elegidos))].mean(axis=1)
        for _ in range(SHAP_REMUESTREOS)
    ])
    corte = np.sort(np.abs(remuestras), axis=1)[:, -TOP][:, None]
    estabilidad = (np.abs(remuestras) >= corte).mean(axis=0)
    ic_bajo, ic_alto = np.percentile(remuestras, [2.5, 97.5], axis=0)
    posicion = {p: j for j, p in enumerate(candidatas)}

    def resumen(palabra: str) -> dict:
        j = posicion[palabra]
        return {
            "palabra": palabra,
            "contribucion": float(medias[j]),
            "ic95": [float(ic_bajo[j]), float(ic_alto[j])],
            "estabilidad": float(estabilidad[j]),
            "efecto_por_aparicion": float(matriz[j].sum() / apariciones[palabra]),
            "n": apariciones[palabra],
            "n_biblia": apariciones_clase["biblia"][palabra],
            "n_coran": apariciones_clase["coran"][palabra],
        }

    top = [resumen(candidatas[j]) for j in np.argsort(-np.abs(medias))[:TOP]]
    top_coran = [resumen(candidatas[j]) for j in np.argsort(-medias)[:TOP]]
    top_biblia = [resumen(candidatas[j]) for j in np.argsort(medias)[:TOP]]
    for r in top:
        print(f"  {r['palabra']:<14} neta {r['contribucion']:+.3f}  IC [{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}]  estab {r['estabilidad']:.2f}  B {r['n_biblia']} C {r['n_coran']}")

    # Ranking sin stopwords de NLTK, con su propia estabilidad dentro de ese subconjunto
    parada = set(json.loads(STOPWORDS.read_text(encoding="utf-8"))["stopwords"])
    contenido = np.array([j for j, p in enumerate(candidatas) if p not in parada])
    rem_contenido = np.abs(remuestras[:, contenido])
    corte_contenido = np.sort(rem_contenido, axis=1)[:, -TOP][:, None]
    estabilidad_contenido = (rem_contenido >= corte_contenido).mean(axis=0)
    top_sin_stopwords = []
    for k in np.argsort(-np.abs(medias[contenido]))[:TOP]:
        fila = resumen(candidatas[contenido[k]])
        fila["estabilidad"] = float(estabilidad_contenido[k])
        top_sin_stopwords.append(fila)
    print("sin stopwords:", [r["palabra"] for r in top_sin_stopwords])

    # Fidelidad: enmascarar un top 20 frente a 20 palabras de frecuencia parecida
    def control_de(lista: list[dict]) -> set[int]:
        conjunto = {palabra_id[r["palabra"]] for r in lista}
        control: set[int] = set()
        for r in lista:
            cercana = min(
                (p for p in candidatas if palabra_id[p] not in conjunto and palabra_id[p] not in control),
                key=lambda p: abs(apariciones[p] - r["n"]),
            )
            control.add(palabra_id[cercana])
        return control

    conjunto_top = {palabra_id[r["palabra"]] for r in top}
    conjunto_contenido = {palabra_id[r["palabra"]] for r in top_sin_stopwords}
    control = control_de(top)
    control_contenido = control_de(top_sin_stopwords)

    def con_mascara(largo: int, mascara: set[int]) -> dict:
        idx = [i for i, s in enumerate(seqs["test"]) if len(s) >= largo]
        secs = [[UNK if t in mascara else t for t in seqs["test"][i][:largo]] for i in idx]
        return metricas(y_test[idx], sigmoide(logits_de(modelo, secs, dispositivo)))

    fidelidad = [
        {"largo": largo, "sin_mascara": con_mascara(largo, set()), "top20": con_mascara(largo, conjunto_top), "control": con_mascara(largo, control),
         "top20_sin_stopwords": con_mascara(largo, conjunto_contenido), "control_sin_stopwords": con_mascara(largo, control_contenido)}
        for largo in (20, 50, 200)
    ]
    for f in fidelidad:
        print(f"L={f['largo']}: bal_acc {f['sin_mascara']['balanced_accuracy']:.3f} -> top20 {f['top20']['balanced_accuracy']:.3f} | control {f['control']['balanced_accuracy']:.3f} | sin stopwords {f['top20_sin_stopwords']['balanced_accuracy']:.3f} | control {f['control_sin_stopwords']['balanced_accuracy']:.3f}")

    # Ejemplos con atribución por palabra: seguros y los menos seguros de la muestra explicada
    p_shap = sigmoide(f_x)
    def ejemplo(fila: int) -> dict:
        i = elegidos[fila]
        palabras = [id_palabra[t] if t != UNK else part["test"][i]["tokens"][k] for k, t in enumerate(seqs["test"][i])]
        return {
            "real": "coran" if y_test[i] == 1 else "biblia",
            "grupo": part["test"][i]["grupo"],
            "p_coran": float(p_shap[fila]),
            "palabras": palabras[:60],
            "shap": [round(float(v), 4) for v in fi[fila, :60]],
        }

    y_shap = y_test[elegidos]
    filas_b = np.where(y_shap == 0)[0]
    filas_c = np.where(y_shap == 1)[0]
    ejemplos = [
        ejemplo(int(filas_b[np.argmin(p_shap[filas_b])])),
        ejemplo(int(filas_c[np.argmax(p_shap[filas_c])])),
        ejemplo(int(filas_b[np.argmax(p_shap[filas_b])])),
        ejemplo(int(filas_c[np.argmin(p_shap[filas_c])])),
    ]

    # 5. Datos de la figura: dos fragmentos de prueba recorridos palabra por palabra.
    # P(Corán) es la del modelo completo sobre el prefijo; las compuertas son de la celda
    # de la capa 1 hacia adelante, la que recibe cada palabra directamente.
    modelo.eval()
    documentos_figura = []
    error_ecuaciones = 0.0
    for clase, valor in CLASES.items():
        candidatos = [i for i, s in enumerate(seqs["test"]) if y_test[i] == valor and len(s) >= PASOS_DIAGRAMA and UNK not in s[:PASOS_DIAGRAMA]]
        mejor_recorrido, mejor_i, mejor_p = -1.0, None, None
        for i in candidatos:
            s = seqs["test"][i][:PASOS_DIAGRAMA]
            p = sigmoide(logits_de(modelo, [s[:k + 1] for k in range(len(s))], dispositivo))
            seguro = p[-1] >= 0.9 if valor == 1 else p[-1] <= 0.1
            if seguro and p.max() - p.min() > mejor_recorrido:
                mejor_recorrido, mejor_i, mejor_p = float(p.max() - p.min()), i, p
        s = seqs["test"][mejor_i][:PASOS_DIAGRAMA]
        with torch.no_grad():
            celdas, error = pasos_lstm(modelo, modelo.embedding(torch.tensor(s, device=dispositivo)))
        error_ecuaciones = max(error_ecuaciones, error)
        documentos_figura.append({
            "real": clase,
            "grupo": part["test"][mejor_i]["grupo"],
            "palabras": [id_palabra[t] for t in s],
            "p_coran": [round(float(v), 4) for v in mejor_p],
            "celda": celdas["capa1_adelante"],
        })
        print(clase, part["test"][mejor_i]["grupo"], " ".join(id_palabra[t] for t in s), np.round(mejor_p, 2).tolist())
    print(f"ecuaciones a mano frente a nn.LSTM, error máximo: {error_ecuaciones:.2e}")

    parametros = {
        "embedding": sum(p.numel() for p in modelo.embedding.parameters()),
        "lstm": sum(p.numel() for p in modelo.lstm.parameters()),
        "densa": sum(p.numel() for p in modelo.densa.parameters()),
        "salida": sum(p.numel() for p in modelo.salida.parameters()),
    }
    anatomia = {"documentos": documentos_figura, "parametros": parametros, "error_ecuaciones": error_ecuaciones}

    resultados = {
        "entorno": {
            "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "semilla": SEMILLA,
            "memoria_pico_mib": memoria_pico,
            "segundos_entrenamiento": tiempo_entrenamiento,
            "segundos_shap": tiempo_shap,
        },
        "datos": {
            "ventana": VENTANA,
            "largo_minimo": LARGO_MINIMO,
            "reparto": reparto,
            "grupos": grupos,
            "vocabulario": len(palabra_id) + 2,
            "largos_por_debajo_de_ventana": {c: sum(1 for d in docs if d["corpus"] == c and len(d["tokens"]) < VENTANA) for c in CLASES},
        },
        "hiperparametros": {
            "embedding": EMBEDDING, "oculta": OCULTA, "capas": CAPAS, "densa": DENSA, "dropout": DROPOUT,
            "lote": LOTE, "lr": LR, "epocas_max": EPOCAS_MAX, "paciencia": PACIENCIA,
        },
        "parametros": parametros,
        "historia": historia,
        "mejor_epoca": mejor_epoca,
        "prueba": resultado_test,
        "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
        "histograma_p": {
            "biblia": np.histogram(p_test[y_test == 0], bins=20, range=(0, 1))[0].tolist(),
            "coran": np.histogram(p_test[y_test == 1], bins=20, range=(0, 1))[0].tolist(),
        },
        "por_largo": por_largo,
        "calibracion": calib,
        "shap": {
            "metodo": "Shapley por permutaciones antitéticas, con cada palabra distinta como jugador y <unk> como referencia",
            "documentos": len(elegidos),
            "permutaciones": 2 * SHAP_PERMUTACIONES,
            "f_base": f_base,
            "error_aditividad_max": float(error_aditividad.max()),
            "correlacion_mitades": correlacion_mitades,
            "min_apariciones": SHAP_MIN_APARICIONES,
            "remuestreos": SHAP_REMUESTREOS,
            "candidatas": len(candidatas),
            "top": top,
            "top_sin_stopwords": top_sin_stopwords,
            "top_coran": top_coran,
            "top_biblia": top_biblia,
        },
        "fidelidad": fidelidad,
        "ejemplos": ejemplos,
        "error_ecuaciones": error_ecuaciones,
    }

    for ruta in (SALIDA_JSON, SALIDA_ANATOMIA, SALIDA_MODELO):
        ruta.parent.mkdir(parents=True, exist_ok=True)
    SALIDA_JSON.write_text(json.dumps(resultados, ensure_ascii=False), encoding="utf-8")
    SALIDA_ANATOMIA.write_text(json.dumps(anatomia, ensure_ascii=False), encoding="utf-8")
    torch.save({"estado": mejor_estado, "palabra_id": palabra_id, "hiper": resultados["hiperparametros"]}, SALIDA_MODELO)
    print(f"entrenamiento {tiempo_entrenamiento:.1f} s | memoria pico {memoria_pico:.0f} MiB | mejor época {mejor_epoca}")
    print("escrito:", SALIDA_JSON.relative_to(RAIZ), SALIDA_ANATOMIA.relative_to(RAIZ), SALIDA_MODELO.relative_to(RAIZ))


if __name__ == "__main__":
    main()
