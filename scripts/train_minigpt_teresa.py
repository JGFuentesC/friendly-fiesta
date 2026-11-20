"""Entrena un mini GPT (decoder-only) desde cero sobre los diálogos de "Teresa".

Corre en el contenedor lab-pytorch del servidor GPU (RTX 3070, PyTorch 2.x,
Python 3.10). También funciona en CPU, pero muy lento; para una prueba rápida
usa --rapido.

Lee:
    data/raw/teresa/teresa.txt      corpus limpio (scripts/descargar_teresa.py)

Escribe:
    data/minigpt_run/teresa_bpe.model, .vocab   tokenizador SentencePiece BPE
    data/minigpt_run/mejor.pt                   pesos con la menor pérdida de validación
    data/minigpt_run/final.pt                   pesos al terminar
    data/minigpt_run/entrenamiento.log          bitácora
    data/processed/minigpt_teresa.json          todo lo que necesitan la SPA, el
                                                cuaderno y el apunte (< 2 MB)

Uso (dentro del contenedor):
    python scripts/train_minigpt_teresa.py                 # corrida completa
    python scripts/train_minigpt_teresa.py --rapido        # 300 pasos, para probar

Arquitectura de referencia: GPT-2 (Radford et al., 2019) a escala de juguete:
6 capas, 384 dimensiones, 6 cabezas, contexto de 256 tokens, pesos atados
entre la tabla de embeddings y la capa de salida.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

RAIZ = Path(__file__).resolve().parents[1]
CORPUS = RAIZ / "data" / "raw" / "teresa" / "teresa.txt"
RUN = RAIZ / "data" / "minigpt_run"
SALIDA = RAIZ / "data" / "processed" / "minigpt_teresa.json"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NL = "<nl>"  # símbolo reservado para el salto de línea entre parlamentos

PROMPT = "Teresa, eres una"
FRASES_ATENCION = [
    "Teresa, yo te amo con toda mi alma",
    "Esa mujer nunca va a ser feliz",
    "Mariano, no me dejes sola",
    "Juro que me las vas a pagar",
]
PREFIJOS = ["Teresa", "yo te", "eres una", "no me", "Mariano", "te lo juro por"]
FRASES_TOKENIZACION = [
    "Teresa, eres una ambiciosa",
    "yo nunca me voy a casar contigo",
    "Arturo de la Barrera",
    "¿por qué me haces esto?",
    "la vecindad",
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def fijar_semilla(semilla: int = SEED) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def bitacora(msg: str) -> None:
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    with open(RUN / "entrenamiento.log", "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def r(x, nd: int = 3):
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


# ---------------------------------------------------------------------------
# Tokenizador
# ---------------------------------------------------------------------------
def entrenar_tokenizador(texto: str, vocab: int) -> spm.SentencePieceProcessor:
    """BPE de SentencePiece entrenado sobre las líneas del corpus."""
    lineas = RUN / "corpus_lineas.txt"
    lineas.write_text("\n".join(l for l in texto.splitlines() if l.strip()), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(lineas), model_prefix=str(RUN / "teresa_bpe"), vocab_size=vocab,
        model_type="bpe", character_coverage=1.0, user_defined_symbols=[NL],
        pad_id=-1, unk_id=0, bos_id=1, eos_id=2, num_threads=4, seed_sentencepiece_size=200000,
        train_extremely_large_corpus=False, minloglevel=2,
    )
    return spm.SentencePieceProcessor(model_file=str(RUN / "teresa_bpe.model"))


def codificar(sp: spm.SentencePieceProcessor, texto: str) -> list[int]:
    return sp.encode(texto.replace("\n", f" {NL} "))


def decodificar(sp: spm.SentencePieceProcessor, ids: list[int]) -> str:
    return re.sub(rf"\s*{NL}\s*", "\n", sp.decode(ids))


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 4000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


class AtencionCausal(nn.Module):
    """Atención multi-cabeza con máscara causal: softmax(QK^T / sqrt(d_k)) V."""

    def __init__(self, c: Config):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proy = nn.Linear(c.n_embd, c.n_embd)
        self.n_head, self.p = c.n_head, c.dropout
        self.drop_res = nn.Dropout(c.dropout)
        self.register_buffer("mascara", torch.tril(torch.ones(c.block_size, c.block_size)).bool(), persistent=False)

    def forward(self, x, devolver_atencion: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        d_k = C // self.n_head
        q, k, v = (t.view(B, T, self.n_head, d_k).transpose(1, 2) for t in (q, k, v))
        if devolver_atencion:
            # Camino explícito: el que se explica en el apunte y se dibuja en la SPA.
            s = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
            s = s.masked_fill(~self.mascara[:T, :T], float("-inf"))
            a = F.softmax(s, dim=-1)
            y = a @ v
        else:
            # Mismo cálculo con el kernel fusionado de PyTorch (más rápido y con menos memoria).
            a = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop_res(self.proy(y)), a


class FFN(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.sube = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.baja = nn.Linear(4 * c.n_embd, c.n_embd)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.baja(F.gelu(self.sube(x), approximate="tanh")))


class Bloque(nn.Module):
    """Bloque pre-LN: x + Atención(LN(x)), luego x + FFN(LN(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.atn, self.ffn = AtencionCausal(c), FFN(c)

    def forward(self, x, devolver_atencion: bool = False):
        h, a = self.atn(self.ln1(x), devolver_atencion)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, a


class MiniGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.bloques = nn.ModuleList([Bloque(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.cabeza = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.cabeza.weight = self.wte.weight  # pesos atados
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proy.weight") or n.endswith("baja.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, objetivos=None, devolver_atencion: bool = False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        atenciones = []
        for b in self.bloques:
            x, a = b(x, devolver_atencion)
            atenciones.append(a)
        logits = self.cabeza(self.ln_f(x))
        perdida = None
        if objetivos is not None:
            perdida = F.cross_entropy(logits.view(-1, logits.size(-1)), objetivos.view(-1))
        return logits, perdida, atenciones

    @torch.no_grad()
    def generar(self, idx, n: int, temperatura=0.8, top_k=50, top_p=None, generador=None):
        modo = self.training
        self.eval()
        for _ in range(n):
            logits, _, _ = self(idx[:, -self.c.block_size:])
            logits = logits[:, -1, :].float() / max(temperatura, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if top_p:
                orden, indices = torch.sort(probs, descending=True)
                acumulada = orden.cumsum(-1)
                orden[(acumulada - orden) > top_p] = 0
                probs = torch.zeros_like(probs).scatter(-1, indices, orden)
                probs = probs / probs.sum(-1, keepdim=True)
            sig = torch.multinomial(probs.cpu(), 1, generator=generador).to(idx.device)
            idx = torch.cat([idx, sig], dim=1)
        self.train(modo)
        return idx


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def particionar(texto: str, sp, frac_val: float = 0.1):
    """Separa capítulos completos (bloques separados por línea en blanco) en train/val."""
    capitulos = [c for c in texto.split("\n\n") if c.strip()]
    rng = random.Random(SEED)
    orden = list(range(len(capitulos)))
    rng.shuffle(orden)
    n_val = max(1, int(len(capitulos) * frac_val))
    val_idx = set(orden[:n_val])
    tr = [codificar(sp, c) + [sp.piece_to_id(NL)] for i, c in enumerate(capitulos) if i not in val_idx]
    va = [codificar(sp, c) + [sp.piece_to_id(NL)] for i, c in enumerate(capitulos) if i in val_idx]
    plano = lambda xs: torch.tensor([t for c in xs for t in c], dtype=torch.long)
    return plano(tr), plano(va), len(capitulos), n_val


def lote(datos: torch.Tensor, B: int, T: int, gen: torch.Generator):
    i = torch.randint(len(datos) - T - 1, (B,), generator=gen)
    x = torch.stack([datos[j:j + T] for j in i])
    y = torch.stack([datos[j + 1:j + T + 1] for j in i])
    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)


@torch.no_grad()
def evaluar(modelo, datos_tr, datos_va, B, T, iters, amp):
    modelo.eval()
    gen = torch.Generator().manual_seed(1234)  # mismos lotes en cada evaluación
    salida = {}
    for nombre, d in (("train", datos_tr), ("val", datos_va)):
        perdidas = []
        for _ in range(iters):
            x, y = lote(d, B, T, gen)
            with amp():
                _, p, _ = modelo(x, y)
            perdidas.append(p.item())
        salida[nombre] = float(np.mean(perdidas))
    modelo.train()
    return salida


def tasa(paso, calentamiento, total, lr_max, lr_min):
    """Calentamiento lineal y luego decaimiento coseno hasta lr_min."""
    if paso < calentamiento:
        return lr_max * (paso + 1) / calentamiento
    if paso >= total:
        return lr_min
    avance = (paso - calentamiento) / max(1, total - calentamiento)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * avance))


def muestra(modelo, sp, semilla=SEED, n=90):
    gen = torch.Generator().manual_seed(semilla)
    ids = torch.tensor([codificar(sp, PROMPT)], device=DEVICE)
    out = modelo.generar(ids, n, temperatura=0.8, top_k=50, generador=gen)
    return decodificar(sp, out[0].tolist())


# ---------------------------------------------------------------------------
# Exportación para la SPA
# ---------------------------------------------------------------------------
def pasos_bpe(sp, palabra: str) -> list[list[str]]:
    """Reproduce la fusión BPE de una frase: de caracteres a tokens, paso a paso.

    SentencePiece BPE asigna a cada pieza un puntaje (mayor = fusión aprendida
    antes). En cada paso se fusiona el par adyacente cuya unión existe en el
    vocabulario con el mayor puntaje.
    """
    puntaje = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.get_piece_size())}
    simbolos = list("▁" + palabra.replace(" ", "▁"))
    historia = [simbolos[:]]
    while True:
        mejor, pos = None, -1
        for i in range(len(simbolos) - 1):
            u = simbolos[i] + simbolos[i + 1]
            if u in puntaje and (mejor is None or puntaje[u] > mejor):
                mejor, pos = puntaje[u], i
        if pos < 0:
            break
        simbolos = simbolos[:pos] + [simbolos[pos] + simbolos[pos + 1]] + simbolos[pos + 2:]
        historia.append(simbolos[:])
    return historia


def exportar(modelo, sp, cfg, hist, muestras, meta, args):
    modelo.eval()
    datos = {"meta": meta, "config": asdict(cfg), "historia": hist, "muestras": muestras, "prompt": PROMPT}

    # Vocabulario completo con puntajes BPE (para el tokenizador en vivo de la SPA).
    datos["vocab"] = [[sp.id_to_piece(i), round(sp.get_score(i), 1)] for i in range(sp.get_piece_size())]
    datos["tokenizacion"] = []
    for f in FRASES_TOKENIZACION:
        ids = codificar(sp, f)
        datos["tokenizacion"].append({"frase": f, "ids": ids, "piezas": [sp.id_to_piece(i) for i in ids],
                                      "pasos": pasos_bpe(sp, f)})

    # Mapas de atención por capa y cabeza.
    datos["atencion"] = []
    with torch.no_grad():
        for f in FRASES_ATENCION:
            ids = codificar(sp, f)
            _, _, atn = modelo(torch.tensor([ids], device=DEVICE), devolver_atencion=True)
            datos["atencion"].append({
                "frase": f, "piezas": [sp.id_to_piece(i) for i in ids],
                "mapas": [r(a[0].float().cpu().numpy(), 3) for a in atn],  # [capa][cabeza][T][T]
            })

        # Distribución del siguiente token: logits crudos de los 50 candidatos más probables.
        datos["siguiente"] = []
        for pre in PREFIJOS:
            ids = codificar(sp, pre)
            logits, _, _ = modelo(torch.tensor([ids], device=DEVICE))
            lg = logits[0, -1].float().cpu()
            probs = F.softmax(lg, -1)
            top = torch.topk(lg, 50)
            datos["siguiente"].append({
                "prefijo": pre, "piezas": [sp.id_to_piece(i) for i in top.indices.tolist()],
                "logits": r(top.values.numpy(), 3), "masa_top50": round(float(probs[top.indices].sum()), 4),
                "entropia_total": round(float(-(probs * torch.log(probs + 1e-12)).sum()), 4),
            })

    # Embeddings de los 300 tokens más frecuentes, en 2D (PCA y t-SNE).
    frec = Counter(meta["_tokens_train"])
    especiales = {sp.piece_to_id(NL), 0, 1, 2}
    top = [i for i, _ in frec.most_common(400) if i not in especiales][:300]
    E = modelo.wte.weight.detach().float().cpu().numpy()[top]
    Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    pca = Ec @ Vt[:2].T
    var = (S ** 2) / (S ** 2).sum()
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=20, init="pca", random_state=SEED, metric="cosine").fit_transform(E)
    except Exception as e:  # sklearn no disponible
        bitacora(f"t-SNE no disponible: {e}")
        tsne = pca
    datos["embeddings"] = {
        "piezas": [sp.id_to_piece(i) for i in top], "frecuencia": [frec[i] for i in top],
        "pca": r(pca, 3), "tsne": r(tsne, 2), "varianza_pca": r(var[:2], 4),
    }
    meta.pop("_tokens_train", None)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    bitacora(f"exportado {SALIDA} ({SALIDA.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--capas", type=int, default=6)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--cabezas", type=int, default=6)
    ap.add_argument("--contexto", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pasos", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--calentamiento", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval_cada", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--paciencia", type=int, default=8, help="evaluaciones sin mejora antes de parar")
    ap.add_argument("--minutos", type=float, default=40.0, help="tope de tiempo de entrenamiento")
    ap.add_argument("--vram", type=float, default=0.36, help="fracción máxima de VRAM para este proceso")
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    if args.rapido:
        args.pasos, args.eval_cada, args.eval_iters, args.calentamiento = 300, 100, 10, 50

    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "entrenamiento.log").write_text("", encoding="utf-8")
    fijar_semilla()
    if DEVICE.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.vram)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    usar_bf16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if usar_bf16 else (lambda: torch.autocast("cpu", enabled=False))

    texto = CORPUS.read_text(encoding="utf-8")
    bitacora(f"corpus: {len(texto):,} caracteres ({len(texto.encode()) / 1e6:.2f} MB) | device={DEVICE}")
    sp = entrenar_tokenizador(texto, args.vocab)
    datos_tr, datos_va, n_cap, n_val = particionar(texto, sp)
    bitacora(f"tokens: train={len(datos_tr):,} val={len(datos_va):,} | capítulos={n_cap} (val={n_val}) | "
             f"caracteres/token={len(texto) / (len(datos_tr) + len(datos_va)):.2f}")

    cfg = Config(sp.get_piece_size(), args.contexto, args.capas, args.cabezas, args.dim, args.dropout)
    modelo = MiniGPT(cfg).to(DEVICE)
    n_param = sum(p.numel() for p in modelo.parameters())
    n_emb = modelo.wte.weight.numel() + modelo.wpe.weight.numel()
    bitacora(f"modelo: {cfg} | parámetros={n_param:,} (sin embeddings: {n_param - n_emb:,})")

    decay = [p for n, p in modelo.named_parameters() if p.dim() >= 2]
    sin_decay = [p for n, p in modelo.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": sin_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=DEVICE.type == "cuda")

    gen = torch.Generator().manual_seed(SEED)
    hist = {"paso": [], "train": [], "lr": [], "eval_paso": [], "eval_train": [], "eval_val": [], "grad_norm": []}
    pasos_muestra = {0, 50, 150, 300, 600, 1000, 1500, 2500, 4000, 6000, 9000, 12000}
    muestras = []
    mejor_val, sin_mejora, paso = float("inf"), 0, 0
    t0 = time.time()
    acumulada = None
    while paso <= args.pasos:
        if paso % args.eval_cada == 0 or paso == args.pasos:
            ev = evaluar(modelo, datos_tr, datos_va, args.batch, args.contexto, args.eval_iters, amp)
            hist["eval_paso"].append(paso)
            hist["eval_train"].append(round(ev["train"], 4))
            hist["eval_val"].append(round(ev["val"], 4))
            mins = (time.time() - t0) / 60
            bitacora(f"paso {paso:6d} | train {ev['train']:.4f} | val {ev['val']:.4f} | ppl val {math.exp(ev['val']):.1f} | "
                     f"{mins:.1f} min | VRAM pico {torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == 'cuda' else 0:.2f} GB")
            if ev["val"] < mejor_val - 1e-3:
                mejor_val, sin_mejora, mejor_paso = ev["val"], 0, paso
                torch.save({"modelo": modelo.state_dict(), "config": asdict(cfg), "paso": paso, "val": ev["val"]}, RUN / "mejor.pt")
            else:
                sin_mejora += 1
            if sin_mejora >= args.paciencia:
                bitacora(f"parada temprana: {args.paciencia} evaluaciones sin mejorar la validación")
                break
            if mins > args.minutos:
                bitacora("tope de tiempo alcanzado")
                break
        if paso in pasos_muestra or (paso % 1000 == 0 and paso > 0):
            texto_m = muestra(modelo, sp)
            ev_v = hist["eval_val"][-1] if hist["eval_paso"] and hist["eval_paso"][-1] == paso else None
            muestras.append({"paso": paso, "val": ev_v, "texto": texto_m})
            bitacora(f"muestra paso {paso}: {texto_m[:120]!r}")
        if paso == args.pasos:
            break

        lr = tasa(paso, args.calentamiento, args.pasos, args.lr, args.lr * 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = lote(datos_tr, args.batch, args.contexto, gen)
        with amp():
            _, perdida, _ = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        gn = torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        pv = perdida.item()
        acumulada = pv if acumulada is None else 0.98 * acumulada + 0.02 * pv
        if paso % 25 == 0:
            hist["paso"].append(paso)
            hist["train"].append(round(acumulada, 4))
            hist["lr"].append(float(f"{lr:.3e}"))
            hist["grad_norm"].append(round(float(gn), 3))
        paso += 1

    minutos = (time.time() - t0) / 60
    torch.save({"modelo": modelo.state_dict(), "config": asdict(cfg), "paso": paso}, RUN / "final.pt")
    texto_final = muestra(modelo, sp)
    muestras.append({"paso": paso, "val": hist["eval_val"][-1], "texto": texto_final, "etiqueta": "final"})

    # Para exportar se usa el mejor modelo según validación.
    ck = torch.load(RUN / "mejor.pt", map_location=DEVICE, weights_only=True)
    modelo.load_state_dict(ck["modelo"])
    muestras.append({"paso": ck["paso"], "val": round(ck["val"], 4), "texto": muestra(modelo, sp), "etiqueta": "mejor"})
    otras = [{"semilla": s, "texto": muestra(modelo, sp, semilla=s, n=120)} for s in (1, 2, 3)]

    meta = {
        "corpus_caracteres": len(texto), "corpus_mb": round(len(texto.encode()) / 1e6, 3),
        "capitulos": n_cap, "capitulos_val": n_val,
        "tokens_train": len(datos_tr), "tokens_val": len(datos_va),
        "parametros": n_param, "parametros_sin_embeddings": n_param - n_emb,
        "pasos": paso, "mejor_paso": ck["paso"], "mejor_val": round(ck["val"], 4),
        "mejor_ppl": round(math.exp(ck["val"]), 2), "minutos": round(minutos, 1),
        "tokens_por_paso": args.batch * args.contexto, "batch": args.batch,
        "lr_max": args.lr, "lr_min": args.lr * 0.1, "calentamiento": args.calentamiento, "weight_decay": args.wd,
        "betas": [0.9, 0.95], "clip": 1.0, "bf16": usar_bf16,
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU",
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE.type == "cuda" else 0,
        "muestras_extra": otras, "rapido": args.rapido,
        "_tokens_train": datos_tr.tolist(),
    }
    bitacora(f"fin: {paso} pasos en {minutos:.1f} min | mejor val {ck['val']:.4f} (ppl {math.exp(ck['val']):.1f}) en paso {ck['paso']}")
    exportar(modelo, sp, cfg, hist, muestras, meta, args)


if __name__ == "__main__":
    main()
