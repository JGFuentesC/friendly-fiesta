"""Entrena los modelos de difusión (DDPM) de la Sesión 07 y exporta los datos de la SPA.

Dónde corre: en el contenedor lab-pytorch del servidor GPU (RTX 3070), con el repo
montado en /workspace/friendly-fiesta. También corre en CPU con --rapido (prueba de humo).

    docker exec -w /workspace/friendly-fiesta lab-pytorch python scripts/train_ddpm.py

Qué entrena (todos con horario beta lineal, T = 1000, semilla 42):
  1. Dos DDPM 2D con un MLP: nubes de puntos con forma de "montañas al atardecer"
     (la imagen del video de referencia) y de las letras "UA".
  2. Un DDPM condicional con U-Net pequeña sobre paisajes de pixel art 24x24 generados
     por procedimiento, con tres "prompts": montañas al atardecer, ciudad de noche y
     playa de día. Se entrena con guía sin clasificador (etiqueta nula el 15 % de las veces).
  3. Un DDPM condicional con la misma U-Net sobre MNIST 28x28 (la etiqueta es el dígito).
     MNIST se lee de los archivos idx en data/mnist/MNIST/raw, sin torchvision.

Qué escribe:
  data/difusion_run/*.pt              checkpoints (gitignored)
  data/difusion_run/metricas.json     tiempos y pérdidas finales
  data/processed/difusion.json        datos para la SPA, el notebook y el apunte (< 3 MB)

Con --solo-2d reentrena solo los modelos 2D (minuto y medio) y actualiza "dos_d" en el JSON.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import time
from pathlib import Path

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "data" / "difusion_run"
OUT_JSON = REPO_ROOT / "data" / "processed" / "difusion.json"
MNIST_RAW = REPO_ROOT / "data" / "mnist" / "MNIST" / "raw"

SEED = 42
T = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPTS = ["montañas al atardecer", "ciudad de noche", "playa de día"]


def fijar_semilla(s: int = SEED) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


# ---------------------------------------------------------------------------
# Horarios de ruido
# ---------------------------------------------------------------------------
def horario_lineal(T: int = T, b0: float = 1e-4, b1: float = 0.02) -> np.ndarray:
    return np.linspace(b0, b1, T, dtype=np.float64)


def horario_coseno(T: int = T, s: float = 0.008) -> np.ndarray:
    """Horario de Nichol y Dhariwal (2021): alpha_bar(t) = cos^2(((t/T + s)/(1 + s)) * pi/2)."""
    t = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = 1 - ab[1:] / ab[:-1]
    return np.clip(betas, 0, 0.999)


class Difusion:
    """Constantes del proceso directo e inverso. Índices de tiempo 0..T-1 (t = índice + 1)."""

    def __init__(self, betas: np.ndarray):
        self.T = len(betas)
        b = torch.tensor(betas, dtype=torch.float32)
        a = 1 - b
        ab = torch.cumprod(a, 0)
        ab_prev = torch.cat([torch.ones(1), ab[:-1]])
        self.betas = b.to(DEVICE)
        self.alphas = a.to(DEVICE)
        self.ab = ab.to(DEVICE)
        self.ab_prev = ab_prev.to(DEVICE)
        # Varianza del posterior q(x_{t-1} | x_t, x_0)
        self.var_post = (b * (1 - ab_prev) / (1 - ab)).to(DEVICE)

    def q_sample(self, x0, t, eps):
        ab = self.ab[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    @torch.no_grad()
    def muestrear(self, modelo, forma, registrar, y=None, guia=1.0, recortar=False, gen=None, x_T=None):
        """Muestreo ancestral (algoritmo 2 de Ho et al., 2020) con guía sin clasificador.

        registrar: conjunto de valores t en {0..T}; se guarda x_t al inicio del paso t
        (t = T es el ruido puro, t = 0 la muestra final). Devuelve (x_0, {t: x_t}).
        """
        x = torch.randn(forma, device=DEVICE, generator=gen) if x_T is None else x_T.clone()
        guardados = {}
        if self.T in registrar:
            guardados[self.T] = x.detach().cpu()
        for t in range(self.T, 0, -1):
            i = t - 1
            tt = torch.full((forma[0],), i, device=DEVICE, dtype=torch.long)
            if y is None:
                eps = modelo(x, tt)
            else:
                eps_c = modelo(x, tt, y)
                if guia == 1.0:
                    eps = eps_c
                else:
                    eps_u = modelo(x, tt, torch.full_like(y, modelo.nulo))
                    eps = eps_u + guia * (eps_c - eps_u)
            ab, ab_prev, b, a = self.ab[i], self.ab_prev[i], self.betas[i], self.alphas[i]
            if recortar:
                x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
                media = (ab_prev.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ab_prev) / (1 - ab)) * x
            else:
                media = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 1:
                z = torch.randn(x.shape, device=DEVICE, generator=gen)
                x = media + self.var_post[i].sqrt() * z
            else:
                x = media
            if (t - 1) in registrar:
                guardados[t - 1] = x.detach().cpu()
        return x, guardados


def instantes(n: int, potencia: float, T: int = T) -> list[int]:
    """n instantes de T a 0, más densos cerca de 0 (donde aparece la estructura fina)."""
    u = np.linspace(1, 0, n)
    ts = sorted({int(round(T * v ** potencia)) for v in u}, reverse=True)
    return ts


def emb_tiempo(t, dim):
    """Codificación sinusoidal del paso t (como en los transformers)."""
    mitad = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(mitad, device=t.device) / mitad)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


# ---------------------------------------------------------------------------
# Datos 2D: nubes de puntos dibujadas con trazos
# ---------------------------------------------------------------------------
def muestrear_trazos(trazos, n, rng, grosor=0.06):
    """Cada trazo es ('seg', x0, y0, x1, y1) o ('arco', cx, cy, r, a0, a1). Muestreo por longitud."""
    largos = []
    for tr in trazos:
        if tr[0] == "seg":
            largos.append(math.hypot(tr[3] - tr[1], tr[4] - tr[2]))
        else:
            largos.append(abs(tr[5] - tr[4]) * tr[3])
    p = np.array(largos) / sum(largos)
    idx = rng.choice(len(trazos), size=n, p=p)
    pts = np.zeros((n, 2))
    for k, i in enumerate(idx):
        tr = trazos[i]
        u = rng.random()
        if tr[0] == "seg":
            pts[k] = [tr[1] + u * (tr[3] - tr[1]), tr[2] + u * (tr[4] - tr[2])]
        else:
            ang = tr[4] + u * (tr[5] - tr[4])
            pts[k] = [tr[1] + tr[3] * math.cos(ang), tr[2] + tr[3] * math.sin(ang)]
    return pts + rng.normal(0, grosor, size=pts.shape)


def forma_montanas():
    pi = math.pi
    return [
        # Montaña grande y montaña chica (contorno)
        ("seg", -2.0, -1.0, -0.7, 1.0), ("seg", -0.7, 1.0, 0.2, -0.2),
        ("seg", 0.0, 0.1, 0.8, 0.9), ("seg", 0.8, 0.9, 2.0, -1.0),
        # Nieve en las cumbres
        ("seg", -1.05, 0.45, -0.7, 0.62), ("seg", -0.7, 0.62, -0.42, 0.55),
        ("seg", 0.55, 0.52, 0.8, 0.6), ("seg", 0.8, 0.6, 1.05, 0.52),
        # Sol detrás de las montañas
        ("arco", 0.15, 0.85, 0.55, 0.15 * pi, 0.85 * pi),
        # Horizonte y reflejos en el agua
        ("seg", -2.2, -1.0, 2.2, -1.0),
        ("seg", -0.35, -1.35, 0.65, -1.35), ("seg", -0.15, -1.65, 0.45, -1.65),
    ]


def forma_ua():
    pi = math.pi
    return [
        # U
        ("seg", -1.9, 1.2, -1.9, -0.4), ("seg", -0.5, 1.2, -0.5, -0.4),
        ("arco", -1.2, -0.4, 0.7, pi, 2 * pi),
        # A
        ("seg", 0.35, -1.1, 1.2, 1.2), ("seg", 1.2, 1.2, 2.05, -1.1),
        ("seg", 0.72, -0.15, 1.68, -0.15),
    ]


class MLP2D(nn.Module):
    def __init__(self, oculta=256, demb=64):
        super().__init__()
        self.demb = demb
        self.t_mlp = nn.Sequential(nn.Linear(demb, oculta), nn.SiLU(), nn.Linear(oculta, oculta))
        self.entrada = nn.Linear(2, oculta)
        self.capas = nn.ModuleList([nn.Linear(oculta, oculta) for _ in range(3)])
        self.salida = nn.Linear(oculta, 2)

    def forward(self, x, t):
        e = self.t_mlp(emb_tiempo(t, self.demb))
        h = self.entrada(x) + e
        for c in self.capas:
            h = h + c(F.silu(h)) + e
        return self.salida(F.silu(h))


def entrenar_2d(nombre, trazos, dif, pasos, rng, log):
    fijar_semilla()
    datos = torch.tensor(muestrear_trazos(trazos, 60000, rng), dtype=torch.float32, device=DEVICE)
    modelo = MLP2D().to(DEVICE)
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, pasos)
    perdidas = []
    t0 = time.time()
    for paso in range(pasos):
        x0 = datos[torch.randint(0, len(datos), (2048,), device=DEVICE)]
        t = torch.randint(0, dif.T, (len(x0),), device=DEVICE)
        eps = torch.randn_like(x0)
        perdida = F.mse_loss(modelo(dif.q_sample(x0, t, eps), t), eps)
        opt.zero_grad()
        perdida.backward()
        opt.step()
        sched.step()
        perdidas.append(perdida.item())
        if paso % 2000 == 0:
            log(f"[2D {nombre}] paso {paso:5d}  pérdida {perdida.item():.4f}")
    dur = time.time() - t0
    torch.save(modelo.state_dict(), RUN_DIR / f"mlp2d_{nombre}.pt")
    log(f"[2D {nombre}] listo en {dur:.1f} s, pérdida final (media últimos 500) {np.mean(perdidas[-500:]):.4f}")
    return modelo, datos.cpu().numpy(), perdidas, dur


# ---------------------------------------------------------------------------
# Paisajes de pixel art 24x24 (los "prompts" del video)
# ---------------------------------------------------------------------------
def _mezcla(c0, c1, u):
    return np.asarray(c0) * (1 - u) + np.asarray(c1) * u


def paisaje(clase: int, rng, n: int = 24) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    img = np.zeros((n, n, 3))
    j = lambda c, s=0.06: np.clip(np.asarray(c) + rng.normal(0, s, 3), 0, 1)
    if clase == 0:  # montañas al atardecer
        horizonte = int(rng.integers(16, 19))
        arriba, abajo = j([0.32, 0.18, 0.45]), j([1.0, 0.55, 0.35])
        u = np.clip(yy / horizonte, 0, 1)[..., None]
        img[:] = _mezcla(arriba, abajo, u)
        sx, sy, sr = rng.uniform(6, 18), rng.uniform(7, horizonte - 3), rng.uniform(2.5, 4.0)
        sol = (xx - sx) ** 2 + (yy - sy) ** 2 <= sr ** 2
        img[sol] = j([1.0, 0.93, 0.65], 0.03)
        for _ in range(int(rng.integers(2, 4))):
            px, ph, w = rng.uniform(2, 22), rng.uniform(5, horizonte - 2), rng.uniform(0.8, 1.4)
            dentro = (yy >= ph + np.abs(xx - px) * w) & (yy < horizonte)
            color = j([0.35, 0.25, 0.45] if rng.random() < 0.5 else [0.22, 0.18, 0.32])
            img[dentro] = color
            nieve = dentro & (yy < ph + 2.2)
            img[nieve] = j([0.95, 0.93, 0.98], 0.02)
        agua = yy >= horizonte
        img[agua] = _mezcla(j([0.2, 0.22, 0.5]), j([0.1, 0.12, 0.3]), ((yy - horizonte) / (n - horizonte))[..., None])[agua]
        reflejo = agua & (np.abs(xx - sx) < sr * 0.6) & ((yy.astype(int) % 2) == 0)
        img[reflejo] = j([1.0, 0.8, 0.5], 0.03)
    elif clase == 1:  # ciudad de noche
        u = (yy / n)[..., None]
        img[:] = _mezcla(j([0.03, 0.04, 0.12], 0.02), j([0.12, 0.12, 0.3], 0.03), u)
        estrellas = rng.random((n, n)) < 0.04
        img[estrellas & (yy < 12)] = [0.9, 0.9, 1.0]
        mx, my, mr = rng.uniform(3, 21), rng.uniform(3, 7), rng.uniform(1.6, 2.6)
        img[(xx - mx) ** 2 + (yy - my) ** 2 <= mr ** 2] = j([0.95, 0.95, 0.8], 0.02)
        x = 0
        while x < n:
            w = min(int(rng.integers(3, 6)), n - x)
            h = int(rng.integers(6, 17))
            col = j([0.12, 0.13, 0.2], 0.04)
            img[n - h:, x:x + w] = col
            ventanas = (rng.random((h, w)) < 0.35)
            ventanas[::2, :] = False
            ventanas[:, ::2] = False
            bloque = img[n - h:, x:x + w]
            bloque[ventanas] = j([1.0, 0.85, 0.35], 0.04)
            x += w
    else:  # playa de día
        horizonte = int(rng.integers(11, 15))
        orilla = int(rng.integers(17, 20))
        u = np.clip(yy / horizonte, 0, 1)[..., None]
        img[:] = _mezcla(j([0.35, 0.6, 0.95]), j([0.7, 0.88, 1.0]), u)
        sx, sy, sr = rng.uniform(3, 21), rng.uniform(2.5, 6), rng.uniform(1.8, 2.8)
        img[(xx - sx) ** 2 + (yy - sy) ** 2 <= sr ** 2] = j([1.0, 0.95, 0.4], 0.03)
        for _ in range(int(rng.integers(1, 3))):
            cx, cy = rng.uniform(2, 22), rng.uniform(2, horizonte - 3)
            nube = ((xx - cx) / 3.2) ** 2 + ((yy - cy) / 1.2) ** 2 <= 1
            img[nube] = j([0.97, 0.98, 1.0], 0.02)
        mar = (yy >= horizonte) & (yy < orilla)
        img[mar] = _mezcla(j([0.05, 0.35, 0.6]), j([0.1, 0.6, 0.75]), ((yy - horizonte) / max(orilla - horizonte, 1))[..., None])[mar]
        espuma = (yy == orilla - 1) & (rng.random((n, n)) < 0.6)
        img[espuma] = [0.95, 0.97, 1.0]
        arena = yy >= orilla
        img[arena] = j([0.93, 0.82, 0.58], 0.03)
    return np.clip(img, 0, 1)


def generar_paisajes(n_por_clase, rng):
    X, Y = [], []
    for c in range(3):
        for _ in range(n_por_clase):
            X.append(paisaje(c, rng))
            Y.append(c)
    X = np.stack(X).transpose(0, 3, 1, 2).astype(np.float32) * 2 - 1
    return X, np.array(Y)


# ---------------------------------------------------------------------------
# MNIST desde los archivos idx (sin torchvision)
# ---------------------------------------------------------------------------
def leer_idx(nombre):
    for ruta in [MNIST_RAW / nombre, MNIST_RAW / (nombre + ".gz")]:
        if ruta.exists():
            datos = gzip.open(ruta).read() if ruta.suffix == ".gz" else ruta.read_bytes()
            ndim = datos[3]
            dims = [int.from_bytes(datos[4 + 4 * k: 8 + 4 * k], "big") for k in range(ndim)]
            return np.frombuffer(datos, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)
    raise FileNotFoundError(f"No encuentro {nombre} en {MNIST_RAW}")


# ---------------------------------------------------------------------------
# U-Net pequeña con condición de clase
# ---------------------------------------------------------------------------
class BloqueRes(nn.Module):
    def __init__(self, cin, cout, demb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(demb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.atajo = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.atajo(x)


class Atencion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.att = nn.MultiheadAttention(c, 4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.n(x).flatten(2).transpose(1, 2)
        h, _ = self.att(h, h, h, need_weights=False)
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class UNet(nn.Module):
    def __init__(self, canales_img=1, n_clases=10, c=(48, 96, 144)):
        super().__init__()
        c1, c2, c3 = c
        demb = 4 * c1
        self.c1 = c1
        self.nulo = n_clases
        self.t_mlp = nn.Sequential(nn.Linear(c1, demb), nn.SiLU(), nn.Linear(demb, demb))
        self.clase = nn.Embedding(n_clases + 1, demb)
        self.entrada = nn.Conv2d(canales_img, c1, 3, padding=1)
        self.r1 = BloqueRes(c1, c1, demb)
        self.baja1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.r2 = BloqueRes(c1, c2, demb)
        self.baja2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.r3 = BloqueRes(c2, c3, demb)
        self.m1 = BloqueRes(c3, c3, demb)
        self.att = Atencion(c3)
        self.m2 = BloqueRes(c3, c3, demb)
        self.r4 = BloqueRes(2 * c3, c3, demb)
        self.sube2 = nn.Conv2d(c3, c3, 3, padding=1)
        self.r5 = BloqueRes(c3 + c2, c2, demb)
        self.sube1 = nn.Conv2d(c2, c2, 3, padding=1)
        self.r6 = BloqueRes(c2 + c1, c1, demb)
        self.n_sal = nn.GroupNorm(8, c1)
        self.salida = nn.Conv2d(c1, canales_img, 3, padding=1)

    def forward(self, x, t, y):
        e = self.t_mlp(emb_tiempo(t, self.c1)) + self.clase(y)
        h1 = self.r1(self.entrada(x), e)
        h2 = self.r2(self.baja1(h1), e)
        h3 = self.r3(self.baja2(h2), e)
        m = self.m2(self.att(self.m1(h3, e)), e)
        u = self.r4(torch.cat([m, h3], 1), e)
        u = self.sube2(F.interpolate(u, size=h2.shape[-2:], mode="nearest"))
        u = self.r5(torch.cat([u, h2], 1), e)
        u = self.sube1(F.interpolate(u, size=h1.shape[-2:], mode="nearest"))
        u = self.r6(torch.cat([u, h1], 1), e)
        return self.salida(F.silu(self.n_sal(u)))


def entrenar_unet(nombre, X, Y, n_clases, dif, epocas, log, p_nulo=0.15, lote=128):
    fijar_semilla()
    modelo = UNet(X.shape[1], n_clases).to(DEVICE)
    ema = UNet(X.shape[1], n_clases).to(DEVICE)
    ema.load_state_dict(modelo.state_dict())
    ema.requires_grad_(False)
    n_param = sum(p.numel() for p in modelo.parameters())
    log(f"[{nombre}] U-Net con {n_param:,} parámetros, {len(X)} imágenes, {epocas} épocas")
    # Los datos viven en CPU: la GPU se comparte y solo sube cada lote
    Xg = torch.from_numpy(np.ascontiguousarray(X))
    Yg = torch.from_numpy(np.asarray(Y, dtype=np.int64))
    pasos_total = epocas * (len(X) // lote)
    opt = torch.optim.AdamW(modelo.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=pasos_total, pct_start=0.05)
    perdidas = []
    t0 = time.time()
    paso = 0
    for ep in range(epocas):
        perm = torch.randperm(len(X))
        for k in range(len(X) // lote):
            idx = perm[k * lote:(k + 1) * lote]
            x0, y = Xg[idx].to(DEVICE, non_blocking=True), Yg[idx].to(DEVICE)
            y[torch.rand(len(y), device=DEVICE) < p_nulo] = n_clases
            t = torch.randint(0, dif.T, (len(x0),), device=DEVICE)
            eps = torch.randn_like(x0)
            # bfloat16 en la GPU: la tarjeta se comparte con Ollama y Stable Diffusion (~1 GB libre)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                pred = modelo(dif.q_sample(x0, t, eps), t, y)
            perdida = F.mse_loss(pred.float(), eps)
            opt.zero_grad()
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), modelo.parameters()):
                    pe.mul_(0.999).add_(pm, alpha=0.001)
            perdidas.append(perdida.item())
            paso += 1
        log(f"[{nombre}] época {ep + 1}/{epocas}  pérdida media {np.mean(perdidas[-(len(X) // lote):]):.4f}  ({time.time() - t0:.0f} s)")
    for be, bm in zip(ema.buffers(), modelo.buffers()):
        be.copy_(bm)
    dur = time.time() - t0
    torch.save({"modelo": modelo.state_dict(), "ema": ema.state_dict()}, RUN_DIR / f"unet_{nombre}.pt")
    ema.eval()
    return ema, perdidas, dur, n_param


# ---------------------------------------------------------------------------
# Utilidades de exportación
# ---------------------------------------------------------------------------
def b64_int16(arr, escala):
    a = np.clip(np.round(np.asarray(arr) * escala), -32767, 32767).astype("<i2")
    return base64.b64encode(a.tobytes()).decode()


def a_uint8(x):
    """Tensor en [-1, 1] -> uint8 en [0, 255]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()


def b64_uint8(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode()


def curva(perdidas, n=200):
    p = np.asarray(perdidas)
    tam = max(1, len(p) // n)
    p = p[: tam * (len(p) // tam)].reshape(-1, tam).mean(1)
    return {"paso": [int((k + 0.5) * tam) for k in range(len(p))], "perdida": [round(float(v), 5) for v in p]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: pocos pasos")
    ap.add_argument("--solo-2d", action="store_true", help="reentrena solo los modelos 2D y actualiza el JSON existente")
    args = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    log(f"Dispositivo: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    t_inicio = time.time()
    fijar_semilla()
    rng = np.random.default_rng(SEED)
    lin, cos = horario_lineal(), horario_coseno()
    dif = Difusion(lin)
    salida = {
        "meta": {"T": T, "semilla": SEED, "dispositivo": str(DEVICE), "prompts": PROMPTS},
        "horarios": {},
    }
    for nombre, b in [("lineal", lin), ("coseno", cos)]:
        ab = np.cumprod(1 - b)
        salida["horarios"][nombre] = {
            "beta": [round(float(v), 6) for v in b],
            "alpha_bar": [round(float(v), 6) for v in ab],
        }

    # ------------------------------------------------------------------ 2D
    pasos_2d = 300 if args.rapido else 16000
    salida["dos_d"] = {}
    for nombre, trazos in [("montanas", forma_montanas()), ("ua", forma_ua())]:
        modelo, datos, perdidas, dur = entrenar_2d(nombre, trazos, dif, pasos_2d, rng, log)
        modelo.eval()
        n_part = 2500
        reg = instantes(60, 1.6)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        x_fin, guard = dif.muestrear(modelo, (n_part, 2), set(reg), gen=gen)
        tray = np.stack([guard[t].numpy() for t in reg])  # (60, n, 2)
        # Campo de ruido predicho en una rejilla, para varios t
        g = np.linspace(-3.2, 3.2, 25)
        gx, gy = np.meshgrid(g, g)
        rejilla = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32, device=DEVICE)
        ts_campo = [10, 50, 100, 200, 350, 500, 700, 900]
        campo = []
        with torch.no_grad():
            for t in ts_campo:
                tt = torch.full((len(rejilla),), t - 1, device=DEVICE, dtype=torch.long)
                campo.append(modelo(rejilla, tt).cpu().numpy())
        # Pérdida en función de t (sobre datos reales)
        with torch.no_grad():
            x0 = torch.tensor(datos[:4096], device=DEVICE)
            ts_p = list(range(1, T + 1, 25))
            perd_t = []
            for t in ts_p:
                tt = torch.full((len(x0),), t - 1, device=DEVICE, dtype=torch.long)
                eps = torch.randn_like(x0)
                perd_t.append(round(F.mse_loss(modelo(dif.q_sample(x0, tt, eps), tt), eps).item(), 5))
        salida["dos_d"][nombre] = {
            "datos": b64_int16(datos[:2400], 1000),
            "n_datos": 2400,
            "instantes": reg,
            "n_particulas": n_part,
            "trayectorias": b64_int16(tray, 1000),
            "rejilla": [round(float(v), 4) for v in g],
            "t_campo": ts_campo,
            "campo_eps": b64_int16(np.stack(campo), 1000),
            "perdida_vs_t": {"t": ts_p, "mse": perd_t},
            "curva": curva(perdidas),
            "segundos": round(dur, 1),
        }
        log(f"[2D {nombre}] exportadas {n_part} trayectorias en {len(reg)} instantes")

    if args.solo_2d:
        previo = json.loads(OUT_JSON.read_text())
        previo["dos_d"] = salida["dos_d"]
        OUT_JSON.write_text(json.dumps(previo, ensure_ascii=False, separators=(",", ":")))
        log(f"Actualizado {OUT_JSON} ({OUT_JSON.stat().st_size / 2**20:.2f} MB) solo con los modelos 2D")
        return

    # ------------------------------------------------------------------ Paisajes
    n_pc = 300 if args.rapido else 12000
    Xp, Yp = generar_paisajes(n_pc, rng)
    log(f"Paisajes generados: {Xp.shape}")
    ep_p = 1 if args.rapido else 14
    ema_p, perd_p, dur_p, npar_p = entrenar_unet("paisajes", Xp, Yp, 3, dif, ep_p, log)
    reg_img = instantes(20, 2.0)
    n_m = 3
    y = torch.tensor(sum([[c] * n_m for c in range(3)], []), device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    _, guard = dif.muestrear(ema_p, (len(y), 3, 24, 24), set(reg_img), y=y, guia=3.0, recortar=True, gen=gen)
    tira_p = np.stack([a_uint8(guard[t]) for t in reg_img], 1).transpose(0, 1, 3, 4, 2)  # (9, 20, 24, 24, 3)
    # Guía: mismo ruido inicial, escalas 0, 1 y 3
    gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    xT = torch.randn((3, 3, 24, 24), device=DEVICE, generator=gen)
    guia_p = []
    for w in [0.0, 1.0, 3.0]:
        g = torch.Generator(device=DEVICE).manual_seed(SEED + 2)
        x, _ = dif.muestrear(ema_p, (3, 3, 24, 24), set(), y=torch.arange(3, device=DEVICE), guia=w, recortar=True, gen=g, x_T=xT)
        guia_p.append(a_uint8(x.cpu()).transpose(0, 2, 3, 1))
    ejemplos = ((Xp[np.concatenate([np.arange(4) + c * n_pc for c in range(3)])] + 1) * 127.5).round().astype(np.uint8).transpose(0, 2, 3, 1)
    salida["paisajes"] = {
        "lado": 24, "canales": 3, "prompts": PROMPTS, "por_prompt": n_m,
        "instantes": reg_img,
        "tira": b64_uint8(tira_p), "forma_tira": list(tira_p.shape),
        "guia_escalas": [0.0, 1.0, 3.0], "guia": b64_uint8(np.stack(guia_p)), "forma_guia": [3, 3, 24, 24, 3],
        "ejemplos": b64_uint8(ejemplos), "forma_ejemplos": list(ejemplos.shape),
        "curva": curva(perd_p), "segundos": round(dur_p, 1), "parametros": npar_p,
        "n_entrenamiento": int(len(Xp)),
    }

    del ema_p
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ MNIST
    Xm = leer_idx("train-images-idx3-ubyte").astype(np.float32) / 127.5 - 1
    Ym = leer_idx("train-labels-idx1-ubyte").astype(np.int64)
    Xt = leer_idx("t10k-images-idx3-ubyte").astype(np.float32) / 127.5 - 1
    if args.rapido:
        Xm, Ym = Xm[:2000], Ym[:2000]
    Xm = Xm[:, None]
    ep_m = 1 if args.rapido else 12
    ema_m, perd_m, dur_m, npar_m = entrenar_unet("mnist", Xm, Ym, 10, dif, ep_m, log)
    y = torch.tensor(list(range(10)) + list(range(6)), device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    _, guard = dif.muestrear(ema_m, (16, 1, 28, 28), set(reg_img), y=y, guia=2.0, recortar=True, gen=gen)
    tira_m = np.stack([a_uint8(guard[t])[:, 0] for t in reg_img], 1)  # (16, 20, 28, 28)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    xT = torch.randn((10, 1, 28, 28), device=DEVICE, generator=gen)
    guia_m = []
    for w in [0.0, 1.0, 3.0]:
        g = torch.Generator(device=DEVICE).manual_seed(SEED + 2)
        x, _ = dif.muestrear(ema_m, (10, 1, 28, 28), set(), y=torch.arange(10, device=DEVICE), guia=w, recortar=True, gen=g, x_T=xT)
        guia_m.append(a_uint8(x.cpu())[:, 0])
    # La pérdida vista por dentro: un dígito de prueba, ruido verdadero vs predicho
    x0 = torch.tensor(Xt[:1][:, None], device=DEVICE)
    y0 = torch.tensor(leer_idx("t10k-labels-idx1-ubyte")[:1].astype(np.int64), device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    eps = torch.randn(x0.shape, device=DEVICE, generator=g)
    ts_l = [20, 100, 250, 500, 750, 950]
    filas, mses = [], []
    with torch.no_grad():
        for t in ts_l:
            tt = torch.full((1,), t - 1, device=DEVICE, dtype=torch.long)
            xt = dif.q_sample(x0, tt, eps)
            ep = ema_m(xt, tt, y0)
            ab = dif.ab[t - 1]
            x0h = (xt - (1 - ab).sqrt() * ep) / ab.sqrt()
            mses.append(round(F.mse_loss(ep, eps).item(), 5))
            filas.append(np.stack([a_uint8(xt.cpu())[0, 0], a_uint8((eps / 3).cpu())[0, 0], a_uint8((ep / 3).cpu())[0, 0], a_uint8(((ep - eps) / 3).cpu())[0, 0], a_uint8(x0h.cpu())[0, 0]]))
    # Pérdida promedio en función de t sobre 512 dígitos de prueba (lotes de 128 por la VRAM)
    etiquetas_prueba = leer_idx("t10k-labels-idx1-ubyte").astype(np.int64)
    with torch.no_grad():
        ts_p = list(range(1, T + 1, 25))
        perd_t = []
        for t in ts_p:
            acum = 0.0
            for k in range(4):
                xb = torch.tensor(Xt[k * 128:(k + 1) * 128][:, None], device=DEVICE)
                yb = torch.tensor(etiquetas_prueba[k * 128:(k + 1) * 128], device=DEVICE)
                tt = torch.full((len(xb),), t - 1, device=DEVICE, dtype=torch.long)
                e = torch.randn_like(xb)
                acum += F.mse_loss(ema_m(dif.q_sample(xb, tt, e), tt, yb), e).item() / 4
            perd_t.append(round(acum, 5))
    salida["mnist"] = {
        "lado": 28, "etiquetas": y.tolist(), "instantes": reg_img, "guia_tira": 2.0,
        "tira": b64_uint8(tira_m), "forma_tira": list(tira_m.shape),
        "guia_escalas": [0.0, 1.0, 3.0], "guia": b64_uint8(np.stack(guia_m)), "forma_guia": [3, 10, 28, 28],
        "perdida_detalle": {"t": ts_l, "mse": mses, "etiqueta": int(y0.item()),
                            "columnas": ["x_t", "ruido verdadero", "ruido predicho", "diferencia", "x0 estimado"],
                            "imagenes": b64_uint8(np.stack(filas)), "forma": [len(ts_l), 5, 28, 28]},
        "perdida_vs_t": {"t": ts_p, "mse": perd_t},
        "curva": curva(perd_m), "segundos": round(dur_m, 1), "parametros": npar_m,
    }

    total = time.time() - t_inicio
    salida["meta"]["segundos_total"] = round(total, 1)
    if DEVICE.type == "cuda":
        salida["meta"]["vram_pico_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        salida["meta"]["gpu"] = torch.cuda.get_device_name(0)
    OUT_JSON.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")))
    metricas = {
        "segundos_total": round(total, 1),
        "vram_pico_mb": salida["meta"].get("vram_pico_mb"),
        "perdida_final": {
            "2d_montanas": float(np.mean(salida["dos_d"]["montanas"]["curva"]["perdida"][-10:])),
            "2d_ua": float(np.mean(salida["dos_d"]["ua"]["curva"]["perdida"][-10:])),
            "paisajes": float(np.mean(perd_p[-500:])),
            "mnist": float(np.mean(perd_m[-500:])),
        },
        "segundos": {"paisajes": round(dur_p, 1), "mnist": round(dur_m, 1),
                     "2d_montanas": salida["dos_d"]["montanas"]["segundos"], "2d_ua": salida["dos_d"]["ua"]["segundos"]},
    }
    (RUN_DIR / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2))
    log(f"Escrito {OUT_JSON} ({OUT_JSON.stat().st_size / 2**20:.2f} MB) en {total / 60:.1f} min")
    log(json.dumps(metricas, ensure_ascii=False))


if __name__ == "__main__":
    main()
