"""Descarga subtítulos en español de la telenovela "Teresa" (Televisa, 2010) y los limpia.

Corre en local (macOS) con el yt-dlp del entorno uv. Busca capítulos y escenas en
YouTube con varias consultas ytsearch, baja SOLO subtítulos en español (manuales o
automáticos, nunca el video) y los convierte a diálogo plano:

- quita la cabecera WEBVTT, las marcas de tiempo y las etiquetas <c>, <00:00:01.000>;
- elimina los duplicados que produce el solapamiento de los subtítulos automáticos
  (cada cue repite la línea anterior y agrega la nueva);
- descarta anotaciones como [Música], [Aplausos] o [risas].

Escribe:
    data/raw/teresa/vtt/<id>.<lang>.vtt   subtítulos crudos (uno por video)
    data/raw/teresa/videos.json           metadatos de los videos usados
    data/raw/teresa/teresa.txt            corpus limpio, un capítulo tras otro

Uso:
    uv run python scripts/descargar_teresa.py               # busca y descarga
    uv run python scripts/descargar_teresa.py --solo-limpiar  # rehace teresa.txt

El corpus queda gitignored (data/raw/teresa/): son diálogos con derechos de Televisa
y solo se usan para entrenar un modelo didáctico.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "raw" / "teresa"
VTT = DESTINO / "vtt"
YTDLP = [sys.executable, "-m", "yt_dlp"]

CONSULTAS = [
    "Teresa capítulo completo Angelique Boyer",
    "Teresa telenovela capitulo Sebastian Rulli Angelique Boyer",
    "Teresa Novelas Completas en Español Latino Angelique Boyer",
    "Teresa tlnovelas capítulos",
    "Teresa tlnovelas Chapters",
    "Teresa Televisa escena Aurora Mariano",
    "Teresa Televisa Arturo de la Barrera escena",
    "Teresa 2010 telenovela capitulo",
    "Teresa Juleta Paulina Aída escena tlnovelas",
    "Teresa Episodes tlnovelas",
]
POR_CONSULTA = 60
# Títulos que no son diálogo de la telenovela: análisis, resúmenes, canciones.
EXCLUIR = ("psicóloga", "analiza", "resumen", "catarsis", "top 5", "tema de", "finales alternativos", "who was right")
DURACION_MINIMA = 240  # segundos; descarta cortos y promocionales

ANOTACION = re.compile(r"\[[^\]]*\]|\([^)]*\)")
ETIQUETA = re.compile(r"<[^>]+>")
TIEMPO = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} -->")


def buscar() -> list[dict]:
    """Devuelve videos únicos cuyo título menciona Teresa, con duración suficiente."""
    vistos: dict[str, dict] = {}
    for consulta in CONSULTAS:
        salida = subprocess.run(
            [*YTDLP, "--flat-playlist", "--print", "%(id)s\t%(title)s\t%(duration)s\t%(channel)s",
             f"ytsearch{POR_CONSULTA}:{consulta}"],
            capture_output=True, text=True, check=False,
        ).stdout
        for linea in salida.splitlines():
            partes = linea.split("\t")
            if len(partes) != 4:
                continue
            vid, titulo, duracion, canal = partes
            try:
                segundos = float(duracion)
            except ValueError:
                continue
            bajo = titulo.lower()
            if "teresa" not in bajo or segundos < DURACION_MINIMA or any(e in bajo for e in EXCLUIR):
                continue
            vistos.setdefault(vid, {"id": vid, "titulo": titulo, "duracion": segundos, "canal": canal})
        print(f"{consulta!r}: {len(vistos)} videos acumulados", flush=True)
    return list(vistos.values())


def descargar(videos: list[dict]) -> None:
    VTT.mkdir(parents=True, exist_ok=True)
    pendientes = [v["id"] for v in videos if not list(VTT.glob(f"{v['id']}.*.vtt"))]
    print(f"descargando subtítulos de {len(pendientes)} videos", flush=True)
    for i in range(0, len(pendientes), 20):
        lote = pendientes[i:i + 20]
        subprocess.run(
            [*YTDLP, "--write-auto-subs", "--write-subs", "--sub-langs", "es.*",
             "--skip-download", "--sub-format", "vtt", "--ignore-errors", "--no-warnings",
             "-o", str(VTT / "%(id)s.%(ext)s"), *[f"https://www.youtube.com/watch?v={v}" for v in lote]],
            check=False, stdout=subprocess.DEVNULL,
        )
        print(f"  lote {i // 20 + 1}: {len(list(VTT.glob('*.vtt')))} archivos vtt", flush=True)


def limpiar_vtt(texto: str) -> list[str]:
    """Convierte un VTT (manual o automático) en líneas de diálogo sin repeticiones."""
    automatico = "<c>" in texto
    lineas: list[str] = []
    for bloque in re.split(r"\n\s*\n", texto):
        filas = bloque.splitlines()
        if not filas or not any(TIEMPO.match(f) for f in filas):
            continue
        cuerpo = [f for f in filas if not TIEMPO.match(f)]
        if automatico:
            # En los automáticos solo las líneas con marcas internas traen palabras nuevas.
            cuerpo = [f for f in cuerpo if "<c>" in f or re.search(r"<\d{2}:", f)]
        for fila in cuerpo:
            limpia = ANOTACION.sub(" ", ETIQUETA.sub("", fila))
            limpia = re.sub(r"\s+", " ", limpia).replace("&nbsp;", " ").strip(" -")
            if limpia and (not lineas or lineas[-1] != limpia):
                lineas.append(limpia)
    return lineas


def elegir_archivo(vid: str) -> Path | None:
    """Prefiere subtítulos manuales; luego el automático original en español."""
    candidatos = sorted(VTT.glob(f"{vid}.*.vtt"))
    for preferido in ("es-419", "es-MX", "es-ES", "es-orig", "es"):
        for c in candidatos:
            if c.name == f"{vid}.{preferido}.vtt":
                return c
    return candidatos[0] if candidatos else None


def tejas(texto: str, n: int = 8) -> set[int]:
    """Conjunto de n-gramas de palabras (hash) para medir solapamiento entre videos."""
    palabras = re.findall(r"\w+", texto.lower())
    return {hash(" ".join(palabras[i:i + n])) for i in range(len(palabras) - n + 1)}


def construir_corpus(videos: list[dict]) -> None:
    """Limpia cada VTT y descarta videos que repiten un capítulo ya incluido.

    Las escenas sueltas son recortes de los capítulos completos: si más de la
    mitad de sus 8-gramas ya aparecen en el corpus, el video se omite. Así la
    partición de validación no hereda frases idénticas del entrenamiento.
    """
    capitulos, usados, vistas = [], [], set()
    omitidos = 0
    videos = [v for v in videos if not any(e in v["titulo"].lower() for e in EXCLUIR)]
    for v in sorted(videos, key=lambda v: -v["duracion"]):
        archivo = elegir_archivo(v["id"])
        if archivo is None:
            continue
        lineas = limpiar_vtt(archivo.read_text(encoding="utf-8", errors="ignore"))
        if len(lineas) < 30:
            continue
        texto = "\n".join(lineas)
        propias = tejas(texto)
        if propias and len(propias & vistas) / len(propias) > 0.5:
            omitidos += 1
            continue
        vistas |= propias
        capitulos.append(texto)
        usados.append({**v, "archivo": archivo.name, "caracteres": len(texto)})
    print(f"omitidos por solapamiento con capítulos ya incluidos: {omitidos}")
    corpus = "\n\n".join(capitulos) + "\n"
    (DESTINO / "teresa.txt").write_text(corpus, encoding="utf-8")
    (DESTINO / "videos.json").write_text(json.dumps(usados, ensure_ascii=False, indent=1), encoding="utf-8")
    mb = len(corpus.encode("utf-8")) / 1e6
    print(f"corpus: {len(usados)} videos, {len(corpus):,} caracteres, {mb:.2f} MB -> {DESTINO / 'teresa.txt'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solo-limpiar", action="store_true", help="no busca ni descarga; rehace teresa.txt")
    args = p.parse_args()
    DESTINO.mkdir(parents=True, exist_ok=True)
    indice = DESTINO / "busqueda.json"
    if args.solo_limpiar and indice.exists():
        videos = json.loads(indice.read_text(encoding="utf-8"))
    else:
        videos = buscar()
        indice.write_text(json.dumps(videos, ensure_ascii=False, indent=1), encoding="utf-8")
        descargar(videos)
    construir_corpus(videos)


if __name__ == "__main__":
    main()
