"""Precalcula los conteos sin filtrar de la sección 1 de la SPA.

Los textos de data/raw/textos_nlp/ no se publican en el repositorio. Este JSON guarda
solo frecuencias de palabras y de letras, suficientes para dibujar la sección 1.

Uso:
    uv run python scripts/build_bow_crudo.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ENTRADAS = {
    "biblia": RAIZ / "data" / "raw" / "textos_nlp" / "biblia_es.txt",
    "coran": RAIZ / "data" / "raw" / "textos_nlp" / "coran_es.txt",
}
DESTINO = RAIZ / "data" / "processed" / "bow_crudo.json"
PATRON_PALABRAS = re.compile(r"[a-záéíóúüñ]+")
PATRON_LETRA = re.compile(r"[a-záéíóúüñ]")


def main() -> None:
    datos = {}
    for nombre, ruta in ENTRADAS.items():
        texto = ruta.read_text(encoding="utf-8").lower()
        datos[nombre] = {
            "bow": dict(Counter(PATRON_PALABRAS.findall(texto)).most_common()),
            "grafias": dict(Counter(PATRON_LETRA.findall(texto))),
        }
        print(nombre, sum(datos[nombre]["bow"].values()), len(datos[nombre]["bow"]))
    DESTINO.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    print("escrito", DESTINO.relative_to(RAIZ))


if __name__ == "__main__":
    main()
