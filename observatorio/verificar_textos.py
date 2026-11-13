"""Verificación automática de las reglas de escritura en los textos del observatorio.

- Toda pregunta directa lleva ¿ y ?; un rótulo que empieza con interrogativo acentuado debe ser pregunta completa.
- Sin guion largo (—) ni guion medio (–).

    uv run python -m observatorio.verificar_textos
"""

import re
import sys
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
INTERROGATIVOS = r"(Qué|Quién|Quiénes|Cómo|Dónde|Cuándo|Cuánto|Cuántos|Cuántas|Cuál|Cuáles|Por qué|Con qué|Contra qué)"


def textos_visibles(html):
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
    return [t.strip() for t in re.split(r"<[^>]+>", html) if t.strip()]


def revisar():
    errores = []
    for ruta in sorted(CARPETA.rglob("*")):
        if ruta.suffix not in {".html", ".py"} or ruta.name == "verificar_textos.py":
            continue
        contenido = ruta.read_text(encoding="utf-8")
        for n, linea in enumerate(contenido.splitlines(), 1):
            if "—" in linea or "–" in linea:
                errores.append(f"{ruta.name}:{n}: guion largo o medio")
        if ruta.suffix == ".html":
            for t in textos_visibles(contenido):
                if re.match(INTERROGATIVOS + r"\b", t) or ("?" in t and "¿" not in t):
                    errores.append(f"{ruta.name}: pregunta sin signos completos: {t[:80]!r}")
                if "¿" in t and "?" not in t:
                    errores.append(f"{ruta.name}: falta el signo de cierre: {t[:80]!r}")
    return errores


if __name__ == "__main__":
    errores = revisar()
    print("\n".join(errores) or "textos correctos")
    sys.exit(1 if errores else 0)
