"""Verifica las reglas de redacción en el texto visible de los archivos indicados.

Acepta HTML, SVG, JS, LaTeX (.tex, .sty) y notebooks (.ipynb, solo celdas markdown).

Comprueba guiones largos o medios, signos de interrogación sin abrir, títulos que
empiezan con interrogativo acentuado sin ¿ y frases con patrón de antítesis.
Sale con código 1 si encuentra algo.

Uso:
    uv run python scripts/verificar_redaccion_html.py spa/seccion_lstm.js ...
    uv run python scripts/verificar_redaccion_html.py --seccion seccion-lstm spa/demo_nlp_comparador_biblia_coran.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

INTERROGATIVOS = r"(qué|quién|quiénes|cómo|dónde|cuándo|cuánto|cuánta|cuántos|cuántas|cuál|cuáles|por qué|con qué|contra qué)"
TITULOS = re.compile(rf"<(h[1-6]|summary|th|figcaption)[^>]*>\s*{INTERROGATIVOS}\b", re.IGNORECASE)
GUIONES = re.compile(r"[—–]")
ANTITESIS = re.compile(r"\b(no es [^.,:;]{1,40}[,:] ?(es|sino)|no se trata de|no [^.,:;]{1,40}, sino|menos [^.,:;]{1,30} y más|deja de ser [^.,:;]{1,30} para)\b", re.IGNORECASE)


def texto_visible(contenido: str) -> str:
    sin_bloques = re.sub(r"<(style|script)\b.*?</\1>", " ", contenido, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]", " ", sin_bloques, flags=re.DOTALL)


def texto_latex(contenido: str) -> str:
    """Quita comentarios, matemáticas y bloques de código, y deja líneas alineadas."""
    def blanco(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")
    sin_comentarios = re.sub(r"(?<!\\)%.*", "", contenido)
    sin_codigo = re.sub(r"\\begin\{(lstlisting|verbatim|minted)\}.*?\\end\{\1\}", blanco, sin_comentarios, flags=re.DOTALL)
    sin_entornos = re.sub(r"\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}", blanco, sin_codigo, flags=re.DOTALL)
    sin_display = re.sub(r"\\\[.*?\\\]|\$\$.*?\$\$", blanco, sin_entornos, flags=re.DOTALL)
    sin_inline = re.sub(r"\$[^$]*\$|\\lstinline(.)(?:(?!\1).)*\1|\\texttt\{[^}]*\}|\\url\{[^}]*\}", " ", sin_display)
    return sin_inline.replace("?`", "¿").replace("!`", "¡")


def texto_notebook(contenido: str) -> str:
    celdas = json.loads(contenido).get("cells", [])
    markdown = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in celdas if c.get("cell_type") == "markdown"]
    texto = "\n".join(markdown)
    texto = re.sub(r"```.*?```", " ", texto, flags=re.DOTALL)
    texto = re.sub(r"`[^`]*`", " ", texto)
    return re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", texto, flags=re.DOTALL)


def seccion(contenido: str, identificador: str) -> str:
    coincidencia = re.search(rf'<section[^>]*id="{identificador}".*?</section>', contenido, flags=re.DOTALL)
    if not coincidencia:
        sys.exit(f"No existe la sección {identificador}")
    return coincidencia.group(0)


def revisar(nombre: str, contenido: str) -> list[str]:
    fallos = []
    for numero, linea in enumerate(contenido.splitlines(), 1):
        if GUIONES.search(linea):
            fallos.append(f"{nombre}:{numero}: guion largo o medio: {linea.strip()[:90]}")
        if ANTITESIS.search(linea):
            fallos.append(f"{nombre}:{numero}: posible antítesis: {linea.strip()[:90]}")
        if TITULOS.search(linea):
            fallos.append(f"{nombre}:{numero}: título interrogativo sin ¿: {linea.strip()[:90]}")
        if nombre.endswith((".html", ".svg", ".tex", ".ipynb")) and "?" in linea and "¿" not in linea and "://" not in linea:
            fallos.append(f"{nombre}:{numero}: signo ? sin ¿: {linea.strip()[:90]}")
    return fallos


def main() -> None:
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("archivos", nargs="+", type=Path)
    analizador.add_argument("--seccion", help="revisa solo la <section> con este id")
    args = analizador.parse_args()
    fallos = []
    for ruta in args.archivos:
        contenido = ruta.read_text(encoding="utf-8")
        if args.seccion:
            contenido = seccion(contenido, args.seccion)
        if ruta.suffix in {".html", ".svg"}:
            contenido = texto_visible(contenido)
        elif ruta.suffix in {".tex", ".sty"}:
            contenido = texto_latex(contenido)
        elif ruta.suffix == ".ipynb":
            contenido = texto_notebook(contenido)
        fallos += revisar(str(ruta), contenido)
    print("\n".join(fallos) if fallos else f"Redacción correcta en {len(args.archivos)} archivo(s).")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
