"""Incrusta data/processed/traza_agente.json dentro de spa/agente_adk.html.

La SPA es un solo archivo que se abre con doble clic, así que no puede hacer
fetch de la traza: el JSON vive en <script id="traza" type="application/json">.
Este script reemplaza ese bloque con la traza actual (los pensamientos largos se
recortan para aligerar la página) y después revisa la redacción del HTML.

Uso:
    uv run python scripts/build_spa_agente.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAZA = REPO_ROOT / "data" / "processed" / "traza_agente.json"
SPA = REPO_ROOT / "spa" / "agente_adk.html"
MAX_PENSAMIENTO = 1500


def recortar(datos: dict) -> dict:
    """Recorta los pensamientos del modelo a MAX_PENSAMIENTO caracteres."""
    def eventos(lista):
        for e in lista:
            for p in e["partes"]:
                if p["tipo"] == "pensamiento" and len(p["texto"]) > MAX_PENSAMIENTO:
                    p["texto"] = p["texto"][:MAX_PENSAMIENTO].rstrip() + " [...]"
    for t in datos["turnos"]:
        eventos(t["eventos"])
    eventos(datos["comparacion"]["con_herramientas"]["eventos"])
    eventos(datos["comparacion"]["sin_herramientas"]["eventos"])
    return datos


def main() -> None:
    datos = recortar(json.loads(TRAZA.read_text(encoding="utf-8")))
    texto = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = SPA.read_text(encoding="utf-8")
    patron = re.compile(r'(<script id="traza" type="application/json">).*?(</script>)', re.DOTALL)
    if not patron.search(html):
        sys.exit("No encontré el bloque <script id=\"traza\"> en la SPA")
    html = patron.sub(lambda m: m.group(1) + texto + m.group(2), html, count=1)
    SPA.write_text(html, encoding="utf-8")
    print(f"Traza incrustada en {SPA.relative_to(REPO_ROOT)} ({len(html) / 1024:.0f} KB)")
    verificador = REPO_ROOT / "scripts" / "verificar_redaccion_html.py"
    sys.exit(subprocess.call([sys.executable, str(verificador), str(SPA)]))


if __name__ == "__main__":
    main()
