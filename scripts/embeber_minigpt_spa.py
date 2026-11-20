"""Embebe data/processed/minigpt_teresa.json dentro de spa/minigpt_teresa.html.

Corre en local (macOS) con el entorno uv. Reemplaza el contenido del bloque
<script id="datos-minigpt" type="application/json"> de la SPA por el JSON del
entrenamiento, de modo que la página sea un solo archivo que abre con doble clic
(sin fetch). Se queda con lo que la SPA usa y redondea para que pese poco.

Uso:
    uv run python scripts/embeber_minigpt_spa.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "processed" / "minigpt_teresa.json"
SPA = RAIZ / "spa" / "minigpt_teresa.html"
BLOQUE = re.compile(r'(<script id="datos-minigpt" type="application/json">)(.*?)(</script>)', re.DOTALL)


def main() -> None:
    d = json.loads(DATOS.read_text(encoding="utf-8"))
    d["meta"].pop("muestras_extra", None)
    texto = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = SPA.read_text(encoding="utf-8")
    if not BLOQUE.search(html):
        raise SystemExit("No encontré el bloque datos-minigpt en la SPA")
    html = BLOQUE.sub(lambda m: m.group(1) + texto + m.group(3), html, count=1)
    SPA.write_text(html, encoding="utf-8")
    print(f"{SPA.relative_to(RAIZ)}: {len(html.encode()) / 1e6:.2f} MB (datos {len(texto.encode()) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
