"""Incrusta data/processed/difusion.json dentro de spa/difusion.html.

Corre en local: uv run python scripts/build_difusion_spa.py
La SPA queda como un solo archivo autocontenido (abre con doble clic, sin fetch). El JSON
va dentro de <script id="datos-difusion" type="application/json"> ... </script>; el script
reemplaza ese bloque en el mismo archivo, así que se puede correr cuantas veces haga falta.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON = REPO_ROOT / "data" / "processed" / "difusion.json"
HTML = REPO_ROOT / "spa" / "difusion.html"

datos = json.loads(JSON.read_text())
# Los betas no se usan en la página: basta con alpha_bar
for h in datos["horarios"].values():
    h.pop("beta", None)
carga = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = HTML.read_text()
patron = re.compile(r'(<script id="datos-difusion" type="application/json">)(.*?)(</script>)', re.DOTALL)
if not patron.search(html):
    raise SystemExit("No encuentro el bloque datos-difusion en spa/difusion.html")
html = patron.sub(lambda m: m.group(1) + carga + m.group(3), html, count=1)
HTML.write_text(html)
print(f"escrito {HTML.relative_to(REPO_ROOT)} ({HTML.stat().st_size / 2**20:.2f} MB)")
