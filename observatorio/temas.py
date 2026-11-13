"""Top 3 de temas por corpus de comentarios con gemma4 en el servidor de Ollama, y unificación de nombres entre meses.

gemma4 (9.6 GB) no cabe completo en los 8 GB de la RTX 3070, así que Ollama reparte capas entre GPU y CPU.
Las llamadas se hacen en serie: varias en paralelo multiplican la caché de contexto y vuelven más lento cada lote.
"""

import json
import logging
import random
import urllib.request

from .bd import conexion
from .config import MODELO_TEMAS, OLLAMA

log = logging.getLogger("observatorio")
MUESTRA = 300
ESQUEMA_TEMAS = {
    "type": "object",
    "properties": {"temas": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
        "type": "object", "properties": {"tema": {"type": "string"}, "porcentaje": {"type": "number"}},
        "required": ["tema", "porcentaje"]}}},
    "required": ["temas"],
}
ESQUEMA_CANON = {
    "type": "object",
    "properties": {"asignaciones": {"type": "array", "items": {
        "type": "object", "properties": {"original": {"type": "string"}, "categoria": {"type": "string"}},
        "required": ["original", "categoria"]}}},
    "required": ["asignaciones"],
}


def _generar(prompt, esquema, num_ctx=16384):
    cuerpo = {"model": MODELO_TEMAS, "prompt": prompt, "format": esquema, "stream": False, "think": False,
              "options": {"num_ctx": num_ctx, "temperature": 0}}
    peticion = urllib.request.Request(f"{OLLAMA}/api/generate", data=json.dumps(cuerpo).encode(),
                                      headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(peticion, timeout=1800) as r:
        return json.loads(json.loads(r.read())["response"])


def extraer(video_id, semilla=7):
    with conexion() as con:
        textos = [r[0] for r in con.execute("SELECT texto FROM comentarios WHERE video_id = ? AND length(texto) > 15", (video_id,))]
    if not textos:
        return
    muestra = random.Random(semilla).sample(textos, min(MUESTRA, len(textos)))
    lista = "\n".join("- " + t.replace("\n", " ")[:220] for t in muestra)
    prompt = (
        "Estos son comentarios de YouTube a una conferencia matutina (mañanera) de la presidenta de México.\n"
        "Identifica los 3 temas concretos más comentados: asuntos de política pública, noticias o problemas "
        "(por ejemplo seguridad, migración, programas sociales, economía, elecciones). No cuentes como tema los "
        "saludos, las bendiciones ni el apoyo o la crítica genérica a la presidenta.\n"
        "Para cada tema da un nombre corto en español (2 a 4 palabras) y el porcentaje aproximado de comentarios "
        "que lo tratan. Ordénalos de mayor a menor.\n\n" + lista
    )
    temas = _generar(prompt, ESQUEMA_TEMAS)["temas"][:3]
    with conexion() as con:
        con.execute("DELETE FROM temas WHERE video_id = ?", (video_id,))
        con.executemany("INSERT INTO temas (video_id, rango, tema, porcentaje) VALUES (?,?,?,?)",
                        [(video_id, i + 1, t["tema"].strip(), float(t["porcentaje"])) for i, t in enumerate(temas)])
    log.info("temas: %s -> %s", video_id, " | ".join(t["tema"] for t in temas))


def canonizar(max_categorias=10):
    """Agrupa los nombres de tema de todos los meses en pocas categorías estables para seguirlas en el tiempo.

    Primero gemma4 propone las categorías; después asigna cada tema con la salida restringida a esa lista
    (enum en el esquema JSON), así no puede inventar categorías nuevas ni pasarse del máximo.
    """
    with conexion() as con:
        originales = sorted({r[0] for r in con.execute("SELECT tema FROM temas")})
    if not originales:
        return
    lista = "\n".join(f"- {t}" for t in originales)
    esquema_cats = {"type": "object", "properties": {"categorias": {
        "type": "array", "minItems": 4, "maxItems": max_categorias, "items": {"type": "string"}}}, "required": ["categorias"]}
    cats = _generar(
        f"Estos temas salieron de comentarios a las mañaneras de la presidenta de México en distintos meses. "
        f"Propón entre 6 y {max_categorias} categorías amplias, sin traslapes entre ellas, que cubran todos los temas. "
        "Cada categoría lleva un nombre corto en español (1 a 3 palabras, mayúscula inicial y acentos correctos).\n\n" + lista,
        esquema_cats)["categorias"]
    cats = list(dict.fromkeys(c.strip() for c in cats if c.strip()))[:max_categorias]
    esquema_asig = {"type": "object", "properties": {"asignaciones": {"type": "array", "items": {
        "type": "object", "properties": {"original": {"type": "string", "enum": originales}, "categoria": {"type": "string", "enum": cats}},
        "required": ["original", "categoria"]}}}, "required": ["asignaciones"]}
    asign = {a["original"]: a["categoria"] for a in _generar(
        "Asigna cada tema a la categoría que mejor lo describe. Categorías: " + "; ".join(cats) + ".\n\nTemas:\n" + lista,
        esquema_asig)["asignaciones"]}
    faltan = [t for t in originales if t not in asign]
    if faltan:
        log.warning("canonización: %d temas sin categoría, se conservan con su nombre", len(faltan))
    with conexion() as con:
        for t in originales:
            con.execute("UPDATE temas SET tema_canonico = ? WHERE tema = ?", (asign.get(t, t), t))
    log.info("canonización: %d temas en %d categorías: %s", len(originales), len(set(asign.values())), ", ".join(cats))


def procesar_todos():
    with conexion() as con:
        ids = [r[0] for r in con.execute(
            "SELECT video_id FROM videos WHERE estado = 'clasificado' "
            "AND video_id NOT IN (SELECT video_id FROM temas) ORDER BY mes")]
    for vid in ids:
        try:
            extraer(vid)
        except Exception as exc:
            log.error("temas: %s falló: %s", vid, exc)
    canonizar()
