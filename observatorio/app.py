"""Observatorio mañanero: API de clasificación de sentimiento y dashboard.

    uv run python -m observatorio.app            # http://127.0.0.1:5055

Endpoints
    GET  /                                   dashboard
    GET  /api/salud                          estado del modelo y de el servidor de Ollama
    POST /api/clasificar                     {"textos": [...]} -> sentimiento y probabilidades
    GET  /api/videos                         mañaneras analizadas con su resumen
    GET  /api/videos/<id>/comentarios        ?sentimiento=negativo&limite=50
    POST /api/videos/<id>/procesar           descarga, clasifica y extrae temas de un video (en segundo plano)
    GET  /api/tendencia                      sentimiento agregado por mes
    GET  /api/temas                          top 3 de temas por mes, con su categoría estable
"""

import json
import logging
import threading
import urllib.request

from flask import Flask, jsonify, render_template, request

from . import enjambre, temas
from .bd import conexion, inicializar
from .config import OLLAMA
from .modelo import Clasificador

app = Flask(__name__)
app.json.ensure_ascii = False
inicializar()
CLF = Clasificador()
MAX_TEXTOS = 5000


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.get("/api/salud")
def salud():
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
            modelos = [m["name"] for m in json.loads(r.read())["models"]]
        ollama = {"ok": True, "url": OLLAMA, "embeddings_disponible": any(m.startswith(CLF.modelo_emb) for m in modelos)}
    except Exception as exc:
        ollama = {"ok": False, "url": OLLAMA, "error": str(exc)}
    return jsonify({"modelo": {"embeddings": CLF.modelo_emb, "reduccion": "PCA (5 componentes)",
                               "clasificador": CLF.pipeline, "clases": CLF.clases, "metricas_prueba": CLF.metricas},
                    "ollama": ollama})


@app.post("/api/clasificar")
def clasificar():
    datos = request.get_json(silent=True) or {}
    textos = datos.get("textos")
    if isinstance(datos.get("texto"), str):
        textos = [datos["texto"]]
    if not isinstance(textos, list) or not textos or not all(isinstance(t, str) for t in textos):
        return jsonify({"error": "Envía {\"textos\": [\"...\"]} o {\"texto\": \"...\"}"}), 400
    if len(textos) > MAX_TEXTOS:
        return jsonify({"error": f"Máximo {MAX_TEXTOS} textos por petición"}), 413
    etiquetas, proba = CLF.clasificar(textos)
    return jsonify({"resultados": [{"texto": t, "sentimiento": e, "probabilidades": CLF.probabilidades(p)}
                                   for t, e, p in zip(textos, etiquetas, proba)]})


RESUMEN = """
SELECT v.video_id, v.titulo, v.fecha, v.mes, v.url, v.estado, v.agente, v.n_comentarios,
       ROUND(v.segundos_descarga, 1) AS segundos_descarga, ROUND(v.segundos_clasificacion, 1) AS segundos_clasificacion,
       COUNT(c.comentario_id) AS clasificados,
       SUM(c.sentimiento = 'positivo') AS positivos, SUM(c.sentimiento = 'neutral') AS neutrales,
       SUM(c.sentimiento = 'negativo') AS negativos,
       AVG(c.p_positivo) AS p_positivo, AVG(c.p_neutral) AS p_neutral, AVG(c.p_negativo) AS p_negativo
FROM videos v LEFT JOIN comentarios c ON c.video_id = v.video_id
GROUP BY v.video_id ORDER BY v.mes
"""


def _resumen():
    with conexion() as con:
        filas = [dict(r) for r in con.execute(RESUMEN)]
    for f in filas:
        n = f["clasificados"] or 0
        for clase, col in [("positivo", "positivos"), ("neutral", "neutrales"), ("negativo", "negativos")]:
            f[f"pct_{clase}"] = round(100 * (f[col] or 0) / n, 1) if n else None
        f["indice_neto"] = round(f["pct_positivo"] - f["pct_negativo"], 1) if n else None
    return filas


@app.get("/api/videos")
def videos():
    return jsonify(_resumen())


@app.get("/api/tendencia")
def tendencia():
    claves = ["mes", "fecha", "video_id", "titulo", "url", "clasificados", "positivos", "neutrales", "negativos",
              "pct_positivo", "pct_neutral", "pct_negativo", "indice_neto"]
    return jsonify([{k: f[k] for k in claves} for f in _resumen() if f["clasificados"]])


@app.get("/api/temas")
def temas_por_mes():
    with conexion() as con:
        filas = con.execute("""SELECT v.mes, v.fecha, t.video_id, t.rango, t.tema, t.porcentaje, t.tema_canonico
                               FROM temas t JOIN videos v USING (video_id) ORDER BY v.mes, t.rango""").fetchall()
    return jsonify([dict(r) for r in filas])


@app.get("/api/videos/<video_id>/comentarios")
def comentarios(video_id):
    sentimiento = request.args.get("sentimiento")
    limite = min(int(request.args.get("limite", 50)), 1000)
    sql = "SELECT comentario_id, texto, likes, sentimiento, p_positivo, p_neutral, p_negativo FROM comentarios WHERE video_id = ?"
    params = [video_id]
    if sentimiento:
        sql += " AND sentimiento = ?"
        params.append(sentimiento)
    sql += " ORDER BY likes DESC LIMIT ?"
    params.append(limite)
    with conexion() as con:
        return jsonify([dict(r) for r in con.execute(sql, params)])


@app.post("/api/videos/<video_id>/procesar")
def procesar(video_id):
    """Procesa un video arbitrario del canal (no entra en la serie mensual si su mes ya tiene video)."""
    with conexion() as con:
        existe = con.execute("SELECT 1 FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        if not existe:
            con.execute("INSERT OR IGNORE INTO videos (video_id, titulo, mes, url) VALUES (?, ?, ?, ?)",
                        (video_id, video_id, f"extra-{video_id}", f"https://www.youtube.com/watch?v={video_id}"))
        con.execute("UPDATE videos SET estado = 'en proceso' WHERE video_id = ?", (video_id,))

    def tarea():
        enjambre.procesar(CLF, agentes=1, video_ids=[video_id])
        temas.extraer(video_id)

    threading.Thread(target=tarea, daemon=True).start()
    return jsonify({"video_id": video_id, "estado": "en proceso", "consulta": f"/api/videos/{video_id}/comentarios"}), 202


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)-12s %(message)s", datefmt="%H:%M:%S")
    app.run(host="127.0.0.1", port=5055, threaded=True)
