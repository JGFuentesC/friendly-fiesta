"""Enjambre de agentes que exploran el canal, descargan comentarios y los clasifican en paralelo.

- Exploradores: consultan la fecha exacta de cada mañanera candidata (la lista del canal solo trae fechas aproximadas).
- Descargadores: cada agente toma un video pendiente de la cola y baja sus comentarios principales con yt-dlp.
- Clasificador: consume lo que entregan los descargadores y lo clasifica por lotes paralelos en la GPU del servidor de Ollama,
  así la descarga (red, CPU de la Mac) y la clasificación (GPU) trabajan al mismo tiempo.
"""

import datetime as dt
import logging
import queue
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from yt_dlp import YoutubeDL

from .bd import conexion
from .config import CANAL, TOMA_DE_POSESION

log = logging.getLogger("observatorio")
ES_MANANERA = re.compile(r"[Cc]onferencia.*([Pp]residenta|matutina)|[Mm]añanera|[Mm]orning [Pp]ress [Cc]onference")
FIN = object()


def _ydl(extractor_args=None, lang="es", **extra):
    # lang=es: sin esto YouTube entrega los títulos traducidos al idioma de la petición
    args = {"youtube": {"lang": [lang]}, "youtubetab": {"lang": [lang]}} if lang else {}
    for k, v in (extractor_args or {}).items():
        args[k] = {**args.get(k, {}), **v}
    return YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "extractor_args": args, **extra})


def meses(desde=TOMA_DE_POSESION, hasta=None):
    hoy = dt.date.today()
    a, m = map(int, desde.split("-"))
    fin = hasta or f"{hoy.year}-{hoy.month:02d}"
    salida = []
    while f"{a}-{m:02d}" <= fin:
        salida.append(f"{a}-{m:02d}")
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)
    return salida


def explorar(agentes=8):
    """Lista las transmisiones del canal y obtiene la fecha exacta de cada mañanera candidata."""
    # yt-dlp solo interpreta las fechas aproximadas ("2 months ago") en inglés: una consulta trae las fechas
    # y otra los títulos originales en español
    with _ydl(lang=None, extract_flat=True, extractor_args={"youtubetab": {"approximate_date": [""]}}) as ydl:
        fechas = {e["id"]: e.get("timestamp") for e in ydl.extract_info(CANAL, download=False)["entries"]}
    with _ydl(extract_flat=True) as ydl:
        entradas = list(ydl.extract_info(CANAL, download=False)["entries"])
    for e in entradas:
        e["timestamp"] = fechas.get(e["id"])
    limite = dt.datetime(int(TOMA_DE_POSESION[:4]) - 1, 1, 1, tzinfo=dt.UTC).timestamp()  # holgura por fechas aproximadas
    candidatas = [e for e in entradas if ES_MANANERA.search(e.get("title") or "")
                  and (e.get("duration") or 0) > 3600 and (e.get("timestamp") or 0) >= limite]
    with conexion() as con:
        conocidas = {r[0] for r in con.execute("SELECT video_id FROM candidatos")}
    nuevas = [e for e in candidatas if e["id"] not in conocidas]
    log.info("exploración: %d mañaneras candidatas, %d sin fecha exacta", len(candidatas), len(nuevas))

    def fecha_exacta(e):
        with _ydl() as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={e['id']}", download=False, process=False)
        f = info.get("release_date") or info.get("upload_date")
        return e["id"], e["title"], f"{f[:4]}-{f[4:6]}-{f[6:]}", e.get("duration")

    with ThreadPoolExecutor(max_workers=agentes, thread_name_prefix="explorador") as pool:
        futuros = [pool.submit(fecha_exacta, e) for e in nuevas]
        for fut in as_completed(futuros):
            try:
                vid, titulo, fecha, dur = fut.result()
            except Exception as exc:  # video privado o eliminado
                log.warning("explorador: %s", exc)
                continue
            with conexion() as con:
                con.execute("INSERT OR REPLACE INTO candidatos VALUES (?,?,?,?,?)", (vid, titulo, fecha, fecha[:7], dur))


def seleccionar(semilla=2024):
    """Elige una mañanera al azar por mes desde la toma de posesión. Una selección ya hecha no se cambia."""
    rng = random.Random(semilla)
    with conexion() as con:
        ya = {r[0] for r in con.execute("SELECT mes FROM videos")}
        for mes in meses():
            opciones = con.execute("SELECT * FROM candidatos WHERE mes = ? ORDER BY video_id", (mes,)).fetchall()
            if not opciones:
                log.warning("selección: %s no tiene mañaneras en el canal", mes)
                continue
            elegido = rng.choice(opciones)  # se consume el generador siempre, para que la selección sea reproducible
            if mes in ya:
                continue
            con.execute("INSERT INTO videos (video_id, titulo, fecha, mes, url) VALUES (?,?,?,?,?)",
                        (elegido["video_id"], elegido["titulo"], elegido["fecha"], mes,
                         f"https://www.youtube.com/watch?v={elegido['video_id']}"))
            log.info("selección: %s -> %s %s", mes, elegido["fecha"], elegido["titulo"][:70])


def descargar_comentarios(video_id):
    """Comentarios principales (sin respuestas, igual que en el entrenamiento del modelo)."""
    args = {"youtube": {"max_comments": ["all", "all", "0", "0"]}}
    with _ydl(getcomments=True, extractor_args=args, ignore_no_formats_error=True) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    return [c for c in (info.get("comments") or []) if c.get("parent", "root") == "root"]


def _guardar(video_id, comentarios, etiquetas, proba, clases):
    k = {c: clases.index(c) for c in ("positivo", "neutral", "negativo")}
    filas = [(c["id"], video_id, c.get("text") or "", c.get("like_count") or 0, c.get("timestamp"), e,
              float(p[k["positivo"]]), float(p[k["neutral"]]), float(p[k["negativo"]]))
             for c, e, p in zip(comentarios, etiquetas, proba)]
    with conexion() as con:
        con.execute("DELETE FROM comentarios WHERE video_id = ?", (video_id,))
        con.executemany("INSERT OR REPLACE INTO comentarios VALUES (?,?,?,?,?,?,?,?,?)", filas)


def procesar(clasificador, agentes=6, video_ids=None):
    """Lanza el enjambre: `agentes` descargadores en paralelo y un clasificador que consume su producción."""
    with conexion() as con:
        if video_ids:
            pendientes = list(video_ids)
        else:
            pendientes = [r[0] for r in con.execute("SELECT video_id FROM videos WHERE estado != 'clasificado' ORDER BY mes")]
    if not pendientes:
        log.info("enjambre: no hay videos pendientes")
        return
    log.info("enjambre: %d videos, %d agentes descargadores", len(pendientes), agentes)
    tareas, entregas = queue.Queue(), queue.Queue()
    for v in pendientes:
        tareas.put(v)

    def descargador(nombre):
        while True:
            try:
                vid = tareas.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            try:
                coms = descargar_comentarios(vid)
            except Exception as exc:
                log.error("%s: %s falló: %s", nombre, vid, exc)
                with conexion() as con:
                    con.execute("UPDATE videos SET estado='error', error=?, agente=? WHERE video_id=?", (str(exc)[:500], nombre, vid))
                continue
            seg = time.time() - t0
            with conexion() as con:
                con.execute("UPDATE videos SET estado='descargado', agente=?, n_comentarios=?, segundos_descarga=? WHERE video_id=?",
                            (nombre, len(coms), seg, vid))
            log.info("%s: %s, %d comentarios en %.0f s", nombre, vid, len(coms), seg)
            entregas.put((vid, coms))

    def clasificador_agente():
        while (item := entregas.get()) is not FIN:
            vid, coms = item
            t0 = time.time()
            etiquetas, proba = clasificador.clasificar([c.get("text") or "" for c in coms])
            _guardar(vid, coms, etiquetas, proba, clasificador.clases)
            seg = time.time() - t0
            with conexion() as con:
                con.execute("UPDATE videos SET estado='clasificado', segundos_clasificacion=? WHERE video_id=?", (seg, vid))
            log.info("clasificador: %s, %d comentarios en %.1f s (%.0f comentarios/s)", vid, len(coms), seg, len(coms) / max(seg, 1e-9))

    hilo_clf = threading.Thread(target=clasificador_agente, name="clasificador")
    hilo_clf.start()
    hilos = [threading.Thread(target=descargador, args=(f"agente-{i + 1}",), name=f"agente-{i + 1}") for i in range(agentes)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    entregas.put(FIN)
    hilo_clf.join()
