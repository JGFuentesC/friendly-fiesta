"""Base SQLite del observatorio: videos, comentarios clasificados y temas por video."""

import sqlite3
from contextlib import contextmanager

from .config import BD

ESQUEMA = """
CREATE TABLE IF NOT EXISTS candidatos (
    video_id TEXT PRIMARY KEY, titulo TEXT, fecha TEXT, mes TEXT, duracion INTEGER
);
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY, titulo TEXT, fecha TEXT, mes TEXT UNIQUE, url TEXT,
    estado TEXT DEFAULT 'pendiente', agente TEXT, n_comentarios INTEGER,
    segundos_descarga REAL, segundos_clasificacion REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS comentarios (
    comentario_id TEXT PRIMARY KEY, video_id TEXT, texto TEXT, likes INTEGER, timestamp INTEGER,
    sentimiento TEXT, p_positivo REAL, p_neutral REAL, p_negativo REAL
);
CREATE INDEX IF NOT EXISTS ix_com_video ON comentarios(video_id);
CREATE TABLE IF NOT EXISTS temas (
    video_id TEXT, rango INTEGER, tema TEXT, porcentaje REAL, tema_canonico TEXT,
    PRIMARY KEY (video_id, rango)
);
"""


@contextmanager
def conexion():
    BD.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(BD, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def inicializar():
    with conexion() as con:
        con.executescript(ESQUEMA)
