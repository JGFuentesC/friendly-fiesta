"""Corre el observatorio completo: explorar el canal, elegir una mañanera por mes, descargar y clasificar
con el enjambre, y extraer los temas con gemma4.

    uv run python -m observatorio.pipeline --agentes 6 --semilla 2024
"""

import argparse
import logging
import time

from . import enjambre, temas
from .bd import conexion, inicializar
from .modelo import Clasificador


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agentes", type=int, default=6, help="agentes descargadores en paralelo")
    p.add_argument("--exploradores", type=int, default=8, help="agentes que consultan fechas exactas")
    p.add_argument("--semilla", type=int, default=2024, help="semilla de la selección aleatoria mensual")
    p.add_argument("--sin-temas", action="store_true", help="omite la extracción de temas con gemma4")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)-12s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("observatorio")
    inicializar()
    t0 = time.time()

    enjambre.explorar(agentes=a.exploradores)
    enjambre.seleccionar(semilla=a.semilla)
    t1 = time.time()
    enjambre.procesar(Clasificador(), agentes=a.agentes)
    t2 = time.time()
    if not a.sin_temas:
        temas.procesar_todos()
    t3 = time.time()

    with conexion() as con:
        r = con.execute("SELECT COUNT(*), SUM(n_comentarios) FROM videos WHERE estado = 'clasificado'").fetchone()
    log.info("listo: %d videos, %d comentarios | exploración %.0f s, enjambre %.0f s, temas %.0f s",
             r[0], r[1] or 0, t1 - t0, t2 - t1, t3 - t2)


if __name__ == "__main__":
    main()
