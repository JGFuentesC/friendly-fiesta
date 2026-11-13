"""Clasificador de sentimiento: embeddinggemma (servidor de Ollama) -> PCA (5) -> pipeline de TPOT.

Es el modelo ganador del notebook 05: exactitud balanceada 0.779 y AUC macro 0.932 en prueba.
"""

import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np

from .config import LOTE_EMBEDDINGS, MODELO, OLLAMA, PETICIONES_PARALELAS


class Clasificador:
    def __init__(self, ruta=MODELO):
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}. Ejecuta la última celda de notebooks/05_youtube_comentarios.ipynb.")
        art = joblib.load(ruta)
        self.modelo_emb = art["modelo_embeddings"]
        self.prefijo = art["prefijo"]
        self.pca = art["pca"]
        self.clf = art["clasificador"]
        self.clases = art["clases"]
        self.metricas = art["metricas_prueba"]
        self.pipeline = art["pipeline"]
        self._candado = threading.Lock()  # predict_proba de sklearn no se comparte entre hilos con seguridad garantizada

    def _embeber_lote(self, textos):
        cuerpo = {"model": self.modelo_emb, "input": [self.prefijo + (t or " ") for t in textos], "truncate": True}
        peticion = urllib.request.Request(f"{OLLAMA}/api/embed", data=json.dumps(cuerpo).encode(),
                                          headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(peticion, timeout=600) as r:
            return json.loads(r.read())["embeddings"]

    def embeber(self, textos):
        """Divide en lotes y los envía en paralelo para mantener ocupada la GPU del servidor de Ollama."""
        lotes = [textos[i:i + LOTE_EMBEDDINGS] for i in range(0, len(textos), LOTE_EMBEDDINGS)]
        with ThreadPoolExecutor(max_workers=PETICIONES_PARALELAS) as pool:
            resultados = list(pool.map(self._embeber_lote, lotes))
        return np.array([v for lote in resultados for v in lote], dtype=np.float32)

    def clasificar(self, textos):
        """Devuelve (etiquetas, probabilidades) con columnas en el orden de self.clases."""
        if not textos:
            return [], np.empty((0, len(self.clases)))
        emb = self.embeber(list(textos))
        with self._candado:
            proba = self.clf.predict_proba(self.pca.transform(emb))
        etiquetas = [self.clases[i] for i in proba.argmax(axis=1)]
        return etiquetas, proba

    def probabilidades(self, proba):
        """Convierte una fila de probabilidades en un diccionario por clase."""
        return {c: round(float(p), 4) for c, p in zip(self.clases, proba)}
