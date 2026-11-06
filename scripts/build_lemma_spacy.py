"""Lematiza la Biblia y el Corán con spaCy y genera el BOW de lemas.

Usa el modelo en español es_core_news_sm (entrenado con noticias), quita
stopwords de spaCy, puntuación y términos cortos, y cuenta la frecuencia de
cada lema. El resultado se guarda en data/processed/ para que la SPA lo
consuma en la nube de palabras.

Uso:
    uv run python scripts/build_lemma_spacy.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import spacy

# Rutas relativas a la raíz del repositorio
RAIZ = Path(__file__).resolve().parents[1]
ENTRADAS = {
    "biblia": RAIZ / "data" / "raw" / "textos_nlp" / "biblia_es.txt",
    "coran": RAIZ / "data" / "raw" / "textos_nlp" / "coran_es.txt",
}
CARPETA_SALIDA = RAIZ / "data" / "processed"

MODELO = "es_core_news_sm"
# El parser y el NER no hacen falta para lematizar y ahorramos tiempo
COMPONENTES_SIN_USO = ["parser", "ner"]
LONGITUD_MINIMA = 3


def leer_fragmentos(ruta: Path) -> list[str]:
    """Devuelve las líneas no vacías del texto, listas para nlp.pipe."""
    texto = ruta.read_text(encoding="utf-8")
    return [linea.strip() for linea in texto.splitlines() if linea.strip()]


def lematizar(ruta: Path, nlp) -> dict:
    """Lematiza el texto y devuelve el BOW de lemas con sus estadísticas."""
    lemas: Counter[str] = Counter()
    leidas = 0

    for doc in nlp.pipe(leer_fragmentos(ruta), batch_size=256):
        for token in doc:
            if token.is_space or not token.is_alpha:
                continue
            leidas += 1
            if token.is_stop:
                continue
            lema = token.lemma_.lower()
            if len(lema) < LONGITUD_MINIMA:
                continue
            lemas[lema] += 1

    tras_filtro = sum(lemas.values())
    return {
        "fuente": f"spaCy {spacy.__version__} ({MODELO})",
        "longitud_minima": LONGITUD_MINIMA,
        "stopwords": sorted(nlp.Defaults.stop_words),
        "n_stopwords": len(nlp.Defaults.stop_words),
        "palabras_leidas": leidas,
        "palabras_tras_filtro": tras_filtro,
        "palabras_unicas": len(lemas),
        "bow": dict(lemas.most_common()),
    }


def main() -> None:
    nlp = spacy.load(MODELO, disable=COMPONENTES_SIN_USO)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    print(f"Modelo: {MODELO} | stopwords en español: {len(nlp.Defaults.stop_words)}")
    print()

    for nombre, ruta in ENTRADAS.items():
        datos = lematizar(ruta, nlp)
        destino = CARPETA_SALIDA / f"lemma_spacy_{nombre}.json"
        destino.write_text(
            json.dumps(datos, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        quitadas = datos["palabras_leidas"] - datos["palabras_tras_filtro"]
        print(f"{nombre.capitalize()}:")
        print(f"  palabras leídas       : {datos['palabras_leidas']:,}")
        print(f"  quitadas (stopwords)  : {quitadas:,}")
        print(f"  lemas tras filtro     : {datos['palabras_tras_filtro']:,}")
        print(f"  vocabulario de lemas  : {datos['palabras_unicas']:,}")
        print(f"  guardado en           : {destino.relative_to(RAIZ)}")
        print()


if __name__ == "__main__":
    main()
