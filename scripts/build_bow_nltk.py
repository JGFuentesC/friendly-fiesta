"""Genera el BOW de la Biblia y el Corán filtrando stopwords con NLTK.

Lee los textos de data/raw/textos_nlp/, aplica las stopwords en español de
nltk.corpus, cuenta frecuencias con nltk.FreqDist y guarda el resultado en
data/processed/ para que la SPA lo consuma.

Uso:
    uv run python scripts/build_bow_nltk.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Rutas relativas a la raíz del repositorio
RAIZ = Path(__file__).resolve().parents[1]
ENTRADAS = {
    "biblia": RAIZ / "data" / "raw" / "textos_nlp" / "biblia_es.txt",
    "coran": RAIZ / "data" / "raw" / "textos_nlp" / "coran_es.txt",
}
CARPETA_SALIDA = RAIZ / "data" / "processed"

# Solo letras castellanas en minúsculas; números y símbolos quedan fuera
PATRON_TOKENS = r"[a-záéíóúüñ]+"
LONGITUD_MINIMA = 3


def preparar_texto(ruta: Path, tokenizador: RegexpTokenizer, vacias: set[str]) -> dict:
    """Tokeniza, quita stopwords de NLTK y devuelve el BOW con sus estadísticas."""
    texto = ruta.read_text(encoding="utf-8").lower()
    tokens = tokenizador.tokenize(texto)
    filtrados = [t for t in tokens if t not in vacias and len(t) >= LONGITUD_MINIMA]
    frecuencias = nltk.FreqDist(filtrados)

    return {
        "fuente": f"NLTK {nltk.__version__} (stopwords en español)",
        "longitud_minima": LONGITUD_MINIMA,
        "stopwords": sorted(vacias),
        "n_stopwords": len(vacias),
        "palabras_leidas": len(tokens),
        "palabras_tras_filtro": len(filtrados),
        "palabras_unicas": len(frecuencias),
        "bow": dict(frecuencias.most_common()),
    }


def main() -> None:
    # Las stopwords se descargan una sola vez y quedan en el caché de NLTK
    nltk.download("stopwords", quiet=True)
    vacias = set(stopwords.words("spanish"))
    tokenizador = RegexpTokenizer(PATRON_TOKENS)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    for nombre, ruta in ENTRADAS.items():
        datos = preparar_texto(ruta, tokenizador, vacias)
        destino = CARPETA_SALIDA / f"bow_nltk_{nombre}.json"
        destino.write_text(
            json.dumps(datos, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{nombre.capitalize()}:")
        print(f"  palabras leídas       : {datos['palabras_leidas']:,}")
        print(f"  stopwords NLTK ({datos['n_stopwords']})  : {datos['palabras_leidas'] - datos['palabras_tras_filtro']:,} términos quitados")
        print(f"  tras filtro           : {datos['palabras_tras_filtro']:,}")
        print(f"  vocabulario (BOW)     : {datos['palabras_unicas']:,}")
        print(f"  guardado en           : {destino.relative_to(RAIZ)}")
        print()


if __name__ == "__main__":
    main()
