"""Vectoriza la Biblia y el Corán con TF-IDF y proyecta 500 documentos a 3D con PCA.

Segmentación en documentos:
  - Biblia: ventanas de 200 palabras sin solape.
  - Corán: primero por sura (se respeta la estructura) y cada sura larga se
    corta en ventanas de hasta 200 palabras.

Después se toma una muestra aleatoria de 500 documentos con la semilla 42 del
repositorio, se calcula TF-IDF con scikit-learn sobre un vocabulario compartido
y se reduce a 3 componentes con PCA. El resultado se guarda en
data/processed/ para la sección 4 de la SPA.

Uso:
    uv run python scripts/build_tfidf_pca.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Rutas relativas a la raíz del repositorio
RAIZ = Path(__file__).resolve().parents[1]
ENTRADAS = {
    "biblia": RAIZ / "data" / "raw" / "textos_nlp" / "biblia_es.txt",
    "coran": RAIZ / "data" / "raw" / "textos_nlp" / "coran_es.txt",
}
DESTINO = RAIZ / "data" / "processed" / "tfidf_pca_muestra.json"

SEMILLA = 42
VENTANA = 200          # palabras por documento
MUESTRA = 500          # documentos proyectados
VOCABULARIO = 5000     # términos máximos del TF-IDF
FRAGMENTO_PALABRAS = 14  # palabras del recorte para el tooltip

PATRON_PALABRAS = re.compile(r"[a-záéíóúüñ]+")
PATRON_SURA = re.compile(r"^SURA \d+:.*$", re.MULTILINE)


def tokenizar(texto: str) -> list[str]:
    """Extrae las palabras en minúsculas; números y símbolos quedan fuera."""
    return PATRON_PALABRAS.findall(texto.lower())


def en_ventanas(palabras: list[str], tamano: int = VENTANA) -> list[list[str]]:
    """Corta una lista de palabras en trozos de hasta `tamano` elementos."""
    return [palabras[i:i + tamano] for i in range(0, len(palabras), tamano)]


def documentos_biblia(texto: str) -> list[list[str]]:
    """La Biblia se corta en ventanas fijas de palabras."""
    return en_ventanas(tokenizar(texto))


def documentos_coran(texto: str) -> list[list[str]]:
    """El Corán se corta por sura y luego en ventanas, sin cruzar suras."""
    inicios = [m.start() for m in PATRON_SURA.finditer(texto)]
    limites = ([0] if inicios and inicios[0] > 0 else []) + inicios + [len(texto)]
    documentos: list[list[str]] = []
    for inicio, fin in zip(limites, limites[1:]):
        documentos.extend(en_ventanas(tokenizar(texto[inicio:fin])))
    return documentos


def main() -> None:
    # 1. Segmentar cada corpus en documentos
    pool: list[dict] = []
    reparto_pool: dict[str, int] = {}
    for nombre, ruta in ENTRADAS.items():
        texto = ruta.read_text(encoding="utf-8")
        cortes = documentos_biblia(texto) if nombre == "biblia" else documentos_coran(texto)
        reparto_pool[nombre] = len(cortes)
        for palabras in cortes:
            pool.append({
                "texto": nombre,
                "documento": " ".join(palabras),
                "vista": " ".join(palabras[:FRAGMENTO_PALABRAS]),
            })

    # 2. Muestra aleatoria de 500 documentos (semilla del repositorio)
    azar = random.Random(SEMILLA)
    muestra = azar.sample(pool, MUESTRA)
    reparto_muestra: dict[str, int] = {}
    for doc in muestra:
        reparto_muestra[doc["texto"]] = reparto_muestra.get(doc["texto"], 0) + 1

    # 3. TF-IDF sobre el vocabulario compartido de la muestra
    vectorizador = TfidfVectorizer(
        analyzer=str.split,   # las palabras ya vienen normalizadas
        min_df=2,
        max_features=VOCABULARIO,
        sublinear_tf=True,
    )
    matriz = vectorizador.fit_transform(d["documento"] for d in muestra)

    # 4. PCA a 3 componentes
    pca = PCA(n_components=3, random_state=SEMILLA)
    coordenadas = pca.fit_transform(matriz.toarray())

    # 5. Guardar la muestra proyectada
    documentos = []
    for indice, (doc, (x, y, z)) in enumerate(zip(muestra, coordenadas)):
        documentos.append({
            "id": indice,
            "texto": doc["texto"],
            "vista": doc["vista"],
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "z": round(float(z), 4),
        })

    datos = {
        "fuente": f"scikit-learn {sklearn.__version__}",
        "semilla": SEMILLA,
        "ventana_palabras": VENTANA,
        "documentos_pool": len(pool),
        "pool": reparto_pool,
        "documentos_muestra": len(muestra),
        "reparto": reparto_muestra,
        "vocabulario": int(matriz.shape[1]),
        "varianza_explicada": [round(float(v), 5) for v in pca.explained_variance_ratio_],
        "documentos": documentos,
    }
    DESTINO.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")

    varianza = datos["varianza_explicada"]
    print(f"Pool de documentos   : {len(pool):,} ({reparto_pool})")
    print(f"Muestra aleatoria    : {len(muestra)} documentos ({reparto_muestra})")
    print(f"Ventana              : {VENTANA} palabras")
    print(f"Matriz TF-IDF        : {matriz.shape[0]:,} x {matriz.shape[1]:,}")
    print(f"Varianza explicada   : PC1 {varianza[0]:.1%} | PC2 {varianza[1]:.1%} | PC3 {varianza[2]:.1%}")
    print(f"Acumulada            : {sum(varianza):.1%}")
    print(f"Guardado en          : {DESTINO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
