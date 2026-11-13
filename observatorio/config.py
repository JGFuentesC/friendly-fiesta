"""Configuración del observatorio mañanero. Todas las rutas se resuelven desde la raíz del repo."""

import os
from pathlib import Path

# LightGBM y XGBoost (importados por TPOT al cargar el modelo) traen cada uno su libomp en macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "observatorio"
BD = DATOS / "observatorio.db"
MODELO = DATOS / "modelo" / "sentimiento_gemma_pca_tpot.joblib"

# Servidor de Ollama con GPU. Se indica con la variable de entorno OLLAMA_URL (p. ej. http://mi-servidor:11434).
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELO_TEMAS = os.environ.get("MODELO_TEMAS", "gemma4:latest")
# con OLLAMA_NUM_PARALLEL=4 en el servidor,: cuatro lotes simultáneos mantienen la GPU ocupada sin encolar
PETICIONES_PARALELAS = int(os.environ.get("PETICIONES_PARALELAS", "4"))
LOTE_EMBEDDINGS = 32  # con 32, 64, 128 o 256 el rendimiento es el mismo (~165 comentarios/s): manda la GPU

CANAL = "https://www.youtube.com/@ClaudiaSheinbaumP/streams"
TOMA_DE_POSESION = "2024-10"  # 1 de octubre de 2024
