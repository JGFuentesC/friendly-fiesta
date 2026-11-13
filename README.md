# friendly-fiesta

Notebooks didácticos para posgrado: regresión lineal, descenso de gradiente estocástico y backpropagation de un perceptrón multicapa (MLP).

## Estructura

```
data/
  raw/file.csv              dataset crudo (999 registros: x1, x2, x3, y)
  processed/train.csv       split 70% (seed 42)
  processed/validation.csv  split 30% (seed 42)
notebooks/
  01_minimos_cuadrados_sgd.ipynb  mínimos cuadrados matriciales y SGD
  02_mlp_backprop.ipynb           forward, backpropagation y tres implementaciones
figures/                    gráficos del repo
reports/                    salidas HTML
scripts/make_split.py       regenera train/validation de forma determinista
```

## Entorno

El entorno se gestiona con [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Regenerar el split de datos:

```bash
uv run python scripts/make_split.py
```

Lanzar JupyterLab:

```bash
uv run jupyter lab
```

## Detección de fraude (IBM SD254)

Análisis de credit scoring con la tarjeta de crédito sintética de IBM. Los datos crudos no están en el repo (ver `.gitignore`); se descargan de Kaggle:

https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions

Descomprimir en `data/raw/ibm/` los archivos:

```
credit_card_transactions-ibm_v2.csv
sd254_cards.csv
sd254_users.csv
```

Luego generar el dataset de features (24 M de filas, ~4 GB, gitignored):

```bash
uv run python scripts/build_fraud_dataset.py
```

Escribe `data/processed/fraud_features.csv` con `id, features numéricas, is_fraud`.

Generar los submuestreos para el entrenamiento (gitignored):

```bash
uv run python scripts/make_fraud_sample.py
```

Escribe `fraud_train.csv` (100k filas, fraude sobremuestreado al 10 %) y `fraud_test.csv` (300k filas, tasa real 0.12 %).

El notebook `notebooks/03_fraude_ff.ipynb` entrena la red FF (PyTorch + GPU) e ilustra la distinción entre hiperparámetros y parámetros, con una búsqueda pequeña de hiperparámetros. Está generado por `scripts/build_fraud_notebook.py` y se ejecuta en $GPU_HOST:

```bash
uv run python scripts/build_fraud_notebook.py
scp notebooks/03_fraude_ff.ipynb data/processed/fraud_train.csv data/processed/fraud_test.csv \
  $GPU_HOST:$RUTA_REMOTA/
ssh $GPU_HOST 'docker exec lab-pytorch bash -lc "cd /workspace/friendly-fiesta && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/03_fraude_ff.ipynb"'
```

## Ejecución en GPU

El notebook `02_mlp_backprop.ipynb` entrena con NumPy, PyTorch y TensorFlow. La versión GPU corre en un servidor remoto con GPU (RTX 3070 en nuestras pruebas) dentro de un contenedor `lab-pytorch`, que trae los tres marcos y Plotly. Los comandos de esta guía usan tres variables de entorno: `GPU_HOST` (alias SSH del servidor), `RUTA_SERVICIO` (carpeta del `compose.yml` de ese contenedor) y `RUTA_REMOTA` (carpeta del repo montada en el contenedor como `/workspace/friendly-fiesta`).

```bash
ssh $GPU_HOST 'cd $RUTA_SERVICIO && docker compose up -d'
ssh -L 8889:localhost:8889 $GPU_HOST
```

Luego abrir `http://localhost:8889` en el navegador local.

## LeNet-5 sobre MNIST

La sección 5 de `spa/demo_nlp_comparador_biblia_coran.html` muestra un clasificador LSTM Biblia/Corán. `scripts/train_lstm_biblia_coran.py` lo entrena en la GPU del servidor remoto y escribe `data/processed/lstm_resultados.json` y `figures/lstm/anatomy_data.json`; `scripts/anatomy_lstm/build.mjs` dibuja la celda con sus compuertas y empaqueta todo en `spa/lstm_datos.js`, que es lo que lee la página.

```bash
scp scripts/train_lstm_biblia_coran.py $GPU_HOST:$RUTA_REMOTA/scripts/
scp data/processed/bow_nltk_biblia.json $GPU_HOST:$RUTA_REMOTA/data/processed/
scp data/raw/textos_nlp/*.txt $GPU_HOST:$RUTA_REMOTA/data/raw/textos_nlp/
ssh $GPU_HOST 'docker exec -w /workspace/friendly-fiesta lab-pytorch python scripts/train_lstm_biblia_coran.py'
scp $GPU_HOST:$RUTA_REMOTA/data/processed/lstm_resultados.json data/processed/
scp $GPU_HOST:$RUTA_REMOTA/figures/lstm/anatomy_data.json figures/lstm/
node scripts/anatomy_lstm/build.mjs
uv run python scripts/verificar_redaccion_html.py --seccion seccion-lstm spa/demo_nlp_comparador_biblia_coran.html
python3 -m http.server 8000   # abrir /spa/demo_nlp_comparador_biblia_coran.html
```

`notebooks/04_lenet_mnist.ipynb` entrena LeNet-5 en la GPU del servidor remoto y explica la arquitectura, los kernels, la operación de convolución y las activaciones capa por capa, con ejemplos inferidos al final. Está generado por `scripts/build_lenet_notebook.py`. El diagrama isométrico de la red (`figures/lenet/lenet_anatomy.svg`, incrustado en el notebook) lo construye `scripts/anatomy_lenet/build.mjs` a partir de `figures/lenet/anatomy_data.json`, que exporta el propio notebook.

```bash
uv run python scripts/build_lenet_notebook.py
scp notebooks/04_lenet_mnist.ipynb $GPU_HOST:$RUTA_REMOTA/notebooks/
scp figures/lenet/lenet_anatomy.svg $GPU_HOST:$RUTA_REMOTA/figures/lenet/
ssh $GPU_HOST 'docker exec lab-pytorch bash -lc "cd /workspace/friendly-fiesta && jupyter nbconvert --to notebook --execute --inplace notebooks/04_lenet_mnist.ipynb"'
node scripts/anatomy_lenet/build.mjs   # tras traer anatomy_data.json, si cambió el modelo
```

## Sentimiento en comentarios de YouTube y observatorio mañanero

`notebooks/05_youtube_comentarios.ipynb` etiqueta el sentimiento de los comentarios de una mañanera, limpia el texto con NLTK y spaCy, y compara dos clasificadores entrenados con LazyPredict y TPOT: uno con variables construidas a mano y otro con embeddings de `embeddinggemma` reducidos con PCA. Incluye control de sobreajuste y XAI con SHAP.

`observatorio/` usa el modelo ganador en una API Flask con dashboard: un enjambre de agentes elige una mañanera al azar por mes desde octubre de 2024, descarga sus comentarios, los clasifica y extrae los temas con `gemma4`. Los embeddings y los temas requieren un servidor de [Ollama](https://ollama.com) con `embeddinggemma` y `gemma4`.

```bash
export OLLAMA_URL=http://<servidor-ollama>:11434
uv run python -m observatorio.pipeline --agentes 6   # descarga, clasifica y extrae temas
uv run python -m observatorio.app                    # http://127.0.0.1:5055
```

Los comentarios descargados, la base del observatorio y el modelo exportado se guardan en `data/youtube_comments/` y `data/observatorio/`, que no se publican porque contienen textos de terceros. El modelo se regenera con la última celda del notebook 05.

## Textos de la Biblia y el Corán

Los textos originales no se incluyen en este repositorio, porque las traducciones pueden tener derechos de terceros. La SPA funciona sin ellos con los datos ya calculados de `data/processed/` (frecuencias, lemas y fragmentos breves). Para regenerar esos datos o reentrenar la LSTM, coloca tus propias copias en `data/raw/textos_nlp/biblia_es.txt` y `data/raw/textos_nlp/coran_es.txt`.

## Licencia

Este proyecto se distribuye bajo la licencia GNU GPL v3.0 (`GPL-3.0-only`). El texto completo está en [`LICENSE`](LICENSE).
