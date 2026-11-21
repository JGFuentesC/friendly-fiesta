# friendly-fiesta

Notebooks didácticos para posgrado: regresión lineal, descenso de gradiente estocástico y backpropagation de un perceptrón multicapa (MLP).

## Estructura

```
agentes/asistente_teresa/   agente de Google ADK (sesión 08)
data/raw/                   datos crudos (los de terceros están gitignored)
data/processed/             JSON y CSV derivados que leen las SPA y los apuntes
notebooks/
  01_minimos_cuadrados_sgd.ipynb  mínimos cuadrados matriciales y SGD
  02_mlp_backprop.ipynb           forward, backpropagation y tres implementaciones
  03_fraude_ff.ipynb              red FF para fraude (gitignored, se genera)
  04_lenet_mnist.ipynb            LeNet-5 sobre MNIST
  05_youtube_comentarios.ipynb    sentimiento en comentarios de YouTube
  06_minigpt_teresa.ipynb         mini GPT entrenado con diálogos de Teresa
  07_difusion.ipynb               modelos de difusión (DDPM)
  08_agente_adk.ipynb             agente simple con Google ADK y Ollama
  09_ruido_hotot.ipynb            ruido gaussiano sobre una foto, paso a paso
  10_difusion_simple.ipynb        difusión 2D: entrenar e inferir con un MLP
  11_difusion_video.ipynb         hola mundo de difusión de video
  12_unet_animefaces.ipynb        U-Net condicional con caras de anime (GPU)
reports/apuntes/            apuntes en LaTeX con estilo Anáhuac (anahuac.sty)
spa/                        páginas interactivas autocontenidas
scripts/                    generadores, entrenamientos y verificadores
observatorio/               API Flask del observatorio mañanero
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

## Apuntes en LaTeX

Los apuntes de las sesiones 06 a 08 comparten `reports/apuntes/anahuac.sty` (portada, encabezados y cajas de definición, intuición, ejemplo, nota y ejercicio). `reports/apuntes/_plantilla/main.tex` es un ejemplo mínimo. Compilan con [tectonic](https://tectonic-typesetting.github.io):

```bash
tectonic -X compile reports/apuntes/06_llm/main.tex
tectonic -X compile reports/apuntes/07_difusion/main.tex
tectonic -X compile reports/apuntes/08_agentes/main.tex
uv run python scripts/verificar_redaccion_html.py reports/apuntes/0*/*.tex spa/*.html notebooks/0[678]_*.ipynb
```

## Sesión 06: mini GPT con diálogos de *Teresa*

`scripts/descargar_teresa.py` baja con yt-dlp solo los subtítulos en español de capítulos y escenas de *Teresa* (Televisa, 2010) y los limpia a diálogo plano en `data/raw/teresa/` (gitignored, son textos de terceros). `scripts/train_minigpt_teresa.py` entrena en la GPU un GPT decoder-only de 12 M de parámetros (6 capas, 384 dimensiones, 6 cabezas, contexto 256, SentencePiece BPE de 4000 piezas) con parada temprana, y exporta `data/processed/minigpt_teresa.json` (curvas, muestras por paso, mapas de atención, embeddings y distribuciones de siguiente token). Requiere `sentencepiece` dentro de `lab-pytorch`.

```bash
uv run python scripts/descargar_teresa.py
scp scripts/train_minigpt_teresa.py $GPU_HOST:$RUTA_REMOTA/scripts/
scp data/raw/teresa/teresa.txt $GPU_HOST:$RUTA_REMOTA/data/raw/teresa/
ssh $GPU_HOST 'docker exec -w /workspace/friendly-fiesta lab-pytorch python scripts/train_minigpt_teresa.py --pasos 6000 --minutos 40'
scp $GPU_HOST:$RUTA_REMOTA/data/processed/minigpt_teresa.json data/processed/
uv run python scripts/figuras_minigpt.py        # figuras y métricas del apunte
uv run python scripts/embeber_minigpt_spa.py    # incrusta el JSON en spa/minigpt_teresa.html
uv run python scripts/build_minigpt_notebook.py # notebook 06; se ejecuta en lab-pytorch con MINIGPT_RAPIDO=1
```

## Sesión 07: modelos de difusión

Inspirada en un reel que construye la difusión como una escalera: un píxel más ruido, ruido puro, una red que revierte el ruido, condicionarla con un texto y aplicarla en bucle. `scripts/train_ddpm.py` entrena cuatro DDPM (T = 1000, horario lineal): dos MLP sobre nubes de puntos 2D, una U-Net condicional sobre paisajes de pixel art de 24×24 con tres prompts y guía sin clasificador, y una U-Net condicional sobre MNIST. Escribe `data/processed/difusion.json`. MNIST se lee de los archivos idx, sin torchvision.

```bash
scp scripts/train_ddpm.py $GPU_HOST:$RUTA_REMOTA/scripts/
ssh $GPU_HOST 'docker exec -w /workspace/friendly-fiesta lab-pytorch python scripts/train_ddpm.py'
scp $GPU_HOST:$RUTA_REMOTA/data/processed/difusion.json data/processed/
uv run python scripts/figuras_difusion.py       # figuras y métricas del apunte
uv run python scripts/build_difusion_spa.py     # incrusta el JSON en spa/difusion.html
uv run python scripts/build_difusion_notebook.py
```

## Sesión 08: agente simple con Google ADK

`agentes/asistente_teresa/` es un `LlmAgent` de Google ADK 2.x que usa un modelo local de Ollama (`qwen3:4b` por defecto) a través de LiteLLM. Tiene tres herramientas deterministas (`buscar_personaje`, `calcular`, `consultar_sesion`) y guarda memoria en el estado de la sesión. Las fechas de `data/processed/curso_sesiones.json` son el calendario del curso. `scripts/grabar_traza_agente.py` corre preguntas fijas y guarda la traza de eventos que reproduce `spa/agente_adk.html`.

```bash
ssh -N -L 11434:localhost:11434 $GPU_HOST &      # túnel a Ollama
export OLLAMA_API_BASE=http://localhost:11434   # opcional: AGENTE_MODELO=ollama_chat/qwen2.5:7b
uv run adk run agentes/asistente_teresa         # o: uv run adk web agentes
uv run python scripts/grabar_traza_agente.py
uv run python scripts/build_spa_agente.py
uv run python scripts/build_agente_notebook.py
```

## Sesiones 09 a 12: difusión sin prisa

Cuatro cuadernos cortos que acompañan a la sesión 07. Los cuatro salen de un generador en `scripts/` con la verificación de redacción incluida.

- `09_ruido_hotot.ipynb` suma ruido a una foto en 500 pasos y anima la imagen, el histograma del tensor y un parche de 6×6 con sus valores. La foto va en `data/raw/hotot/hotot.jpg` y no se versiona, así que el cuaderno se publica sin salidas; sirve cualquier JPG.
- `10_difusion_simple.ipynb` entrena e infiere con 2000 puntos en forma de corazón y un `MLPRegressor` de scikit-learn. Corre en local en segundos.
- `11_difusion_video.ipynb` trata cada video (8 cuadros de 10×10 con una pelota en movimiento) como un solo tensor. La red predice el video limpio. Corre en local en menos de un minuto.
- `12_unet_animefaces.ipynb` entrena la U-Net condicional de `scripts/train_ddpm.py` con 12 000 caras de [`huggan/anime-faces`](https://huggingface.co/datasets/huggan/anime-faces) (CC0 según su ficha, 32×32) y etiquetas automáticas por color de cabello. Necesita GPU; tarda unos 5 minutos. El zip se descarga a `data/animefaces/`, que no se versiona.

```bash
uv run python scripts/build_ruido_notebook.py
uv run python scripts/build_difusion_simple_notebook.py
uv run python scripts/build_difusion_video_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/10_difusion_simple.ipynb notebooks/11_difusion_video.ipynb

uv run python scripts/build_unet_anime_notebook.py
scp scripts/train_ddpm.py $GPU_HOST:$RUTA_REMOTA/scripts/
scp notebooks/12_unet_animefaces.ipynb $GPU_HOST:$RUTA_REMOTA/notebooks/
ssh $GPU_HOST 'docker exec -w /workspace/friendly-fiesta lab-pytorch jupyter nbconvert --to notebook --execute --inplace notebooks/12_unet_animefaces.ipynb'
scp $GPU_HOST:$RUTA_REMOTA/notebooks/12_unet_animefaces.ipynb notebooks/
```

## Textos de la Biblia y el Corán

Los textos originales no se incluyen en este repositorio, porque las traducciones pueden tener derechos de terceros. La SPA funciona sin ellos con los datos ya calculados de `data/processed/` (frecuencias, lemas y fragmentos breves). Para regenerar esos datos o reentrenar la LSTM, coloca tus propias copias en `data/raw/textos_nlp/biblia_es.txt` y `data/raw/textos_nlp/coran_es.txt`.

## Licencia

Este proyecto se distribuye bajo la licencia GNU GPL v3.0 (`GPL-3.0-only`). El texto completo está en [`LICENSE`](LICENSE).
