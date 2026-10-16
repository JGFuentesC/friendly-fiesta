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

El notebook `notebooks/03_fraude_ff.ipynb` entrena la red FF (PyTorch + GPU) e ilustra la distinción entre hiperparámetros y parámetros, con una búsqueda pequeña de hiperparámetros. Está generado por `scripts/build_fraud_notebook.py` y se ejecuta en platypy:

```bash
uv run python scripts/build_fraud_notebook.py
scp notebooks/03_fraude_ff.ipynb data/processed/fraud_train.csv data/processed/fraud_test.csv \
  platypy:/home/jose/platypy/services/pytorch/notebooks/friendly-fiesta/
ssh platypy 'docker exec lab-pytorch bash -lc "cd /workspace/friendly-fiesta && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/03_fraude_ff.ipynb"'
```

## Ejecución en GPU

El notebook `02_mlp_backprop.ipynb` entrena con NumPy, PyTorch y TensorFlow. La versión GPU corre en `platypy` (servidor AI-Lab con RTX 3070) dentro del contenedor `lab-pytorch`, que trae los tres marcos y Plotly.

```bash
ssh platypy 'cd ~/platypy/services/pytorch && docker compose up -d'
ssh -L 8889:localhost:8889 platypy
```

Luego abrir `http://localhost:8889` en el navegador local.
