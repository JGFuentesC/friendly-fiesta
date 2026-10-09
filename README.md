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

## Ejecución en GPU

El notebook `02_mlp_backprop.ipynb` entrena con NumPy, PyTorch y TensorFlow. La versión GPU corre en `platypy` (servidor AI-Lab con RTX 3070) dentro del contenedor `lab-pytorch`, que trae los tres marcos y Plotly.

```bash
ssh platypy 'cd ~/platypy/services/pytorch && docker compose up -d'
ssh -L 8889:localhost:8889 platypy
```

Luego abrir `http://localhost:8889` en el navegador local.
