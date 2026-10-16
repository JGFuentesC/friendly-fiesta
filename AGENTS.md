# AGENTS.md

Repositorio de notebooks didácticos (posgrado) en español: mínimos cuadrados, SGD y backpropagation de un MLP. No es una librería: `package = false` en `pyproject.toml`, así que no hay `uv build` ni módulos importables.

## Entorno

- Gestión con [uv](https://docs.astral.sh/uv/). Python `>=3.12`.
- `uv sync` instala el entorno en `.venv`.
- `uv run python scripts/...` para ejecutar scripts.
- `uv run jupyter lab` para levantar los notebooks localmente.

## Comandos

```bash
uv sync
uv run python scripts/make_split.py      # regenera data/processed/train.csv y validation.csv
uv run python scripts/build_notebook.py  # regenera notebooks/02_mlp_backprop.ipynb
uv run jupyter lab
```

## Gotchas

- **`notebooks/02_mlp_backprop.ipynb` está GENERADO.** Su fuente es `scripts/build_notebook.py` (celdas definidas ahí con `nbformat`). Editar el `.ipynb` a mano se sobrescribe al regenerar; cambia el script, no el notebook. El notebook `01_minimos_cuadrados_sgd.ipynb` es manual.
- **Dependencias no declaradas.** `torch` y `tensorflow` se usan en `02_mlp_backprop.ipynb` pero NO están en `pyproject.toml`; solo existen en el contenedor GPU `lab-pytorch` (servidor `platypy`, RTX 3070). En local las celdas de PyTorch/TensorFlow fallan. `scipy` tampoco está declarado (entra como dependencia transitiva de `scikit-learn`). `plotly` + `kaleido` generan los PNG de `figures/`.
- **`data/raw/file.csv` es de ancho fijo** (separado por espacios, `sep=r"\s+"`), no CSV por comas. Columnas: `x1 x2 x3 y`.
- **Seed fija 42** en `make_split.py` (split 70/30) y en todo el MLP: los resultados y los archivos `train.csv`/`validation.csv` son deterministas.
- **`data/raw/ibm/` está gitignored** (entrada nueva aún sin commitear): datos de tarjetas de crédito IBM usados para un análisis de credit scoring aparte, no parte de los notebooks commiteados.
- Repo solo para macOS con acceso al AI-Lab; `README.md` documenta el túnel SSH al Jupyter remoto (`ssh -L 8889:localhost:8889 platypy`).
