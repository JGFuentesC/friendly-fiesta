# Seguridad

Para reportar una vulnerabilidad o un secreto expuesto en este repositorio, abre un aviso privado desde la pestaña **Security > Report a vulnerability** de GitHub. Evita los issues públicos para este tipo de reportes.

## Controles automáticos

El flujo `.github/workflows/seguridad.yml` corre en cada push a `main`, en cada pull request y cada lunes:

- **gitleaks** revisa todo el historial en busca de llaves, tokens y contraseñas.
- **pip-audit** compara las versiones fijadas en `uv.lock` con las bases de vulnerabilidades de PyPI y OSV.
- **bandit** hace análisis estático de `scripts/`, `agentes/` y `observatorio/`.

Dependabot propone cada semana actualizaciones de dependencias de Python y de las acciones de GitHub.

## Revisión local antes de publicar

```bash
gitleaks git . --redact          # historial completo
gitleaks dir . --redact          # árbol de trabajo (`.gitleaks.toml` excluye `.venv`)
uv export --frozen --no-hashes --format requirements-txt > requisitos.txt
uvx pip-audit --no-deps --disable-pip -r requisitos.txt
uvx bandit -q -r scripts agentes observatorio -ll
```

Con `pre-commit install` el gancho de gitleaks revisa cada commit antes de crearlo.

## Excepciones conocidas

- `setuptools` queda fijado en `<81` en `pyproject.toml` de forma deliberada desde el commit 927f9bc. Sus avisos (`PYSEC-2026-3447`, `GHSA-xmj3-9gf7-h52w`) afectan la instalación de paquetes y se ignoran en la auditoría automática hasta poder liberar el límite.
- bandit marca `urllib.request.urlopen` (B310) en `observatorio/`. Las direcciones salen de una constante o de las variables `OLLAMA_URL` u `OLLAMA_API_BASE` que fija quien corre el servicio, así que la regla se omite en el flujo.

## Datos que nunca se versionan

Credenciales (`.env`, `.envrc`, `*.pem`, `*.key`, `credentials*.json`), textos con derechos de terceros (corpus de *Teresa*, traducciones de la Biblia y el Corán), comentarios descargados de YouTube, la foto de entrada de la sesión 09 (`data/raw/hotot/`), el zip de caras de anime de la sesión 12 (`data/animefaces/`) y checkpoints de modelos. Las reglas están en `.gitignore`.
