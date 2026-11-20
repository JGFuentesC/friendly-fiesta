"""Genera notebooks/08_agente_adk.ipynb a partir de celdas definidas aquí.

Sesión 08: creación de un agente simple con Google ADK 2.x. El cuaderno corre en
local (no necesita GPU) y habla con Ollama del servidor del laboratorio a través
de un túnel SSH:

    ssh -N -L 11434:localhost:11434 $GPU_HOST

Para dejar la versión ejecutada:

    uv run python scripts/build_agente_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=900 notebooks/08_agente_adk.ipynb

El generador falla si un título interrogativo o una pregunta no lleva ¿.
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "08_agente_adk.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

# ---------------------------------------------------------------------------
cells.append(md(r"""# Sesión 08. Un agente simple con Google ADK

Un **modelo de lenguaje** (LLM) estima una distribución condicional $p_\theta(y \mid x)$: recibe un texto $x$ y genera una continuación $y$ token por token. Por sí solo solo produce texto.

Un **agente** envuelve al modelo en un bucle de control. En cada paso $t$ el agente mira el historial $h_t$ (mensajes, resultados de herramientas, estado) y elige una acción $a_t$ según una política

$$a_t \sim \pi_\theta(a \mid h_t), \qquad a_t \in \{\text{responder}\} \cup \{\text{llamar a } f_1, \dots, f_K\}.$$

Si la acción es una llamada a herramienta, el programa ejecuta la función, obtiene una observación $o_t$ y la agrega al historial: $h_{t+1} = h_t \oplus (a_t, o_t)$. El bucle termina cuando el modelo responde en texto. Este es el patrón **ReAct** (razonar, actuar, observar).

En este cuaderno vamos a:

1. Escribir tres herramientas como funciones de Python con type hints y docstring.
2. Ver el **esquema JSON** que ADK genera para cada una y que recibe el modelo.
3. Construir un `LlmAgent` con un modelo local (`qwen3:4b` en Ollama, vía LiteLLM).
4. Ejecutarlo con `Runner` e `InMemorySessionService`, y leer los **eventos** uno por uno.
5. Usar el **estado de sesión** como memoria entre turnos.
6. Comparar la misma pregunta con y sin herramientas.
7. Abrir el agente en la interfaz de desarrollo `adk web`.

El agente vive en `agentes/asistente_teresa/agent.py` y responde sobre la telenovela *Teresa* (2010) y el calendario del curso. El cuaderno se genera con `scripts/build_agente_notebook.py`; si quieres cambiar una celda, cambia el script y regenera."""))

cells.append(md(r"""## 1. Preparación

Antes de ejecutar, abre el túnel al servidor del laboratorio en otra terminal:

```bash
ssh -N -L 11434:localhost:11434 $GPU_HOST
```

Con el túnel activo, `http://localhost:11434` es el Ollama del servidor. LiteLLM lee esa dirección de la variable `OLLAMA_API_BASE`."""))

cells.append(code(r"""import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")


def find_repo_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / "agentes" / "asistente_teresa").exists():
            return parent
    return p


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "agentes"))

with urllib.request.urlopen(os.environ["OLLAMA_API_BASE"] + "/api/tags", timeout=10) as r:
    modelos = sorted(m["name"] for m in json.load(r)["models"])
print("Ollama responde. Modelos con 'qwen':", [m for m in modelos if "qwen" in m])

import google.adk
print("google-adk", google.adk.__version__)"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 2. Una herramienta es una función

Para ADK una herramienta es una función de Python común. Tres cosas importan:

- el **nombre** de la función, que el modelo usará para pedirla;
- los **type hints** de los argumentos, que se vuelven tipos del esquema JSON;
- la **docstring**, que se vuelve la descripción. El modelo decide cuándo llamar a la herramienta leyendo esa descripción, así que es parte del *prompt*.

Las herramientas deben ser deterministas y devolver diccionarios con un campo `status`, para que el modelo distinga un resultado válido de un error.

`calcular` evalúa expresiones aritméticas recorriendo el árbol sintáctico con `ast`. Usar `eval` sobre texto que escribió un modelo permitiría ejecutar código arbitrario."""))

cells.append(code(r"""from asistente_teresa import agent as modulo

print(modulo.calcular("7 * 24"))
print(modulo.calcular("(2 + 3) ** 2 / 4"))
print(modulo.calcular("__import__('os').system('ls')"))"""))

cells.append(md(r"""`buscar_personaje` y `consultar_sesion` reciben además un parámetro `tool_context`. ADK lo inyecta al ejecutar la herramienta y lo oculta del esquema: el modelo nunca lo ve. A través de `tool_context.state` la herramienta lee y escribe el **estado de la sesión**.

Para probarlas fuera del agente basta un objeto con un atributo `state`."""))

cells.append(code(r"""class ContextoFalso:
    def __init__(self):
        self.state = {}


ctx = ContextoFalso()
print(modulo.buscar_personaje("aida", ctx))
print(modulo.buscar_personaje("Mariano Sánchez", ctx))
print(modulo.buscar_personaje("Pedro Infante", ctx))
print("estado:", ctx.state)
print(modulo.consultar_sesion(8, ctx))"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 3. El esquema JSON que recibe el modelo

ADK envuelve cada función en un `FunctionTool` y arma una `FunctionDeclaration` (nombre, descripción y parámetros en JSON Schema). LiteLLM la traduce al formato de herramientas de OpenAI, que es el que entiende Ollama. Observa que `tool_context` no aparece."""))

cells.append(code(r"""from google.adk.models.lite_llm import _function_declaration_to_tool_param
from google.adk.tools import FunctionTool

decl = FunctionTool(modulo.consultar_sesion)._get_declaration()
print(json.dumps(_function_declaration_to_tool_param(decl), ensure_ascii=False, indent=2))"""))

cells.append(md(r"""Con el esquema, el modelo convierte una pregunta en una llamada estructurada. Formalmente, el modelo genera una cadena $c$ que el programa decodifica como $(f, \mathbf{u})$, el nombre de la herramienta y sus argumentos; la llamada es válida si $f$ existe y $\mathbf{u}$ cumple el esquema de $f$. La herramienta devuelve $o = f(\mathbf{u})$.

Todas las herramientas viajan en **cada** petición al modelo y consumen tokens de entrada."""))

cells.append(code(r"""for f in [modulo.buscar_personaje, modulo.calcular, modulo.consultar_sesion]:
    d = FunctionTool(f)._get_declaration()
    texto = json.dumps(_function_declaration_to_tool_param(d), ensure_ascii=False)
    print(f"{d.name:18s} {len(texto):5d} caracteres  parámetros: {list(d.parameters_json_schema['properties'])}")"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 4. El `LlmAgent`

Un `LlmAgent` reúne cuatro piezas:

| Campo | Papel |
|---|---|
| `model` | El LLM. Con `LiteLlm(model="ollama_chat/qwen3:4b")` usamos un modelo local. |
| `instruction` | El mensaje de sistema. Admite plantillas `{clave?}` que ADK rellena con el estado. |
| `tools` | La lista de funciones disponibles. |
| `name`, `description` | Identifican al agente en los eventos y cuando hay varios agentes. |

Elegimos `qwen3:4b` (2.4 GB) porque en las pruebas encadenó dos herramientas de forma fiable. `qwen2.5:7b` llamó a la primera herramienta y luego hizo la cuenta de memoria."""))

cells.append(code(r"""agente = modulo.construir_agente()
print("nombre:", agente.name)
print("modelo:", agente.model.model)
print("herramientas:", [t.__name__ for t in agente.tools])
print()
print(agente.instruction)"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 5. `Runner`, sesión y eventos

El `Runner` ejecuta el bucle ReAct. Para cada mensaje del usuario:

1. arma la petición (instrucción, historial de la sesión, esquemas);
2. llama al modelo;
3. si el modelo pide herramientas, las ejecuta y vuelve al paso 1;
4. si el modelo responde en texto, termina.

Cada paso queda como un **evento** (`Event`) con autor, contenido (texto, `function_call` o `function_response`), cambios de estado (`actions.state_delta`) y tokens. El `SessionService` guarda la sesión: historial de eventos y estado. `InMemorySessionService` la guarda en memoria y se pierde al reiniciar."""))

cells.append(code(r"""from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP, USUARIO = "asistente_teresa", "estudiante"
servicio = InMemorySessionService()
runner = Runner(agent=agente, app_name=APP, session_service=servicio)
sesion = await servicio.create_session(app_name=APP, user_id=USUARIO)
print("sesión:", sesion.id, "estado inicial:", sesion.state)


def describir(ev, t0):
    pref = f"[{ev.timestamp - t0:6.1f} s] {ev.author:18s}"
    for p in (ev.content.parts if ev.content else []):
        if p.thought and p.text:
            print(pref, "PENSAMIENTO:", p.text.strip().replace("\n", " ")[:160], "...")
        elif p.function_call:
            print(pref, "LLAMADA     :", p.function_call.name, json.dumps(p.function_call.args, ensure_ascii=False))
        elif p.function_response:
            print(pref, "OBSERVACIÓN :", json.dumps(p.function_response.response, ensure_ascii=False))
        elif p.text:
            print(pref, "TEXTO       :", p.text.strip())
    if ev.actions and ev.actions.state_delta:
        print(pref, "ESTADO Δ    :", dict(ev.actions.state_delta))
    if ev.usage_metadata:
        u = ev.usage_metadata
        print(pref, f"tokens      : entrada {u.prompt_token_count}, salida {u.candidates_token_count}")


async def preguntar(texto, runner=runner, sesion=sesion):
    print("USUARIO:", texto)
    t0 = time.time()
    eventos = []
    msg = types.Content(role="user", parts=[types.Part(text=texto)])
    async for ev in runner.run_async(user_id=USUARIO, session_id=sesion.id, new_message=msg):
        eventos.append(ev)
        describir(ev, t0)
    print(f"--- {len(eventos)} eventos en {time.time() - t0:.1f} s")
    return eventos"""))

cells.append(md(r"""### Turno 1: una pregunta sin herramientas

Un saludo no requiere datos externos: el modelo responde en el primer paso y el bucle termina. `qwen3` es un modelo con razonamiento; sus tokens de pensamiento llegan como partes con `thought=True`."""))
cells.append(code(r"""ev1 = await preguntar("Hola, ¿qué puedes hacer?")"""))

cells.append(md(r"""### Turno 2: una herramienta

Ahora el modelo debe pedir `buscar_personaje`. Verás tres eventos: la llamada (autor: el agente), la observación (la respuesta de la función) y la respuesta final, que ya incorpora el dato."""))
cells.append(code(r"""ev2 = await preguntar("¿Quién interpreta a Teresa en la telenovela de 2010?")"""))

cells.append(md(r"""### Turno 3: dos herramientas encadenadas

Para convertir a horas los días que faltan, el modelo necesita primero el dato de `consultar_sesion` y luego `calcular`. Es un bucle ReAct de dos vueltas: pensar, actuar, observar, pensar otra vez, actuar, observar, responder."""))
cells.append(code(r"""ev3 = await preguntar("¿Cuántos días faltan para la sesión 8 del curso? Convierte esos días a horas.")"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 6. Estado de sesión como memoria

El historial de mensajes ya es una memoria, pero crece y se mezcla con todo lo demás. El **estado** es un diccionario explícito asociado a la sesión: las herramientas lo escriben con `tool_context.state[...] = ...` y ADK registra cada cambio en `actions.state_delta`.

La instrucción del agente contiene `{personajes_consultados?}`. Antes de cada llamada al modelo, ADK sustituye esa plantilla por el valor actual del estado; el signo `?` evita un error cuando la clave todavía no existe."""))
cells.append(code(r"""ev4 = await preguntar("¿Y qué actor hace de Arturo de la Barrera?")
actual = await servicio.get_session(app_name=APP, user_id=USUARIO, session_id=sesion.id)
print("\nestado de la sesión:", actual.state)"""))
cells.append(code(r"""ev5 = await preguntar("¿Qué personajes te he preguntado hasta ahora en esta conversación?")"""))

cells.append(md(r"""### La sesión completa

Todos los eventos de los cinco turnos quedan guardados en la sesión. Contamos cuántos son de cada tipo y cuántos tokens consumió el modelo."""))
cells.append(code(r"""import pandas as pd

actual = await servicio.get_session(app_name=APP, user_id=USUARIO, session_id=sesion.id)
filas = []
for ev in actual.events:
    for p in (ev.content.parts if ev.content else []):
        tipo = "function_call" if p.function_call else "function_response" if p.function_response else "pensamiento" if p.thought else "texto"
        filas.append({"autor": ev.author, "tipo": tipo,
                      "tokens_entrada": ev.usage_metadata.prompt_token_count if ev.usage_metadata else None,
                      "tokens_salida": ev.usage_metadata.candidates_token_count if ev.usage_metadata else None})
df = pd.DataFrame(filas)
print(df.groupby(["autor", "tipo"]).size())
consultas = [ev.usage_metadata for ev in actual.events if ev.usage_metadata]
print(f"\n{len(consultas)} consultas al modelo; tokens de entrada:", sum(u.prompt_token_count for u in consultas),
      "; tokens de salida:", sum(u.candidates_token_count for u in consultas))"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 7. Con y sin herramientas

Repetimos la pregunta del turno 3 con un agente idéntico sin herramientas y con la instrucción recortada. El modelo no tiene forma de saber la fecha de hoy ni el calendario: puede negarse o inventar."""))
cells.append(code(r"""sin = modulo.construir_agente(con_herramientas=False)
servicio_sin = InMemorySessionService()
runner_sin = Runner(agent=sin, app_name=APP, session_service=servicio_sin)
sesion_sin = await servicio_sin.create_session(app_name=APP, user_id=USUARIO)
ev_sin = await preguntar("¿Cuántos días faltan para la sesión 8 del curso? Convierte esos días a horas.",
                         runner=runner_sin, sesion=sesion_sin)"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 8. La traza grabada y la interfaz `adk web`

`scripts/grabar_traza_agente.py` repite estas preguntas y guarda todos los eventos, los esquemas y la comparación en `data/processed/traza_agente.json`. Esa traza alimenta la SPA `spa/agente_adk.html` y el apunte de la sesión."""))
cells.append(code(r"""traza = json.loads((ROOT / "data" / "processed" / "traza_agente.json").read_text(encoding="utf-8"))
print("modelo:", traza["modelo"], " fecha de referencia:", traza["hoy"])
for t in traza["turnos"]:
    pasos = [p["nombre"] for e in t["eventos"] for p in e["partes"] if p["tipo"] == "function_call"]
    print(f"{t['id']:17s} {t['duracion_s']:6.1f} s  herramientas: {pasos}")"""))

cells.append(md(r"""ADK trae dos formas de usar el agente sin escribir el `Runner` a mano. Desde la raíz del repositorio y con el túnel activo:

```bash
uv run adk run agentes/asistente_teresa    # conversación en la terminal
uv run adk web agentes                     # interfaz en http://localhost:8000
```

`adk web` busca paquetes dentro de `agentes/` que expongan `root_agent` (por eso `__init__.py` contiene `from . import agent`). En la pestaña **Events** de la interfaz se ve el mismo flujo de eventos de este cuaderno, con cada `function_call` y `function_response`, y en **State** el estado de la sesión."""))

cells.append(md(r"""## 9. Ejercicios

1. Agrega una cuarta herramienta `listar_sesiones()` que devuelva todos los temas del curso. ¿Cambia el número de tokens de entrada de cada petición? Mídelo con `usage_metadata`.
2. Quita la docstring de `calcular` y repite el turno 3. ¿El modelo sigue llamando a la herramienta?
3. Cambia `AGENTE_MODELO` a `ollama_chat/qwen2.5:7b` y repite los cinco turnos. ¿En cuáles se comporta distinto?
4. Haz que `consultar_sesion` acepte también el tema (por ejemplo "difusión") y devuelva el número de sesión. Ajusta los type hints para que el esquema lo refleje.
5. Usa `before_tool_callback` para impedir que `calcular` reciba expresiones de más de 50 caracteres y observa qué hace el modelo con el rechazo."""))

nb.cells = cells

# Verificación de la regla de signos de interrogación en encabezados y títulos.
INTERROG = r"(qué|quién|quiénes|cómo|dónde|cuándo|cuánto|cuántos|cuántas|cuál|cuáles|por qué|con qué|contra qué)"
problemas = []
for i, c in enumerate(nb.cells):
    if c.cell_type != "markdown":
        continue
    texto = re.sub(r"```.*?```", "", c.source, flags=re.DOTALL)
    for line in texto.splitlines():
        s = line.strip()
        m = re.match(r"^(#+\s*|\d+\.\s*|-\s*)?(\*\*)?(.*)$", s)
        body = (m.group(3) if m else s).strip()
        if re.match(rf"^{INTERROG}\b", body, flags=re.IGNORECASE) and not body.startswith("¿"):
            problemas.append((i, s))
        if "?" in re.sub(r"\{[^}]*\?\}|`[^`]*`", "", body) and "¿" not in body:
            problemas.append((i, s))
if problemas:
    for p in problemas:
        print("Pregunta sin ¿ en la celda", p[0], ":", p[1], file=sys.stderr)
    sys.exit(1)

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("escrito", OUT.relative_to(REPO_ROOT), f"({len(nb.cells)} celdas)")
