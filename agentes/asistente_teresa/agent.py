"""Asistente Teresa: agente simple con Google ADK 2.x y un modelo local de Ollama.

Responde preguntas sobre la telenovela "Teresa" (2010) y sobre el calendario del
curso. Tiene tres herramientas deterministas escritas en Python:

- buscar_personaje: consulta una ficha JSON de personajes y guarda en el estado
  de la sesión la lista de personajes ya consultados (memoria de corto plazo).
- calcular: evalúa una expresión aritmética con el módulo ast, sin eval.
- consultar_sesion: devuelve fecha, tema y días que faltan para una sesión.

ADK construye el esquema JSON de cada herramienta a partir de su firma (type
hints) y de su docstring; por eso las docstrings explican qué hace la función y
qué significa cada argumento. El parámetro tool_context lo inyecta ADK y no
aparece en el esquema que ve el modelo.

Variables de entorno:
    OLLAMA_API_BASE  URL de Ollama (por defecto http://localhost:11434, el túnel SSH).
    AGENTE_MODELO    modelo de LiteLLM (por defecto ollama_chat/qwen3:4b).
    AGENTE_HOY       fecha de referencia AAAA-MM-DD para consultar_sesion (opcional).

Uso:
    uv run adk run agentes/asistente_teresa
    uv run adk web agentes
"""

from __future__ import annotations

import ast
import json
import operator
import os
import unicodedata
from datetime import date
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")

REPO_ROOT = Path(__file__).resolve().parents[2]
FICHA = REPO_ROOT / "data" / "processed" / "teresa_personajes.json"
CALENDARIO = REPO_ROOT / "data" / "processed" / "curso_sesiones.json"
MODELO = os.getenv("AGENTE_MODELO", "ollama_chat/qwen3:4b")


def _normalizar(texto: str) -> str:
    """Pasa a minúsculas y quita tildes para comparar nombres con tolerancia."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.lower().split())


# ---------------------------------------------------------------------------
# Herramienta 1: búsqueda en la ficha de personajes (usa el estado de sesión)
# ---------------------------------------------------------------------------
def buscar_personaje(nombre: str, tool_context: ToolContext) -> dict:
    """Busca un personaje de la telenovela Teresa (2010) en la ficha del curso.

    Úsala cuando el usuario pregunte quién es un personaje, qué actor o actriz
    lo interpreta o qué papel tiene en la historia.

    Args:
        nombre: Nombre o apellido del personaje, por ejemplo "Teresa" o "Arturo".

    Returns:
        Un diccionario con status "ok" y los datos del personaje (nombre, actor,
        rol, descripción), o status "no_encontrado" con la lista de personajes
        disponibles en la ficha.
    """
    ficha = json.loads(FICHA.read_text(encoding="utf-8"))
    consulta = _normalizar(nombre)
    for p in ficha["personajes"]:
        claves = [_normalizar(a) for a in p["alias"]] + [_normalizar(p["nombre"])]
        if any(consulta == c or consulta in c for c in claves):
            consultados = list(tool_context.state.get("personajes_consultados", []))
            if p["nombre"] not in consultados:
                consultados.append(p["nombre"])
            tool_context.state["personajes_consultados"] = consultados
            datos = {k: p[k] for k in ("nombre", "actor", "rol", "descripcion")}
            return {"status": "ok", **datos, "total_personajes_consultados": len(consultados)}
    return {"status": "no_encontrado", "disponibles": [p["nombre"] for p in ficha["personajes"]]}


# ---------------------------------------------------------------------------
# Herramienta 2: calculadora segura con ast
# ---------------------------------------------------------------------------
_OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluar(nodo: ast.AST) -> float:
    """Recorre el árbol sintáctico y solo admite números y operadores aritméticos."""
    if isinstance(nodo, ast.Expression):
        return _evaluar(nodo.body)
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)) and not isinstance(nodo.value, bool):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        izquierda, derecha = _evaluar(nodo.left), _evaluar(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > 100:
            raise ValueError("exponente demasiado grande")
        return _OPERADORES[type(nodo.op)](izquierda, derecha)
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](_evaluar(nodo.operand))
    raise ValueError(f"elemento no permitido: {type(nodo).__name__}")


def calcular(expresion: str) -> dict:
    """Evalúa una expresión aritmética y devuelve el resultado exacto.

    Úsala para cualquier operación numérica en lugar de calcular de memoria.
    Admite números, paréntesis y los operadores + - * / // % **.

    Args:
        expresion: Expresión aritmética en notación de Python, por ejemplo "7 * 24".

    Returns:
        Un diccionario con status "ok", la expresión y el resultado, o status
        "error" con el motivo.
    """
    try:
        resultado = _evaluar(ast.parse(expresion, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        return {"status": "error", "expresion": expresion, "motivo": str(e)}
    if isinstance(resultado, float) and resultado.is_integer():
        resultado = int(resultado)
    return {"status": "ok", "expresion": expresion, "resultado": resultado}


# ---------------------------------------------------------------------------
# Herramienta 3: calendario del curso (guarda la última sesión consultada)
# ---------------------------------------------------------------------------
def consultar_sesion(numero: int, tool_context: ToolContext) -> dict:
    """Consulta el calendario del curso de Inteligencia Artificial Aplicada.

    Úsala cuando el usuario pregunte por la fecha o el tema de una sesión, o
    cuántos días faltan para ella.

    Args:
        numero: Número de la sesión, de 1 a 8.

    Returns:
        Un diccionario con status "ok", número, fecha (AAAA-MM-DD), tema y
        dias_restantes (negativo si la sesión ya pasó), o status "error".
    """
    calendario = json.loads(CALENDARIO.read_text(encoding="utf-8"))
    hoy = date.fromisoformat(os.getenv("AGENTE_HOY", date.today().isoformat()))
    for s in calendario["sesiones"]:
        if s["numero"] == int(numero):
            fecha = date.fromisoformat(s["fecha"])
            tool_context.state["ultima_sesion_consultada"] = s["numero"]
            return {
                "status": "ok",
                "numero": s["numero"],
                "fecha": s["fecha"],
                "tema": s["tema"],
                "hoy": hoy.isoformat(),
                "dias_restantes": (fecha - hoy).days,
            }
    return {"status": "error", "motivo": f"no existe la sesión {numero}; hay sesiones de 1 a {len(calendario['sesiones'])}"}


INSTRUCCION = """Eres el asistente del curso de Inteligencia Artificial Aplicada de la Universidad Anáhuac.
Respondes en español, con frases breves y claras y sin emojis, sobre dos temas: la telenovela Teresa (2010) y el calendario del curso.

Reglas:
1. Si te preguntan por un personaje, un actor o una actriz de Teresa, llama a buscar_personaje. No respondas de memoria.
2. Si te preguntan por la fecha, el tema o los días que faltan para una sesión, llama a consultar_sesion con el número de sesión.
3. Para cualquier operación aritmética llama a calcular; nunca hagas cuentas de memoria. Si la operación depende de un dato que da otra herramienta, primero llama a esa herramienta y después a calcular con el dato.
4. Si el usuario solo saluda o pregunta qué sabes hacer, responde sin herramientas.
5. Basa tu respuesta final en los resultados de las herramientas y menciona los datos concretos.

Memoria de esta conversación:
- Personajes consultados hasta ahora: {personajes_consultados?}
- Última sesión consultada: {ultima_sesion_consultada?}
Si te preguntan qué personajes han consultado, usa esa lista."""


def construir_agente(modelo: str = MODELO, con_herramientas: bool = True) -> LlmAgent:
    """Devuelve el LlmAgent; con_herramientas=False sirve para comparar sin herramientas."""
    return LlmAgent(
        name="asistente_teresa" if con_herramientas else "asistente_sin_herramientas",
        model=LiteLlm(model=modelo),
        description="Responde sobre la telenovela Teresa (2010) y el calendario del curso.",
        instruction=INSTRUCCION if con_herramientas else INSTRUCCION.split("\n\nReglas:")[0],
        tools=[buscar_personaje, calcular, consultar_sesion] if con_herramientas else [],
    )


root_agent = construir_agente()
