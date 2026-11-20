"""Graba la traza completa del agente asistente_teresa sobre preguntas fijas.

Ejecuta el agente con Runner e InMemorySessionService en una sola sesión (para
que el estado se conserve entre turnos) y guarda, por cada evento: autor, texto,
llamadas a herramienta con argumentos, respuestas de herramienta, cambios de
estado, marca de tiempo, segundos desde el inicio del turno y tokens.

También registra:
- el esquema JSON de cada herramienta tal como ADK lo arma (FunctionDeclaration)
  y tal como LiteLLM lo envía a Ollama (formato de herramientas de OpenAI);
- cada petición al modelo (instrucción de sistema y número de mensajes);
- una corrida de la pregunta de encadenamiento con un agente sin herramientas.

Requiere el túnel SSH a Ollama:
    ssh -N -L 11434:localhost:11434 $GPU_HOST

Uso:
    uv run python scripts/grabar_traza_agente.py
    uv run python scripts/grabar_traza_agente.py --modelo ollama_chat/qwen3:4b --salida data/agente_pruebas/traza_qwen3.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agentes"))

from google.adk.models.lite_llm import _function_declaration_to_tool_param  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402

from asistente_teresa import agent as modulo  # noqa: E402

SALIDA = REPO_ROOT / "data" / "processed" / "traza_agente.json"
APP = "asistente_teresa"
USUARIO = "estudiante"

PREGUNTAS = [
    {"id": "sin_herramientas", "objetivo": "Saludo: el modelo responde directo, sin herramientas.",
     "texto": "Hola, ¿qué puedes hacer?"},
    {"id": "una_herramienta", "objetivo": "Exige una herramienta: buscar_personaje.",
     "texto": "¿Quién interpreta a Teresa en la telenovela de 2010?"},
    {"id": "encadenada", "objetivo": "Encadena dos herramientas: consultar_sesion y después calcular.",
     "texto": "¿Cuántos días faltan para la sesión 8 del curso? Convierte esos días a horas."},
    {"id": "estado_escribe", "objetivo": "Segunda búsqueda: agrega otro personaje al estado.",
     "texto": "¿Y qué actor hace de Arturo de la Barrera?"},
    {"id": "estado_lee", "objetivo": "Usa el estado: la instrucción recibe la lista de personajes consultados.",
     "texto": "¿Qué personajes te he preguntado hasta ahora en esta conversación?"},
]


def esquemas_herramientas(agente) -> list[dict]:
    """Esquema de cada herramienta: declaración de ADK y forma enviada por LiteLLM."""
    salida = []
    for f in agente.tools:
        decl = FunctionTool(f)._get_declaration()
        salida.append({
            "nombre": decl.name,
            "adk_function_declaration": decl.model_dump(mode="json", exclude_none=True),
            "litellm_tool": _function_declaration_to_tool_param(decl),
        })
    return salida


def evento_a_dict(ev, t0: float) -> dict:
    """Convierte un Event de ADK en un diccionario serializable."""
    partes = []
    if ev.content and ev.content.parts:
        for p in ev.content.parts:
            if p.function_call:
                partes.append({"tipo": "function_call", "id": p.function_call.id,
                               "nombre": p.function_call.name, "args": p.function_call.args})
            elif p.function_response:
                partes.append({"tipo": "function_response", "id": p.function_response.id,
                               "nombre": p.function_response.name, "respuesta": p.function_response.response})
            elif p.text:
                partes.append({"tipo": "pensamiento" if p.thought else "texto", "texto": p.text})
    uso = None
    if ev.usage_metadata:
        uso = {"entrada": ev.usage_metadata.prompt_token_count,
               "salida": ev.usage_metadata.candidates_token_count,
               "total": ev.usage_metadata.total_token_count}
    return {
        "id": ev.id,
        "autor": ev.author,
        "rol": ev.content.role if ev.content else None,
        "partes": partes,
        "estado_delta": dict(ev.actions.state_delta) if ev.actions and ev.actions.state_delta else {},
        "final": ev.is_final_response(),
        "timestamp": ev.timestamp,
        "t_rel_s": round(ev.timestamp - t0, 3),
        "tokens": uso,
    }


async def correr(agente, preguntas: list[dict], peticiones: list[dict]) -> tuple[list[dict], dict]:
    """Corre las preguntas en una misma sesión y devuelve los turnos y el estado final."""
    servicio = InMemorySessionService()
    runner = Runner(agent=agente, app_name=APP, session_service=servicio)
    sesion = await servicio.create_session(app_name=APP, user_id=USUARIO)
    turnos = []
    for q in preguntas:
        peticiones.append({"marca": "turno", "pregunta": q["id"]})
        t0 = time.time()
        mensaje = types.Content(role="user", parts=[types.Part(text=q["texto"])])
        eventos = []
        async for ev in runner.run_async(user_id=USUARIO, session_id=sesion.id, new_message=mensaje):
            eventos.append(evento_a_dict(ev, t0))
        dur = round(time.time() - t0, 3)
        final = next((p["texto"] for e in reversed(eventos) if e["final"] for p in e["partes"] if p["tipo"] == "texto"), "")
        actual = await servicio.get_session(app_name=APP, user_id=USUARIO, session_id=sesion.id)
        turnos.append({**q, "t0": t0, "duracion_s": dur, "eventos": eventos,
                       "respuesta_final": final, "estado_despues": dict(actual.state)})
        print(f"[{q['id']}] {dur:.1f} s, {len(eventos)} eventos: {final[:90]!r}")
    actual = await servicio.get_session(app_name=APP, user_id=USUARIO, session_id=sesion.id)
    return turnos, dict(actual.state)


def registrador(peticiones: list[dict]):
    """Callback before_model: anota qué recibe el modelo en cada llamada."""
    def antes_del_modelo(callback_context, llm_request):
        cfg = llm_request.config
        nombres = []
        for t in (cfg.tools or []):
            nombres += [d.name for d in (t.function_declarations or [])]
        si = cfg.system_instruction
        texto_si = si if isinstance(si, str) else "".join(p.text or "" for p in si.parts) if si else ""
        peticiones.append({
            "marca": "llm",
            "t": time.time(),
            "mensajes": len(llm_request.contents),
            "herramientas": nombres,
            "instruccion_sistema": texto_si,
        })
        return None
    return antes_del_modelo


async def principal(modelo: str, salida: Path) -> None:
    peticiones: list[dict] = []
    con = modulo.construir_agente(modelo=modelo)
    con.before_model_callback = registrador(peticiones)
    t_inicio = time.time()
    turnos, estado = await correr(con, PREGUNTAS, peticiones)

    peticiones_sin: list[dict] = []
    sin = modulo.construir_agente(modelo=modelo, con_herramientas=False)
    sin.before_model_callback = registrador(peticiones_sin)
    encadenada = [q for q in PREGUNTAS if q["id"] == "encadenada"]
    turnos_sin, _ = await correr(sin, encadenada, peticiones_sin)
    turno_con = next(t for t in turnos if t["id"] == "encadenada")

    datos = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hoy": os.getenv("AGENTE_HOY", date.today().isoformat()),
        "modelo": modelo,
        "adk_version": __import__("google.adk").adk.__version__,
        "agente": {"nombre": con.name, "descripcion": con.description,
                   "instruccion": modulo.INSTRUCCION,
                   "herramientas": [f.__name__ for f in con.tools]},
        "esquemas": esquemas_herramientas(con),
        "turnos": turnos,
        "estado_final": estado,
        "peticiones_llm": peticiones,
        "duracion_total_s": round(time.time() - t_inicio, 3),
        "comparacion": {
            "pregunta": encadenada[0]["texto"],
            "con_herramientas": {"respuesta": turno_con["respuesta_final"], "duracion_s": turno_con["duracion_s"],
                                 "eventos": turno_con["eventos"]},
            "sin_herramientas": {"respuesta": turnos_sin[0]["respuesta_final"], "duracion_s": turnos_sin[0]["duracion_s"],
                                 "eventos": turnos_sin[0]["eventos"],
                                 "instruccion": sin.instruction},
        },
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Traza escrita en {salida.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", default=modulo.MODELO)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    a = ap.parse_args()
    salida = a.salida if a.salida.is_absolute() else REPO_ROOT / a.salida
    asyncio.run(principal(a.modelo, salida))
