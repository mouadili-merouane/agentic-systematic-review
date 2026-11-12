"""Accès LLM : limitation de débit, relances automatiques, sortie JSON validée par Pydantic.

Indépendant du fournisseur (Groq, Gemini, OpenAI...) : tout passe par init_chat_model.
"""
import json, re, time, threading
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 2.0  # secondes entre deux appels (quota gratuit Groq)


def make_llm(model, provider, temperature=0):
    from langchain.chat_models import init_chat_model
    return init_chat_model(model, model_provider=provider, temperature=temperature)


def _is_retryable(e):
    msg = str(e).lower()
    return any(k in msg for k in ["rate", "429", "limit", "timeout", "503", "502", "overloaded", "connection"])


def _throttle():
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


@retry(retry=retry_if_exception(_is_retryable), wait=wait_exponential(min=5, max=90), stop=stop_after_attempt(8),
       reraise=True)
def call(llm, system, user):
    _throttle()
    resp = llm.invoke([("system", system), ("user", user)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def _extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("Aucun objet JSON dans la réponse")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return json.loads(text[start:i + 1])
    raise ValueError("JSON incomplet")


def call_json(llm, system, user, schema: type[BaseModel], attempts=3):
    """Demande une réponse JSON conforme au schéma ; renvoie l'objet Pydantic validé."""
    schema_txt = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    sys_full = (f"{system}\n\nRespond ONLY with one JSON object that validates against this JSON schema "
                f"(no prose, no markdown):\n{schema_txt}")
    msg, last_err = user, None
    for _ in range(attempts):
        raw = call(llm, sys_full, msg)
        try:
            return schema.model_validate(_extract_json(raw))
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_err = e
            msg = f"{user}\n\nYour previous answer was invalid ({str(e)[:300]}). Return ONLY the corrected JSON."
    raise RuntimeError(f"Sortie JSON invalide après {attempts} essais : {last_err}")
