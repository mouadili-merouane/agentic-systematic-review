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


class QuotaExceeded(RuntimeError):
    """Quota journalier du fournisseur atteint : inutile de réessayer, changer de modèle ou attendre."""


def _is_daily_quota(e):
    msg = str(e).lower()
    return any(k in msg for k in ["per day", "tpd", "rpd", "daily"])


def _is_retryable(e):
    msg = str(e).lower()
    if isinstance(e, QuotaExceeded):
        return False
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
    try:
        resp = llm.invoke([("system", system), ("user", user)])
    except Exception as e:
        if _is_daily_quota(e):
            raise QuotaExceeded(f"Quota journalier atteint : {str(e)[:300]}") from e
        raise
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


# ---------------------------------------------------------------- bascule automatique entre modèles
def _is_unavailable(e):
    msg = str(e).lower()
    return any(k in msg for k in ["model_not_found", "decommissioned", "does not exist", "not found"])


class FallbackLLM:
    """Chaîne de modèles : si le quota journalier d'un modèle est atteint (ou s'il n'existe plus),
    on passe automatiquement au suivant. Chaque bascule est signalée (piste d'audit)."""

    def __init__(self, names, provider, on_switch=None, temperature=0):
        self.names, self.provider, self.on_switch = list(names), provider, on_switch
        self.models = {}
        self.i = 0
        self.temperature = temperature

    @property
    def current(self):
        return self.names[self.i]

    def _model(self):
        if self.current not in self.models:
            self.models[self.current] = make_llm(self.current, self.provider, self.temperature)
        return self.models[self.current]

    def invoke(self, messages):
        while True:
            try:
                return self._model().invoke(messages)
            except Exception as e:
                if (_is_daily_quota(e) or _is_unavailable(e)) and self.i + 1 < len(self.names):
                    old = self.current
                    self.i += 1
                    print(f"\n↪ {old} indisponible ({'quota' if _is_daily_quota(e) else 'modèle'}) -> {self.current}")
                    if self.on_switch:
                        self.on_switch(old, self.current, str(e)[:200])
                    continue
                raise
