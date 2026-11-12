"""Agents de sélection (screening).

- Titre/résumé : deux screeners indépendants (personas différentes, modèles pouvant différer),
  traitement par lots de N articles pour économiser le quota ; un arbitre tranche les désaccords.
- Texte intégral : décision fondée sur les passages récupérés par le RAG dans l'article.
Chaque décision porte les critères satisfaits / non satisfaits (I.., E..) et une justification.
"""
from typing import Literal
from pydantic import BaseModel, Field
from ..llm import call_json
from ..schemas import Protocol

Decision = Literal["include", "exclude", "uncertain"]


class ScreenDecision(BaseModel):
    id: str
    decision: Decision
    criteria_failed: list[str] = Field(default_factory=list,
                                       description="ids of inclusion criteria not met or exclusion criteria triggered")
    rationale: str = Field(description="1-2 sentences citing the evidence from the title/abstract")
    confidence: float = Field(ge=0, le=1)


class BatchDecisions(BaseModel):
    decisions: list[ScreenDecision]


PERSONAS = {
    "A": ("You are Screener A, a sensitive reviewer: when information is missing or ambiguous, "
          "prefer 'uncertain' over 'exclude' so that no relevant study is lost."),
    "B": ("You are Screener B, a rigorous reviewer: apply each criterion literally and exclude only on "
          "explicit evidence; use 'uncertain' when the abstract does not allow a decision."),
}


def _criteria(p: Protocol) -> str:
    inc = "\n".join(f"{c.id}: {c.text}" for c in p.inclusion_criteria)
    exc = "\n".join(f"{c.id}: {c.text}" for c in p.exclusion_criteria)
    return f"Research question: {p.research_question}\nINCLUSION (all required):\n{inc}\nEXCLUSION (any excludes):\n{exc}"


def _rec_txt(r, max_abs=1500):
    abs_ = (r.get("abstract") or "[no abstract available]")[:max_abs]
    return f"[id={r['id']}] TITLE: {r['title']} ({r.get('year')})\nVENUE: {r.get('venue','')}\nABSTRACT: {abs_}"


def screen_batch(llm, persona: str, p: Protocol, records: list[dict]) -> list[ScreenDecision]:
    system = (PERSONAS[persona] + "\nYou screen records for a systematic review at the title/abstract stage.\n"
              + _criteria(p) + "\nReturn one decision per record, using the exact record ids.")
    user = "\n\n---\n\n".join(_rec_txt(r) for r in records)
    out = call_json(llm, system, user, BatchDecisions).decisions
    got = {d.id: d for d in out}
    # sécurité : un id manquant -> 'uncertain' (sera arbitré), jamais une exclusion silencieuse
    return [got.get(r["id"]) or ScreenDecision(id=r["id"], decision="uncertain", rationale="no decision returned",
                                               confidence=0) for r in records]


def arbitrate(llm, p: Protocol, record: dict, a: dict, b: dict) -> ScreenDecision:
    system = ("You are the senior reviewer resolving a disagreement between two screeners at the "
              "title/abstract stage. Decide 'include' (go to full-text review) or 'exclude'. "
              "When genuine doubt remains, include: full-text review will settle it.\n" + _criteria(p))
    user = (_rec_txt(record, 3000) + f"\n\nScreener A: {a['decision']} — {a['rationale']}"
            f"\nScreener B: {b['decision']} — {b['rationale']}")
    d = call_json(llm, system, user, ScreenDecision)
    d.id = record["id"]
    if d.decision == "uncertain":
        d.decision = "include"
    return d


def screen_fulltext(llm, p: Protocol, record: dict, passages: list[str], basis: str) -> ScreenDecision:
    system = ("You assess eligibility at the FULL-TEXT stage of a systematic review. Use only the evidence "
              "given. Decide 'include' or 'exclude'; list the failed criterion ids (first = main reason).\n"
              + _criteria(p))
    ev = "\n\n".join(f"[passage {i+1}] {t}" for i, t in enumerate(passages))
    user = f"{_rec_txt(record, 3000)}\nEVIDENCE BASIS: {basis}\n\n{ev}"
    d = call_json(llm, system, user, ScreenDecision)
    d.id = record["id"]
    if d.decision == "uncertain":
        d.decision = "exclude"
        d.criteria_failed = d.criteria_failed or ["insufficient_information"]
    return d
