"""Agent Stratégie de recherche.

Le LLM propose les blocs de concepts (synonymes + MeSH) ; les requêtes propres à chaque base sont
ensuite assemblées par du code déterministe -> stratégie reproductible et auditable.
"""
import json
from ..llm import call_json
from ..schemas import Protocol, SearchStrategy

SYSTEM = """You are an information specialist designing a systematic-review search strategy.
Build 2-4 concept blocks (typically population, intervention/exposure, outcome).
Within a block, terms are synonyms joined with OR; blocks are joined with AND.
Favour sensitivity (recall) but avoid terms so generic they flood results (e.g. 'model', 'data').
Use short phrases (1-3 words). Give real MeSH headings only when you are confident they exist."""


def run(llm, protocol: Protocol, feedback: str | None = None) -> SearchStrategy:
    user = "Protocol:\n" + protocol.model_dump_json(indent=2)
    if feedback:
        user += f"\n\nHuman reviewer feedback to apply:\n{feedback}"
    return call_json(llm, SYSTEM, user, SearchStrategy)


# ---------- assemblage déterministe des requêtes ----------
def _q(t):
    return f'"{t}"' if " " in t or "-" in t else t


def build_queries(s: SearchStrategy, years: list[int]) -> dict:
    y0, y1 = years
    # PubMed : [tiab] + [Mesh], filtre de dates
    pm_blocks = []
    for b in s.concept_blocks:
        terms = [f'{_q(k)}[tiab]' for k in b.keywords] + [f'"{m}"[Mesh]' for m in b.mesh_terms]
        pm_blocks.append("(" + " OR ".join(terms) + ")")
    pubmed = " AND ".join(pm_blocks) + f' AND ("{y0}"[dp] : "{y1}"[dp])'
    if s.exclusion_keywords:
        pubmed += " NOT (" + " OR ".join(f"{_q(k)}[ti]" for k in s.exclusion_keywords) + ")"

    # OpenAlex : booléen dans `search` (titre + résumé + texte)
    oa = " AND ".join("(" + " OR ".join(_q(k) for k in b.keywords) + ")" for b in s.concept_blocks)

    # arXiv : champ abs:
    ax = " AND ".join("(" + " OR ".join(f'abs:{_q(k)}' for k in b.keywords) + ")" for b in s.concept_blocks)

    # Semantic Scholar /search/bulk : + (AND), | (OR), guillemets pour les phrases
    s2 = " + ".join("(" + " | ".join(_q(k) for k in b.keywords) + ")" for b in s.concept_blocks)

    return {"pubmed": pubmed, "openalex": oa, "arxiv": ax, "semantic_scholar": s2}


def to_json(s: SearchStrategy, queries: dict) -> dict:
    return {"strategy": json.loads(s.model_dump_json()), "queries": queries}
