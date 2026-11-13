"""Agent Synthèse & Rédaction : synthèse narrative fondée exclusivement sur le tableau d'extraction.

Chaque affirmation sur une étude cite son identifiant entre crochets, ex. [3f2a9c1b7d0e] ;
les identifiants sont convertis en références numérotées lors de l'assemblage du manuscrit.
"""
import json
from pydantic import BaseModel, Field
from ..llm import call_json


class Synthesis(BaseModel):
    abstract: str = Field(description="structured abstract: Background, Methods, Results, Conclusions (~250 words)")
    introduction: str = Field(description="2 short paragraphs: context and objective")
    results_characteristics: str = Field(description="narrative of study characteristics (countries, data, models)")
    results_performance: str = Field(description="narrative synthesis of predictive performance, ML vs comparators")
    results_risk_of_bias: str = Field(description="summary of PROBAST appraisal")
    discussion: str = Field(description="main findings, comparison with expectations, implications")
    limitations: str = Field(description="limitations of the evidence AND of this automated review")
    conclusion: str


SYSTEM = """You are the synthesis-and-writing agent of a PRISMA 2020 systematic review.
Write ONLY from the evidence table supplied. Rules:
- every statement about specific studies must cite their ids in square brackets, e.g. [a1b2c3d4e5f6];
- never invent numbers; report AUC values exactly as in the table and give ranges/medians you can compute from it;
- state explicitly when evidence is limited or heterogeneous; no meta-analysis pooling is performed (narrative synthesis);
- academic English, concise.
Research question: {question}"""


def compact_table(extractions: dict, records: dict) -> list[dict]:
    rows = []
    for i, e in extractions.items():
        r = records[i]
        rows.append({"id": i, "title": r["title"][:120], "year": r.get("year"),
                     **{k: e[k] for k in ["country", "data_source", "sample_size", "hf_population", "outcome",
                                          "ml_models", "best_model", "best_auc", "best_auc_ci", "comparator",
                                          "comparator_auc", "validation", "calibration_reported", "key_finding"]},
                     "rob_overall": e["risk_of_bias"]["overall"], "basis": e.get("basis")})
    return rows


def run(llm, question: str, table: list[dict], counts: dict, feedback: str | None = None,
        note: str | None = None) -> Synthesis:
    user = (f"PRISMA counts: {json.dumps(counts)}\n\nEVIDENCE TABLE ({len(table)} included studies):\n"
            + json.dumps(table, ensure_ascii=False))
    if note:
        user += f"\n\nIMPORTANT CONTEXT: {note}"
    if feedback:
        user += f"\n\nREVISION REQUIRED — fix these issues raised by the reviewer:\n{feedback}"
    return call_json(llm, SYSTEM.format(question=question), user, Synthesis)
