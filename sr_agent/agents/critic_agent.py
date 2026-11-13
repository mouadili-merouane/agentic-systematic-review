"""Agent Critique / Contrôle qualité.

1. Contrôles déterministes : citations inexistantes, études incluses jamais citées, valeurs d'AUC citées
   absentes du tableau, citations d'évidence non retrouvées dans le texte source (anti-hallucination).
2. Relecture LLM : cohérence, sur-interprétation, respect du protocole et des items PRISMA.
"""
import json, re
from typing import Literal
from pydantic import BaseModel, Field
from ..llm import call_json

ID = re.compile(r"\b[0-9a-f]{12}\b")


class Issue(BaseModel):
    severity: Literal["high", "medium", "low"]
    section: str
    issue: str
    fix: str


class Review(BaseModel):
    issues: list[Issue]
    overall_assessment: str = Field(description="2 sentences")


def _norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def deterministic_checks(synthesis: dict, table: list[dict], extractions: dict, texts: dict) -> dict:
    body = " ".join(str(v) for v in synthesis.values())
    cited = set(ID.findall(body))
    valid = {r["id"] for r in table}
    aucs = {round(float(x), 2) for r in table for x in (r.get("best_auc"), r.get("comparator_auc")) if x}
    if aucs:                                   # statistiques descriptives calculables depuis le tableau
        v = sorted(aucs)
        aucs |= {round(v[len(v) // 2], 2), round((v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2, 2),
                 round(sum(v) / len(v), 2)}
    nums = {round(float(m), 2) for m in re.findall(r"\b0\.\d{2,3}\b", body)}
    quotes_total = quotes_ok = 0
    unverified = []
    for i, e in extractions.items():
        src = _norm(texts.get(i, ""))
        for ev in e.get("evidence", []):
            quotes_total += 1
            q = _norm(ev["quote"])
            if q and (q in src or q[:60] in src):
                quotes_ok += 1
            else:
                unverified.append({"id": i, "field": ev["field"], "quote": ev["quote"][:120]})
    return {
        "invalid_citations": sorted(cited - valid),
        "uncited_studies": sorted(valid - cited),
        "numbers_not_in_table": sorted(n for n in nums - aucs
                                       if not any(abs(n - a) < 0.006 for a in aucs)),
        "evidence_quotes_total": quotes_total,
        "evidence_quotes_verified": quotes_ok,
        "evidence_verification_rate": round(quotes_ok / quotes_total, 3) if quotes_total else None,
        "unverified_quotes": unverified[:30],
    }


SYSTEM = """You are the critic / quality-assurance agent of a PRISMA 2020 systematic review.
Review the draft against the evidence table and the automated checks. Flag: claims not supported by the table,
over-interpretation (causal language, pooled estimates that were not computed), missing PRISMA elements
(limitations, risk of bias, heterogeneity), inconsistencies between sections, citation problems.
Severity 'high' only for factual errors or unsupported claims. Be concise and actionable."""


def run(llm, synthesis: dict, table: list[dict], checks: dict) -> Review:
    user = (f"AUTOMATED CHECKS:\n{json.dumps({k: v for k, v in checks.items() if k != 'unverified_quotes'})}\n\n"
            f"EVIDENCE TABLE:\n{json.dumps(table, ensure_ascii=False)}\n\nDRAFT:\n{json.dumps(synthesis, ensure_ascii=False)}")
    return call_json(llm, SYSTEM, user, Review)


def feedback_text(review: dict, checks: dict) -> str:
    lines = [f"- [{i['severity']}] {i['section']}: {i['issue']} -> {i['fix']}" for i in review["issues"]
             if i["severity"] in ("high", "medium")]
    if checks["invalid_citations"]:
        lines.append(f"- [high] citations: ids not in the table: {checks['invalid_citations']} -> remove them")
    if checks["numbers_not_in_table"]:
        lines.append(f"- [medium] numbers {checks['numbers_not_in_table']} are not AUC values from the table -> "
                     f"check them (keep only if they are correct statistics)")
    if checks["uncited_studies"]:
        lines.append(f"- [medium] these included studies are never cited: {checks['uncited_studies']} -> cite them")
    return "\n".join(lines)


def needs_revision(review: dict, checks: dict) -> bool:
    return bool(checks["invalid_citations"] or any(i["severity"] == "high" for i in review["issues"]))
