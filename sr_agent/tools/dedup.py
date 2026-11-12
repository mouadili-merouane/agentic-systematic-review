"""Dédoublonnage : DOI identique -> titre normalisé identique -> titre quasi identique (fuzzy ≥ 95, année ±1).

Les doublons sont fusionnés (on garde le résumé le plus long et on complète les identifiants manquants).
"""
import re
from rapidfuzz import fuzz

ID_FIELDS = ["doi", "pmid", "pmcid", "arxiv_id", "pdf_url", "venue", "url"]


def _norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _merge(a: dict, b: dict) -> dict:
    if len(b.get("abstract", "")) > len(a.get("abstract", "")):
        a["abstract"] = b["abstract"]
    for f in ID_FIELDS:
        if not a.get(f) and b.get(f):
            a[f] = b[f]
    if not a.get("year") and b.get("year"):
        a["year"] = b["year"]
    a["found_in"] = sorted(set(a.get("found_in", [a["source"]])) | set(b.get("found_in", [b["source"]])))
    return a


def deduplicate(records: list[dict], threshold=95):
    kept, by_doi, by_title, log = [], {}, {}, []
    for r in records:
        r = {**r, "found_in": r.get("found_in") or [r["source"]]}
        if not r.get("title"):
            log.append({"dropped": r["id"], "reason": "no title"})
            continue
        doi, nt = r.get("doi"), _norm(r["title"])
        target = by_doi.get(doi) if doi else None
        reason = "doi"
        if target is None and nt in by_title:
            target, reason = by_title[nt], "exact_title"
        if target is None:
            for k in kept:
                if doi and k.get("doi") and doi != k["doi"]:
                    continue  # deux DOI différents = deux articles distincts
                if (abs((k.get("year") or 0) - (r.get("year") or 0)) <= 1
                        and fuzz.token_sort_ratio(_norm(k["title"]), nt) >= threshold):
                    target, reason = k, "fuzzy_title"
                    break
        if target is not None:
            _merge(target, r)
            log.append({"duplicate": r["id"], "of": target["id"], "source": r["source"], "rule": reason})
            if doi:
                by_doi.setdefault(doi, target)
            continue
        kept.append(r)
        if doi:
            by_doi[doi] = r
        by_title[nt] = r
    return kept, log
