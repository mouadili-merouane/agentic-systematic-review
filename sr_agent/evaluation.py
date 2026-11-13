"""Évaluation du pipeline : complétude, exactitude (vs. vérité terrain humaine), cohérence inter-agents,
gain de temps. Produit runs/<id>/eval/evaluation.json + evaluation_report.md.

Vérité terrain : un échantillon aléatoire stratifié de décisions est re-étiqueté par un humain EN AVEUGLE
(sans voir la décision de l'IA) via `label_screening_sample` ; idem pour un échantillon de champs extraits
via `label_extraction_sample`. Les deux sont reprenables après coupure (JSONL incrémental).
"""
import json, random, datetime, re
from collections import Counter
from .metrics import cohen_kappa, binary

EXTRACT_FIELDS = ["country", "data_source", "sample_size", "hf_population", "outcome", "ml_models", "best_model",
                  "best_auc", "best_auc_ci", "comparator", "comparator_auc", "validation", "calibration_reported",
                  "explainability"]
PRISMA_ITEMS = {  # éléments clés du checklist PRISMA 2020 recherchés dans le manuscrit
    "title identifies systematic review": r"systematic review",
    "structured abstract": r"## Abstract",
    "objectives / research question": r"Research question",
    "eligibility criteria": r"Inclusion criteria",
    "information sources + search date": r"were searched on",
    "full search strategy": r"Appendix A",
    "selection process": r"### Selection process",
    "data collection process": r"### Data collection",
    "risk of bias method": r"PROBAST",
    "synthesis methods": r"### Synthesis methods",
    "flow diagram": r"PRISMA 2020 flow diagram",
    "study characteristics": r"Characteristics of included studies",
    "risk of bias results": r"Risk of bias \(PROBAST",
    "results of syntheses": r"### 3.3",
    "limitations": r"### Limitations",
    "references": r"## References",
}


def _latest(store, name):
    return {r["id"]: r for r in store.read_jsonl(name)}


def _empty(v):
    return v is None or v == [] or (isinstance(v, str) and v.strip().lower() in ("", "not reported", "nr", "none",
                                                                                 "unclear"))


# ---------------------------------------------------------------- vérité terrain humaine (interactif)
def make_screening_sample(store, n=30, seed=42):
    F = _latest(store, "05_ta_final.jsonl")
    inc = sorted(i for i, d in F.items() if d["decision"] == "include")
    exc = sorted(i for i, d in F.items() if d["decision"] == "exclude")
    rnd = random.Random(seed)
    sample = rnd.sample(inc, min(n // 2, len(inc))) + rnd.sample(exc, min(n - n // 2, len(exc)))
    rnd.shuffle(sample)
    store.save_json("eval/screening_sample_ids.json", sample)
    return sample


def label_screening_sample(store, n=30, ask=input, show=print):
    (store.dir / "eval").mkdir(exist_ok=True)
    ids = store.load_json("eval/screening_sample_ids.json") or make_screening_sample(store, n)
    recs = {r["id"]: r for r in store.load_json("04_records.json")}
    p = store.load_json("01_protocol.json")
    show("CRITÈRES\n" + "\n".join(f"{c['id']}: {c['text']}" for c in p["inclusion_criteria"] + p["exclusion_criteria"]))
    done = store.done_ids("eval/human_screening.jsonl")
    for k, i in enumerate(ids, 1):
        if i in done:
            continue
        r = recs[i]
        show(f"\n[{k}/{len(ids)}] {r['title']} ({r.get('year')})\n{(r.get('abstract') or '[pas de résumé]')[:1800]}")
        a = ""
        while a not in ("i", "e"):
            a = ask("Décision (i = inclure en texte intégral / e = exclure) : ").strip().lower()[:1]
        store.append_jsonl("eval/human_screening.jsonl", {"id": i, "human": "include" if a == "i" else "exclude"})
    show("✔ étiquetage terminé")


def label_extraction_sample(store, n_studies=6, fields=("sample_size", "outcome", "best_auc", "validation"),
                            ask=input, show=print, seed=7):
    (store.dir / "eval").mkdir(exist_ok=True)
    E = _latest(store, "08_extraction.jsonl")
    ids = sorted(E)
    ids = random.Random(seed).sample(ids, min(n_studies, len(ids)))
    recs = {r["id"]: r for r in store.load_json("04_records.json")}
    done = {(r["id"], r["field"]) for r in store.read_jsonl("eval/human_extraction.jsonl")}
    for i in ids:
        show(f"\n=== {recs[i]['title']} ===  (source : {recs[i].get('url')})")
        quotes = {q["field"]: q["quote"] for q in E[i].get("evidence", [])}
        for f in fields:
            if (i, f) in done:
                continue
            show(f"  {f} = {E[i].get(f)!r}   | citation : {quotes.get(f, '-')}")
            a = ""
            while a not in ("o", "n", "?"):
                a = ask("  correct ? (o = oui / n = non / ? = impossible à vérifier) : ").strip().lower()[:1]
            store.append_jsonl("eval/human_extraction.jsonl", {"id": i, "field": f, "verdict": a})
    show("✔ vérification terminée")


# ---------------------------------------------------------------- métriques
def _timing(store):
    log = store.read_jsonl("audit_log.jsonl")
    ts = lambda e: datetime.datetime.fromisoformat(e["ts"])
    marks = {}
    for e in log:
        marks.setdefault(e["action"], ts(e))
    human = [e for e in log if e["agent"] == "human"]
    first = ts(log[0]) if log else None
    last = max((ts(e) for e in log if e["action"] in ("report_built", "critique_done", "extraction_done")),
               default=None)
    # temps actif : somme des intervalles entre événements, plafonnés à 15 min (exclut nuits / pauses)
    times = sorted(ts(e) for e in log)
    active = sum(min((b - a).total_seconds(), 900) for a, b in zip(times, times[1:]))
    return {"first_event": str(first), "last_event": str(last),
            "elapsed_hours_wall_clock": round((last - first).total_seconds() / 3600, 2) if first and last else None,
            "active_hours_pipeline": round(active / 3600, 2),
            "human_interventions": len(human),
            "human_overrides": sum(1 for e in log if e["action"] in ("ta_overrides",)),
            "model_switches": sum(1 for e in log if e["action"] == "model_switch")}


def evaluate(store, manual_min_per_abstract=1.0, manual_min_per_fulltext=10.0, manual_min_per_extraction=30.0):
    (store.dir / "eval").mkdir(exist_ok=True)
    out = {}
    recs = store.load_json("04_records.json", [])
    A, B = _latest(store, "05_ta_A.jsonl"), _latest(store, "05_ta_B.jsonl")
    F = _latest(store, "05_ta_final.jsonl")
    FT = _latest(store, "07_ft.jsonl")
    E = _latest(store, "08_extraction.jsonl")
    inc_ft = [i for i, d in FT.items() if d["decision"] == "include"]
    smp = store.load_json("08_extraction_sample.json")
    if smp:
        inc_ft = [i for i in smp["ids"] if i in inc_ft]
    E = {i: E[i] for i in inc_ft if i in E}

    # --- cohérence inter-agents
    ids = [r["id"] for r in recs if r["id"] in A and r["id"] in B]
    out["consistency"] = {
        "ta_kappa_A_B": round(cohen_kappa([binary(A[i]["decision"]) for i in ids],
                                          [binary(B[i]["decision"]) for i in ids]), 3),
        "ta_raw_agreement": round(sum(binary(A[i]["decision"]) == binary(B[i]["decision"]) for i in ids) / len(ids), 3)
        if ids else None,
        "ta_arbitrated_share": round(sum(F[i]["by"] == "arbiter" for i in F) / len(F), 3) if F else None,
        "ta_screener_A_include_rate": round(sum(binary(A[i]["decision"]) == "include" for i in ids) / len(ids), 3),
        "ta_screener_B_include_rate": round(sum(binary(B[i]["decision"]) == "include" for i in ids) / len(ids), 3),
    }
    crit = store.load_json("10_critique.json", {})
    if crit:
        c = crit["checks"]
        out["consistency"].update({
            "evidence_quotes_verified_in_source": c.get("evidence_verification_rate"),
            "critic_revision_rounds": crit.get("rounds"),
            "invalid_citations_final": len(c.get("invalid_citations", [])),
            "uncited_included_studies_final": len(c.get("uncited_studies", []))})

    # --- complétude
    field_fill = {f: round(sum(not _empty(e.get(f)) for e in E.values()) / len(E), 3) for f in EXTRACT_FIELDS} if E else {}
    ms = (store.dir / "report" / "manuscript.md")
    md = ms.read_text(encoding="utf-8") if ms.exists() else ""
    prisma = {k: bool(re.search(v, md)) for k, v in PRISMA_ITEMS.items()}
    st = _latest(store, "06_fulltext.jsonl")
    out["completeness"] = {
        "pipeline_stages_completed": sum(store.has(f) for f in ["01_protocol.json", "02_search_strategy.json",
                                                               "04_records.json", "05_ta_final.jsonl", "07_ft.jsonl",
                                                               "08_extraction.jsonl", "09_synthesis.json",
                                                               "report/manuscript.md"]),
        "pipeline_stages_total": 8,
        "fulltext_retrieval_rate": round(sum(d["retrieved"] for d in st.values()) / len(st), 3) if st else None,
        "studies_extracted_vs_included": f"{len(E)}/{len([i for i, d in FT.items() if d['decision'] == 'include'])}",
        "included_assessed_on_full_text": sum(d.get("basis") == "full_text_rag" for d in E.values()),
        "included_assessed_on_abstract_only": sum(d.get("basis") == "abstract_only" for d in E.values()),
        "extraction_field_fill_rate_mean": round(sum(field_fill.values()) / len(field_fill), 3) if field_fill else None,
        "extraction_field_fill_rate": field_fill,
        "prisma_items_present": f"{sum(prisma.values())}/{len(prisma)}",
        "prisma_items_missing": [k for k, v in prisma.items() if not v],
        "decisions_with_rationale": round(sum(bool(d.get("rationale")) for d in F.values()) / len(F), 3) if F else None,
    }

    # --- exactitude vs humain
    H = {r["id"]: r["human"] for r in store.read_jsonl("eval/human_screening.jsonl")}
    if H:
        tp = sum(H[i] == "include" and F[i]["decision"] == "include" for i in H)
        tn = sum(H[i] == "exclude" and F[i]["decision"] == "exclude" for i in H)
        fp = sum(H[i] == "exclude" and F[i]["decision"] == "include" for i in H)
        fn = sum(H[i] == "include" and F[i]["decision"] == "exclude" for i in H)
        out["accuracy_screening_vs_human"] = {
            "n": len(H), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round((tp + tn) / len(H), 3),
            "sensitivity_recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "kappa_ai_vs_human": round(cohen_kappa([H[i] for i in H], [F[i]["decision"] for i in H]), 3),
            "disagreements": [{"id": i, "human": H[i], "ai": F[i]["decision"], "ai_rationale": F[i]["rationale"][:200]}
                              for i in H if H[i] != F[i]["decision"]]}
    X = store.read_jsonl("eval/human_extraction.jsonl")
    if X:
        v = Counter(r["verdict"] for r in X)
        checked = v["o"] + v["n"]
        out["accuracy_extraction_vs_human"] = {
            "fields_checked": len(X), "correct": v["o"], "incorrect": v["n"], "unverifiable": v["?"],
            "accuracy": round(v["o"] / checked, 3) if checked else None,
            "by_field": {f: f"{sum(r['verdict'] == 'o' for r in X if r['field'] == f)}/"
                            f"{sum(r['verdict'] in 'on' for r in X if r['field'] == f)}"
                         for f in sorted({r['field'] for r in X})}}

    # --- temps
    t = _timing(store)
    n_abs, n_ft, n_ex = len(recs), len(FT), len(E)
    manual_h = (2 * n_abs * manual_min_per_abstract + 2 * n_ft * manual_min_per_fulltext
                + n_ex * manual_min_per_extraction) / 60
    t.update({"assumptions_manual_minutes": {"per_abstract_per_reviewer": manual_min_per_abstract,
                                             "per_fulltext_per_reviewer": manual_min_per_fulltext,
                                             "per_study_extraction": manual_min_per_extraction,
                                             "dual_review": True},
              "manual_estimate_hours_screening_and_extraction": round(manual_h, 1),
              "reference_benchmark": "Borah et al., BMJ Open 2017: mean 67.3 weeks to complete and publish a "
                                     "systematic review (195 PROSPERO reviews)."})
    if t.get("active_hours_pipeline"):
        t["speedup_vs_manual_estimate"] = round(manual_h / max(t["active_hours_pipeline"], 0.01), 1)
    out["time"] = t
    out["counts"] = store.load_json("prisma_counts.json", {})
    store.save_json("eval/evaluation.json", out)
    (store.dir / "eval" / "evaluation_report.md").write_text(to_markdown(out), encoding="utf-8")
    return out


def to_markdown(o):
    L = ["# Evaluation report", ""]
    c = o["consistency"]
    L += ["## 1. Inter-agent consistency", "| Metric | Value |", "|---|---|"] + \
         [f"| {k} | {v} |" for k, v in c.items()]
    cp = {k: v for k, v in o["completeness"].items() if k != "extraction_field_fill_rate"}
    L += ["", "## 2. Completeness", "| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cp.items()]
    L += ["", "**Extraction fill rate by field**", "", "| Field | Fill rate |", "|---|---|"] + \
         [f"| {k} | {v} |" for k, v in o["completeness"]["extraction_field_fill_rate"].items()]
    for key, title in [("accuracy_screening_vs_human", "3. Screening accuracy vs blinded human reviewer"),
                       ("accuracy_extraction_vs_human", "4. Extraction accuracy vs human verification")]:
        L += ["", f"## {title}"]
        if key in o:
            L += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in o[key].items()
                                                          if k not in ("disagreements",)]
        else:
            L += ["*Not yet labelled — run the human labelling cell.*"]
    t = o["time"]
    L += ["", "## 5. Time", "| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in t.items()]
    return "\n".join(L) + "\n"
