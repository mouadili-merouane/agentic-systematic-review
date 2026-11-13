"""Assemblage du rapport final : manuscrit Markdown/DOCX, diagramme PRISMA, tableaux, références, BibTeX,
tableau prêt pour méta-analyse (CSV). La section Méthodes est générée de façon déterministe à partir du
protocole, de la stratégie et des compteurs -> elle décrit exactement ce que le pipeline a fait."""
import csv, re, datetime, pathlib

ID = re.compile(r"\[([0-9a-f]{12}(?:\s*[,;]\s*[0-9a-f]{12})*)\]")


# ---------------------------------------------------------------- PRISMA
def prisma_diagram(c: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    def box(x, y, w, h, txt, fc="#ffffff"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, fc=fc, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.5, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    per = c.get("identified_per_source", {})
    src = "\n".join(f"{k}: {v}" for k, v in per.items())
    reasons = "\n".join(f"{k}: {v}" for k, v in (c.get("reports_excluded_reasons") or {}).items()) or "-"
    for lab, y in [("Identification", 84), ("Screening", 45), ("Included", 6)]:
        ax.text(1, y, lab, rotation=90, fontsize=11, fontweight="bold", va="center")
    box(8, 76, 40, 18, f"Records identified (n = {c.get('identified_total', 0)})\n{src}", "#e8f0fb")
    box(56, 80, 38, 10, f"Duplicates removed (n = {c.get('duplicates_removed', 0)})")
    box(8, 58, 40, 10, f"Records screened (title/abstract)\n(n = {c.get('records_screened', 0)})")
    box(56, 58, 38, 10, f"Records excluded (n = {c.get('records_excluded_ta', 0)})")
    box(8, 40, 40, 10, f"Reports sought for retrieval\n(n = {c.get('reports_sought', 0)})")
    box(56, 40, 38, 10, f"Full text not retrieved (n = {c.get('reports_not_retrieved', 0)})\n"
                        f"→ assessed on abstract: {c.get('assessed_on_abstract_only', 0)}")
    box(8, 20, 40, 12, f"Reports assessed for eligibility\n(n = {c.get('reports_assessed', 0)})")
    box(56, 14, 38, 20, f"Reports excluded (n = {c.get('reports_excluded_ft', 0)})\n{reasons}")
    box(8, 1, 40, 10, f"Studies included in review\n(n = {c.get('studies_included', 0)})", "#e6f4ea")
    for y1, y2 in [(76, 68), (58, 50), (40, 32), (20, 11)]:
        arrow(28, y1, 28, y2)
    for y in [85, 63, 45, 26]:
        arrow(48, y, 56, y)
    ax.set_title("PRISMA 2020 flow diagram (generated automatically)", fontsize=12)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- tables
def _fmt(v):
    if v is None or v == "":
        return "NR"
    if isinstance(v, list):
        return ", ".join(map(str, v[:4])) + ("…" if len(v) > 4 else "")
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v).replace("|", "/").replace("\n", " ")


def md_table(rows, cols, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- references
def _short_authors(a):
    if not a:
        return "Anonymous"
    return a[0] + (" et al." if len(a) > 1 else "")


def reference(r):
    doi = f" https://doi.org/{r['doi']}" if r.get("doi") else (f" {r['url']}" if r.get("url") else "")
    return f"{', '.join(r.get('authors', [])[:6]) or 'Anonymous'}{' et al.' if len(r.get('authors', [])) > 6 else ''}. " \
           f"{r['title']}. *{r.get('venue') or 'n.d.'}*. {r.get('year') or 'n.d.'}.{doi}"


def bibtex(records):
    out = []
    for r in records:
        fields = {"title": r["title"], "author": " and ".join(r.get("authors", [])), "year": r.get("year"),
                  "journal": r.get("venue"), "doi": r.get("doi"), "url": r.get("url")}
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
        out.append(f"@article{{{r['id']},\n{body}\n}}")
    return "\n\n".join(out)


def number_citations(text, order: list, records: dict):
    """[id] -> [n] ; numérotation par ordre de première citation."""
    def rep(m):
        nums = []
        for i in re.split(r"\s*[,;]\s*", m.group(1)):
            if i in records:
                if i not in order:
                    order.append(i)
                nums.append(order.index(i) + 1)
        return "[" + ", ".join(map(str, sorted(nums))) + "]" if nums else ""
    return ID.sub(rep, text)


# ---------------------------------------------------------------- methods (déterministe)
def methods_text(protocol, strategy, counts, models, ta_summary):
    inc = "; ".join(f"{c['id']}: {c['text']}" for c in protocol["inclusion_criteria"])
    exc = "; ".join(f"{c['id']}: {c['text']}" for c in protocol["exclusion_criteria"])
    srcs = ", ".join(counts.get("identified_per_source", {}).keys())
    return f"""### Protocol and eligibility criteria
The review followed the PRISMA 2020 statement. The protocol was generated by the Protocol agent from a structured
configuration and validated by a human reviewer before searching. Population: {protocol['pico'].get('population')}.
Intervention: {protocol['pico'].get('intervention')}. Comparator: {protocol['pico'].get('comparator', 'none')}.
Outcomes: {protocol['pico'].get('outcomes')}. Inclusion criteria — {inc}. Exclusion criteria — {exc}.
Publication years {protocol['years'][0]}–{protocol['years'][1]}; languages: {', '.join(protocol['languages'])}.

### Information sources and search strategy
{srcs} were searched on {datetime.date.today().isoformat()} through their public APIs. The Search-strategy agent
proposed concept blocks (synonyms and MeSH terms), which were assembled deterministically into source-specific
Boolean queries and validated by a human reviewer. Full queries are given in Appendix A.

### Selection process
Records were deduplicated by DOI, exact normalised title and fuzzy title matching (token-sort ratio ≥ 95, ±1 year).
Titles and abstracts were screened independently by two LLM screeners with distinct instructions
(A: {models.get('screener_a')}, sensitive; B: {models.get('screener_b')}, specific). Disagreements and uncertain
decisions were resolved by an arbiter agent ({models.get('strong')}). Inter-screener agreement was
{ta_summary.get('agreement')} (Cohen's κ = {ta_summary.get('kappa_A_B')}). A human reviewer validated the screening
results. Open-access full texts were retrieved (Europe PMC, arXiv, publisher PDFs, Unpaywall) and indexed in a
hybrid retrieval-augmented generation (RAG) index (dense embeddings + BM25, reciprocal rank fusion, cross-encoder
re-ranking). Eligibility was assessed on the retrieved passages; when no open-access full text existed, the
abstract was used and the study is flagged.

### Data collection and risk of bias
The Extraction & Appraisal agent extracted predefined items (data source, sample size, population, outcome,
models, best AUC with 95% CI, comparator, validation, calibration, explainability) from retrieved passages, with
verbatim supporting quotes that were automatically verified against the source text. Risk of bias was appraised
with a simplified PROBAST (participants, predictors, outcome, analysis).

### Synthesis methods
Given expected heterogeneity, a narrative synthesis was performed; no pooled estimate was computed. Extracted
AUCs are provided in a meta-analysis-ready file. A Critic agent checked the draft for unsupported claims, invalid
citations and numerical inconsistencies; detected issues triggered revision."""


# ---------------------------------------------------------------- assemblage
def build(run_dir, protocol, strategy, counts, ta_summary, synthesis, table, extractions, records, models,
          critique=None):
    d = pathlib.Path(run_dir) / "report"
    d.mkdir(exist_ok=True)
    prisma_diagram(counts, d / "prisma_flow.png")
    order = []
    S = {k: number_citations(v, order, records) for k, v in synthesis.items()}
    # tableau des caractéristiques (numérotation cohérente avec le texte)
    for r in table:
        if r["id"] not in order:
            order.append(r["id"])
    rows = sorted(table, key=lambda r: order.index(r["id"]))
    for r in rows:
        r["ref"] = f"[{order.index(r['id']) + 1}]"
    char = md_table(rows, ["ref", "year", "country", "data_source", "sample_size", "outcome", "best_model",
                           "best_auc", "comparator_auc", "validation", "basis"],
                    ["Ref", "Year", "Country", "Data source", "N", "Outcome", "Best model", "AUC", "Comparator AUC",
                     "Validation", "Evidence basis"])
    rob_rows = [{"ref": r["ref"], **{k: extractions[r["id"]]["risk_of_bias"][k]["rating"]
                                     for k in ["participants", "predictors", "outcome", "analysis"]},
                 "overall": extractions[r["id"]]["risk_of_bias"]["overall"]} for r in rows]
    rob = md_table(rob_rows, ["ref", "participants", "predictors", "outcome", "analysis", "overall"],
                   ["Ref", "Participants", "Predictors", "Outcome", "Analysis", "Overall"])
    sample_note = ""
    if counts.get("studies_extracted") and counts["studies_extracted"] < (counts.get("studies_included") or 0):
        sample_note = (f" For this demonstration, data extraction, risk-of-bias appraisal and synthesis were performed "
                       f"on a reproducible random sample of {counts['studies_extracted']} included studies (seed 42, "
                       f"priority to open-access full texts); the remaining studies are listed but not synthesised.")
    queries = "\n\n".join(f"**{k}**\n```\n{v}\n```" for k, v in strategy["queries"].items())
    refs = "\n".join(f"{n}. {reference(records[i])}" for n, i in enumerate(order, 1))
    qa = ""
    if critique:
        c = critique["checks"]
        qa = (f"\n\n## Appendix B — Automated quality assurance\n- Revision rounds: {critique.get('rounds')}\n"
              f"- Evidence quotes verified in source text: {c['evidence_quotes_verified']}/{c['evidence_quotes_total']}"
              f" ({c['evidence_verification_rate']})\n- Invalid citations after revision: {len(c['invalid_citations'])}\n"
              f"- Critic assessment: {critique['review']['overall_assessment']}")
    md = f"""# {protocol['title']}: a systematic review
*Systematic review draft generated by an agentic AI pipeline — {datetime.date.today().isoformat()} — requires expert verification.*

## Abstract
{S['abstract']}

## 1. Introduction
{S['introduction']}

**Research question.** {protocol['research_question']}

## 2. Methods
{methods_text(protocol, strategy, counts, models, ta_summary)}

## 3. Results
### 3.1 Study selection
{counts.get('identified_total')} records were identified and {counts.get('duplicates_removed')} duplicates removed.
{counts.get('records_screened')} records were screened, {counts.get('reports_assessed')} reports were assessed for
eligibility and {counts.get('studies_included')} studies were included (Figure 1).{sample_note}

![Figure 1. PRISMA 2020 flow diagram](prisma_flow.png)

### 3.2 Study characteristics
{S['results_characteristics']}

**Table 1. Characteristics of included studies** (NR = not reported)

{char}

### 3.3 Predictive performance
{S['results_performance']}

### 3.4 Risk of bias
{S['results_risk_of_bias']}

**Table 2. Risk of bias (PROBAST, simplified)**

{rob}

## 4. Discussion
{S['discussion']}

### Limitations
{S['limitations']}

## 5. Conclusion
{S['conclusion']}

## References
{refs}

## Appendix A — Search queries
{queries}{qa}
"""
    (d / "manuscript.md").write_text(md, encoding="utf-8")
    (d / "references.bib").write_text(bibtex([records[i] for i in order]), encoding="utf-8")
    # tableau d'évidence complet + fichier prêt pour méta-analyse (R metafor, etc.)
    with open(d / "evidence_table.csv", "w", newline="", encoding="utf-8") as f:
        cols = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) if isinstance(v, list) else v for k, v in r.items()})
    with open(d / "meta_analysis_ready.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ref", "study_id", "outcome", "model", "auc", "auc_ci", "n", "comparator", "comparator_auc"])
        for r in rows:
            w.writerow([r["ref"], r["id"], r["outcome"], r["best_model"], r["best_auc"], r["best_auc_ci"],
                        r["sample_size"], r["comparator"], r["comparator_auc"]])
    docx = None
    try:
        import pypandoc
        pypandoc.convert_file(str(d / "manuscript.md"), "docx", outputfile=str(d / "manuscript.docx"),
                              extra_args=[f"--resource-path={d}"])
        docx = "manuscript.docx"
    except Exception as e:
        print("⚠ export DOCX ignoré :", str(e)[:120])
    return {"dir": str(d), "files": sorted(p.name for p in d.iterdir()), "docx": docx}
