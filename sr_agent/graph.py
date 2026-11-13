"""Orchestration LangGraph : Coordinateur du pipeline.

Les données lourdes vivent sur disque (RunStore) ; l'état du graphe ne contient que le contrôle
(feedback humain, compteurs). Chaque nœud saute son travail si son résultat existe déjà -> reprise
après coupure de session. Les pauses humaines utilisent `interrupt()` + un checkpointer SQLite.
"""
import sqlite3, operator
from dataclasses import dataclass, field
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

from .utils.checkpoint import RunStore
from .schemas import Protocol, SearchStrategy, Record
from .agents import protocol_agent, search_strategy_agent, screening_agent
from .agents import extraction_agent, synthesis_agent, critic_agent
from .tools import report as report_tool
import hashlib, json
from .tools.sources import SEARCHERS
from .tools.dedup import deduplicate
from .tools import fulltext as ft
from .rag.index import HybridIndex
from .metrics import cohen_kappa, binary
from .llm import QuotaExceeded
import re
from collections import Counter

MAX_CRITIC_ROUNDS = 2   # révisions automatiques max demandées par l'agent critique
BATCH = 6          # articles par appel LLM au screening titre/résumé
SYNC_EVERY = 8     # sauvegarde GitHub tous les N appels pendant les longues boucles


@dataclass
class Ctx:
    store: RunStore
    cfg: dict                      # protocole YAML
    llm_fast: object
    llm_strong: object
    email: Optional[str] = None
    s2_key: Optional[str] = None
    hitl: bool = True
    llm_b: object = None           # screener B (par défaut = llm_strong)
    sync: object = None            # fonction de sauvegarde GitHub (optionnelle)
    extra: dict = field(default_factory=dict)

    def maybe_sync(self, i, msg):
        if self.sync and i and i % SYNC_EVERY == 0:
            try:
                self.sync(msg)
            except Exception as e:
                print("⚠ sync:", str(e)[:120])


class State(TypedDict, total=False):
    feedback: Optional[str]
    approved: Annotated[list, operator.add]   # étapes validées par l'humain
    counts: dict


# ---------------------------------------------------------------- helpers
def _review(ctx: Ctx, state: State, stage: str, summary: dict, accept_any=False) -> dict:
    """Point de contrôle humain. Retourne {'approve': bool, 'feedback': str}.
    Une étape déjà validée (fichier approvals.json ou état du graphe) n'est pas redemandée."""
    approvals = ctx.store.load_json("approvals.json", {})
    if stage in approvals or stage in (state.get("approved") or []):
        return {"approve": True, "feedback": None, "already": True}
    if not ctx.hitl:
        ans = {"approve": True, "feedback": None}
    else:
        ans = interrupt({"stage": stage, "summary": summary})
        if isinstance(ans, str):
            ans = {"approve": ans.strip().lower() in ("", "ok", "y", "yes", "oui"), "feedback": ans}
    if accept_any:
        ans["approve"] = True       # la réponse contient d'éventuelles corrections, déjà appliquées ensuite
    ctx.store.log("human", f"review_{stage}", **ans)
    if ans["approve"]:
        approvals[stage] = True
        ctx.store.save_json("approvals.json", approvals)
    return ans


def _unapprove(ctx: Ctx, *stages):
    a = ctx.store.load_json("approvals.json", {})
    for st in stages:
        a.pop(st, None)
    ctx.store.save_json("approvals.json", a)


def _latest(store, name):
    """Dernière décision par article (les corrections humaines sont ajoutées après)."""
    return {r["id"]: r for r in store.read_jsonl(name)}


_OVR = re.compile(r"\b(include|exclude)\s+((?:[0-9a-f]{12}[\s,;]*)+)", re.I)


def _apply_overrides(store, name, text):
    """'include id1 id2' / 'exclude id3' dans la réponse humaine -> corrections tracées."""
    n = 0
    for verb, ids in _OVR.findall(text or ""):
        for i in re.findall(r"[0-9a-f]{12}", ids):
            store.append_jsonl(name, {"id": i, "decision": verb.lower(), "by": "human",
                                      "rationale": "human override", "criteria_failed": []})
            n += 1
    return n


def ft_counts(S):
    """Recalcule les compteurs PRISMA de l'étape texte intégral."""
    R = _latest(S, "07_ft.jsonl")
    assessed = [d for d in R.values() if d["decision"] != "not_retrieved"]
    excl = [d for d in assessed if d["decision"] == "exclude"]
    reasons = Counter((d.get("criteria_failed") or ["unspecified"])[0] for d in excl)
    S.prisma(reports_assessed=len(assessed), reports_excluded_ft=len(excl), reports_excluded_reasons=dict(reasons),
             studies_included=sum(d["decision"] == "include" for d in assessed),
             assessed_on_abstract_only=sum(d.get("basis") == "abstract_only" for d in assessed))


def human_exclude(S, ids_with_reason: dict):
    """Exclusion humaine a posteriori (texte intégral) : {id: 'E3' ou motif}. Tracée dans la piste d'audit."""
    for i, reason in ids_with_reason.items():
        S.append_jsonl("07_ft.jsonl", {"id": i, "decision": "exclude", "by": "human", "basis": "human_review",
                                       "criteria_failed": [reason], "rationale": f"human exclusion ({reason})"})
    S.log("human", "ft_overrides", ids=ids_with_reason)
    ft_counts(S)


# ---------------------------------------------------------------- nodes
def make_nodes(ctx: Ctx):
    S = ctx.store

    def protocol(state: State):
        fb = state.get("feedback")
        if S.has("01_protocol.json") and not fb:
            return {}
        p = protocol_agent.run(ctx.llm_strong, ctx.cfg, fb)
        _unapprove(ctx, "protocol", "search_strategy")
        S.save_json("01_protocol.json", p.model_dump())
        S.log("protocol_agent", "protocol_generated", feedback=fb,
              n_inclusion=len(p.inclusion_criteria), n_exclusion=len(p.exclusion_criteria))
        return {"feedback": None}

    def review_protocol(state: State):
        p = S.load_json("01_protocol.json")
        ans = _review(ctx, state, "protocol", {
            "question": p["research_question"],
            "inclusion": [f'{c["id"]}: {c["text"]}' for c in p["inclusion_criteria"]],
            "exclusion": [f'{c["id"]}: {c["text"]}' for c in p["exclusion_criteria"]]})
        return {"approved": ["protocol"]} if ans["approve"] else {"feedback": ans["feedback"]}

    def search_strategy(state: State):
        fb = state.get("feedback")
        if S.has("02_search_strategy.json") and not fb:
            return {}
        p = Protocol(**S.load_json("01_protocol.json"))
        s = search_strategy_agent.run(ctx.llm_strong, p, fb)
        _unapprove(ctx, "search_strategy")
        q = search_strategy_agent.build_queries(s, p.years)
        S.save_json("02_search_strategy.json", search_strategy_agent.to_json(s, q))
        # une nouvelle stratégie invalide les résultats de recherche précédents
        for src in ctx.cfg["sources"]:
            S.delete(f"03_raw_{src}.json")
        S.delete("04_records.json")
        S.log("search_strategy_agent", "strategy_generated", feedback=fb, queries=q)
        return {"feedback": None}

    def review_strategy(state: State):
        st = S.load_json("02_search_strategy.json")
        ans = _review(ctx, state, "search_strategy", {
            "blocks": {b["name"]: b["keywords"] + [m + " [MeSH]" for m in b["mesh_terms"]]
                       for b in st["strategy"]["concept_blocks"]},
            "pubmed_query": st["queries"]["pubmed"]})
        return {"approved": ["search_strategy"]} if ans["approve"] else {"feedback": ans["feedback"]}

    def retrieve(state: State):
        st = S.load_json("02_search_strategy.json")
        years = ctx.cfg["years"]
        counts = {}
        for src, n in ctx.cfg["sources"].items():
            fname = f"03_raw_{src}.json"
            if not S.has(fname):                      # reprise : sources déjà faites sautées
                kw = {"mailto": ctx.email} if src == "openalex" else \
                     {"email": ctx.email} if src == "pubmed" else \
                     {"api_key": ctx.s2_key} if src == "semantic_scholar" else {}
                try:
                    recs = SEARCHERS[src](st["queries"][src], years, n, **kw)
                    S.save_json(fname, [r.model_dump() for r in recs])
                    S.log("retrieval_agent", "search_done", source=src, n=len(recs),
                          query=st["queries"][src])
                except Exception as e:                # une base en panne ne bloque pas la revue
                    S.log("retrieval_agent", "search_failed", source=src, error=str(e)[:300])
                    print(f"⚠ {src} a échoué : {e}")
                    continue
            counts[src] = len(S.load_json(fname))
        S.prisma(identified_per_source=counts, identified_total=sum(counts.values()))
        return {"counts": counts}

    def dedup(state: State):
        if not S.has("04_records.json"):
            raw = []
            for src in ctx.cfg["sources"]:
                raw += S.load_json(f"03_raw_{src}.json", [])
            kept, log = deduplicate(raw)
            S.save_json("04_records.json", kept)
            S.save_json("04_dedup_log.json", log)
            S.log("retrieval_agent", "dedup_done", before=len(raw), after=len(kept))
            S.prisma(duplicates_removed=len(raw) - len(kept), records_screened=len(kept))
        return {"counts": S.load_json("prisma_counts.json")}


    # ================= LOT 2 =================
    def screen_ta(state: State):
        recs = S.load_json("04_records.json")
        p = Protocol(**S.load_json("01_protocol.json"))
        calls = 0
        for persona, llm, fname in [("A", ctx.llm_fast, "05_ta_A.jsonl"),
                                    ("B", ctx.llm_b or ctx.llm_strong, "05_ta_B.jsonl")]:
            done = S.done_ids(fname)
            todo = [r for r in recs if r["id"] not in done]
            for k in range(0, len(todo), BATCH):
                batch = todo[k:k + BATCH]
                for d in screening_agent.screen_batch(llm, persona, p, batch):
                    S.append_jsonl(fname, {**d.model_dump(), "screener": persona})
                calls += 1
                print(f"  screener {persona}: {min(k + BATCH, len(todo))}/{len(todo)}", end="\r")
                ctx.maybe_sync(calls, f"screening TA {persona} en cours")
            print()
        A, B = _latest(S, "05_ta_A.jsonl"), _latest(S, "05_ta_B.jsonl")
        final_done = S.done_ids("05_ta_final.jsonl")
        by_id = {r["id"]: r for r in recs}
        for i, r in enumerate(recs):
            if r["id"] in final_done:
                continue
            a, b = A[r["id"]], B[r["id"]]
            if a["decision"] == b["decision"] and a["decision"] != "uncertain":
                out = {"id": r["id"], "decision": a["decision"], "by": "consensus",
                       "criteria_failed": sorted(set(a["criteria_failed"]) | set(b["criteria_failed"])),
                       "rationale": f"A: {a['rationale']} | B: {b['rationale']}"}
            else:
                d = screening_agent.arbitrate(ctx.llm_strong, p, by_id[r["id"]], a, b)
                out = {**d.model_dump(), "by": "arbiter"}
                calls += 1
                ctx.maybe_sync(calls, "arbitrage TA en cours")
            S.append_jsonl("05_ta_final.jsonl", out)
        ids = [r["id"] for r in recs]
        kappa = cohen_kappa([binary(A[i]["decision"]) for i in ids], [binary(B[i]["decision"]) for i in ids])
        F = _latest(S, "05_ta_final.jsonl")
        inc = [i for i in ids if F[i]["decision"] == "include"]
        summ = {"n": len(ids), "included": len(inc), "excluded": len(ids) - len(inc),
                "kappa_A_B": round(kappa, 3),
                "agreement": round(sum(binary(A[i]["decision"]) == binary(B[i]["decision"]) for i in ids) / len(ids), 3),
                "arbitrated": sum(F[i]["by"] == "arbiter" for i in ids)}
        S.save_json("05_ta_summary.json", summ)
        S.log("screening_agents", "ta_screening_done", **summ)
        S.prisma(records_excluded_ta=len(ids) - len(inc), reports_sought=len(inc))
        return {}

    def review_ta(state: State):
        summ = S.load_json("05_ta_summary.json")
        F = _latest(S, "05_ta_final.jsonl")
        recs = {r["id"]: r for r in S.load_json("04_records.json")}
        arb = [f"{i} [{F[i]['decision']}] {recs[i]['title'][:90]}" for i in F if F[i]["by"] == "arbiter"][:8]
        ans = _review(ctx, state, "screening_title_abstract", {
            "résumé": [f"{k}: {v}" for k, v in summ.items()],
            "exemples arbitrés (id [décision] titre)": arb,
            "aide": ["'ok' pour valider ; corrections possibles : 'include <id> <id>' / 'exclude <id>'"]},
            accept_any=True)
        n = _apply_overrides(S, "05_ta_final.jsonl", ans.get("feedback"))
        if n:
            F = _latest(S, "05_ta_final.jsonl")
            inc = sum(d["decision"] == "include" for d in F.values())
            S.prisma(records_excluded_ta=len(F) - inc, reports_sought=inc)
            S.log("human", "ta_overrides", n=n)
        return {"approved": ["screening_title_abstract"]}

    def fetch_fulltext(state: State):
        F = _latest(S, "05_ta_final.jsonl")
        recs = {r["id"]: r for r in S.load_json("04_records.json")}
        inc = [i for i, d in F.items() if d["decision"] == "include"]
        done = S.done_ids("06_fulltext.jsonl")
        (S.dir / "fulltext").mkdir(exist_ok=True)
        for n, i in enumerate([i for i in inc if i not in done], 1):
            text, src = ft.fetch(recs[i], ctx.email)
            if text:
                (S.dir / "fulltext" / f"{i}.txt").write_text(text, encoding="utf-8")
            S.append_jsonl("06_fulltext.jsonl", {"id": i, "retrieved": bool(text), "source": src,
                                                 "n_chars": len(text or "")})
            print(f"  texte intégral {n}: {'✔ ' + src if text else '✘ ' + src[:40]}", " " * 20, end="\r")
            ctx.maybe_sync(n, "récupération textes intégraux en cours")
        print()
        st = _latest(S, "06_fulltext.jsonl")
        got = sum(st[i]["retrieved"] for i in inc if i in st)
        S.prisma(reports_not_retrieved=len(inc) - got, fulltext_retrieved=got,
                 fulltext_sources=dict(Counter(st[i]["source"] for i in inc if i in st and st[i]["retrieved"])))
        S.log("retrieval_agent", "fulltext_done", sought=len(inc), retrieved=got)
        return {}

    def build_index(state: State):
        idx = HybridIndex(S.dir / "rag")
        if not idx.exists():
            docs = {p.stem: p.read_text(encoding="utf-8") for p in (S.dir / "fulltext").glob("*.txt")}
            if docs:
                n = idx.build(docs)
                S.log("retrieval_agent", "rag_index_built", papers=len(docs), chunks=n)
                if ctx.sync:
                    ctx.sync("index RAG construit")
        return {}

    def screen_ft(state: State):
        p = Protocol(**S.load_json("01_protocol.json"))
        F = _latest(S, "05_ta_final.jsonl")
        recs = {r["id"]: r for r in S.load_json("04_records.json")}
        st = _latest(S, "06_fulltext.jsonl")
        idx = HybridIndex(S.dir / "rag")
        idx = idx.load() if idx.exists() else None
        need_ft = bool(ctx.cfg.get("fulltext_required", False))
        q1 = f"{p.pico.get('population','')} study design cohort data source sample size {p.pico.get('intervention','')}"
        q2 = f"{p.pico.get('outcomes','')} AUC C-statistic performance results"
        done = S.done_ids("07_ft.jsonl")
        todo = [i for i, d in F.items() if d["decision"] == "include" and i not in done]
        for n, i in enumerate(todo, 1):
            r = recs[i]
            has = st.get(i, {}).get("retrieved") and idx is not None
            if not has and need_ft:
                S.append_jsonl("07_ft.jsonl", {"id": i, "decision": "not_retrieved", "basis": "none",
                                               "criteria_failed": [], "rationale": "full text not available"})
                continue
            if has:
                seen, passages = set(), []
                for q in (q1, q2):
                    for h in idx.search(q, k=4, paper_id=i):
                        if (h["pos"]) not in seen:
                            seen.add(h["pos"]); passages.append(h["text"])
                basis = "full_text_rag"
            else:
                passages, basis = [r.get("abstract") or ""], "abstract_only"
            d = screening_agent.screen_fulltext(ctx.llm_strong, p, r, passages, basis)
            S.append_jsonl("07_ft.jsonl", {**d.model_dump(), "basis": basis, "by": "fulltext_agent"})
            print(f"  éligibilité {n}/{len(todo)}", end="\r")
            ctx.maybe_sync(n, "screening texte intégral en cours")
        print()
        _ft_counts()
        return {}

    def _ft_counts():
        ft_counts(S)

    def _unused():
        R = _latest(S, "07_ft.jsonl")
        assessed = [d for d in R.values() if d["decision"] != "not_retrieved"]
        excl = [d for d in assessed if d["decision"] == "exclude"]
        reasons = Counter((d.get("criteria_failed") or ["unspecified"])[0] for d in excl)
        S.prisma(reports_assessed=len(assessed), reports_excluded_ft=len(excl),
                 reports_excluded_reasons=dict(reasons),
                 studies_included=sum(d["decision"] == "include" for d in assessed),
                 assessed_on_abstract_only=sum(d.get("basis") == "abstract_only" for d in assessed))

    def review_ft(state: State):
        R = _latest(S, "07_ft.jsonl")
        recs = {r["id"]: r for r in S.load_json("04_records.json")}
        inc = [f"{i} ({R[i].get('basis')}) {recs[i]['title'][:90]}" for i in R if R[i]["decision"] == "include"]
        c = S.load_json("prisma_counts.json")
        ans = _review(ctx, state, "screening_full_text", {
            "études incluses": inc,
            "exclusions (critère principal)": [f"{k}: {v}" for k, v in c.get("reports_excluded_reasons", {}).items()],
            "aide": ["'ok' pour valider ; corrections possibles : 'include <id>' / 'exclude <id>'"]},
            accept_any=True)
        if _apply_overrides(S, "07_ft.jsonl", ans.get("feedback")):
            _ft_counts()
        return {"approved": ["screening_full_text"]}


    # ================= LOT 3 =================
    def _included():
        R = _latest(S, "07_ft.jsonl")
        return [i for i, d in R.items() if d["decision"] == "include"]

    def _extraction_set():
        """Études à extraire : toutes, ou un échantillon reproductible si cfg.max_extraction est défini."""
        inc = sorted(_included())
        n = ctx.cfg.get("max_extraction")
        if not n or n >= len(inc):
            return inc
        saved = S.load_json("08_extraction_sample.json")
        if saved and set(saved["ids"]) <= set(inc):
            return saved["ids"]
        import random
        R = _latest(S, "07_ft.jsonl")
        done = [i for i in inc if i in S.done_ids("08_extraction.jsonl")]
        rnd = random.Random(42)
        full = [i for i in inc if R[i].get("basis") == "full_text_rag" and i not in done]
        rest = [i for i in inc if i not in full and i not in done]
        rnd.shuffle(full); rnd.shuffle(rest)
        ids = (done + full + rest)[:n]
        S.save_json("08_extraction_sample.json", {
            "ids": ids, "n_included": len(inc),
            "rule": f"sample of {n} included studies (seed 42), priority to open-access full texts"})
        S.log("coordinator", "extraction_sample", n=len(ids), n_included=len(inc))
        S.prisma(studies_extracted=len(ids))
        return ids

    def _source_text(i, recs):
        f = S.dir / "fulltext" / f"{i}.txt"
        return f.read_text(encoding="utf-8") if f.exists() else (recs[i].get("abstract") or "")

    def extract(state: State):
        p = Protocol(**S.load_json("01_protocol.json"))
        recs = {r["id"]: r for r in S.load_json("04_records.json")}
        R = _latest(S, "07_ft.jsonl")
        idx = HybridIndex(S.dir / "rag")
        idx = idx.load() if idx.exists() else None
        done = S.done_ids("08_extraction.jsonl")
        todo = [i for i in _extraction_set() if i not in done]
        for n, i in enumerate(todo, 1):
            basis = R[i].get("basis")
            if basis == "full_text_rag" and idx is not None:
                seen, passages = set(), []
                for q in extraction_agent.QUERIES:
                    for h in idx.search(q, k=2, paper_id=i):
                        if h["pos"] not in seen:
                            seen.add(h["pos"]); passages.append(h["text"])
            else:
                passages = [recs[i].get("abstract") or ""]
            try:
                e = extraction_agent.run(ctx.llm_strong, p.research_question, recs[i], passages)
            except QuotaExceeded:
                raise
            except Exception as err:          # une étude en échec ne bloque pas les autres (re-tentée au prochain run)
                S.log("extraction_agent", "extraction_failed", id=i, error=str(err)[:300])
                print(f"\n⚠ extraction {i} échouée (sera re-tentée) : {str(err)[:120]}")
                continue
            S.append_jsonl("08_extraction.jsonl", {"id": i, "basis": basis, **e.model_dump()})
            print(f"  extraction {n}/{len(todo)}", end="\r")
            ctx.maybe_sync(n, "extraction en cours")
        print()
        S.log("extraction_agent", "extraction_done", n=len(_extraction_set()))
        return {}

    def _table():
        recs = {r["id"]: r for r in S.load_json("04_records.json")}
        E = _latest(S, "08_extraction.jsonl")
        target = _extraction_set()
        missing = [i for i in target if i not in E]
        if missing:
            S.log("synthesis_agent", "studies_without_extraction", ids=missing)
        E = {i: E[i] for i in target if i in E}
        return synthesis_agent.compact_table(E, recs), E, recs

    def synthesize(state: State):
        fb = state.get("feedback")
        if S.has("09_synthesis.json") and not fb:
            return {}
        p = Protocol(**S.load_json("01_protocol.json"))
        table, _, _ = _table()
        smp = S.load_json("08_extraction_sample.json")
        note = None
        if smp:
            note = (
                f"the evidence table covers a random sample of {len(table)} of the {smp['n_included']} included "
                f"studies (demonstration under compute constraints); state this explicitly in the abstract, results "
                f"and limitations, and do not generalise beyond the sample.")
        syn = synthesis_agent.run(ctx.llm_strong, p.research_question, table, S.load_json("prisma_counts.json"), fb,
                                  note)
        S.save_json("09_synthesis.json", syn.model_dump())
        S.log("synthesis_agent", "synthesis_written", revision_feedback=fb)
        return {"feedback": None}

    def critique(state: State):
        syn = S.load_json("09_synthesis.json")
        h = hashlib.sha1(json.dumps(syn, sort_keys=True).encode()).hexdigest()[:10]
        prev = S.load_json("10_critique.json", {"rounds": 0})
        if prev.get("synthesis_hash") == h:
            return {}
        table, E, recs = _table()
        texts = {i: _source_text(i, recs) for i in E}
        checks = critic_agent.deterministic_checks(syn, table, E, texts)
        rev = critic_agent.run(ctx.llm_strong, syn, table, checks).model_dump()
        revise = critic_agent.needs_revision(rev, checks) and prev["rounds"] < MAX_CRITIC_ROUNDS
        cur = {"rounds": prev["rounds"] + (1 if revise else 0), "synthesis_hash": h, "checks": checks,
               "review": rev, "revision_requested": revise}
        S.save_json("10_critique.json", cur)
        hist = S.load_json("10_critique_history.json", [])
        S.save_json("10_critique_history.json", hist + [cur])
        S.log("critic_agent", "critique_done", n_issues=len(rev["issues"]), revise=revise,
              evidence_verification_rate=checks["evidence_verification_rate"])
        return {"feedback": critic_agent.feedback_text(rev, checks) if revise else None}

    def report(state: State):
        table, E, recs = _table()
        out = report_tool.build(S.dir, S.load_json("01_protocol.json"), S.load_json("02_search_strategy.json"),
                                S.load_json("prisma_counts.json"), S.load_json("05_ta_summary.json", {}),
                                S.load_json("09_synthesis.json"), table, E, recs,
                                ctx.extra.get("models", {}), S.load_json("10_critique.json"))
        S.log("synthesis_agent", "report_built", **out)
        return {}

    def review_report(state: State):
        c = S.load_json("10_critique.json", {})
        ans = _review(ctx, state, "final_report", {
            "fichiers": [f"report/{f}" for f in sorted(p.name for p in (S.dir / "report").iterdir())],
            "contrôle qualité": [f"citations d'évidence vérifiées : {c.get('checks', {}).get('evidence_verification_rate')}",
                                 f"révisions automatiques : {c.get('rounds')}",
                                 f"avis du critique : {c.get('review', {}).get('overall_assessment', '')}"],
            "aide": ["Lire report/manuscript.md ; 'ok' pour valider, sinon décrire les corrections à apporter"]})
        if ans["approve"]:
            return {"approved": ["final_report"]}
        return {"feedback": "Human reviewer: " + ans["feedback"]}

    return dict(protocol=protocol, review_protocol=review_protocol, search_strategy=search_strategy,
                review_strategy=review_strategy, retrieve=retrieve, dedup=dedup,
                screen_ta=screen_ta, review_ta=review_ta, fetch_fulltext=fetch_fulltext,
                build_index=build_index, screen_ft=screen_ft, review_ft=review_ft,
                extract=extract, synthesize=synthesize, critique=critique, report=report,
                review_report=review_report)


# ---------------------------------------------------------------- graph
def build_graph(ctx: Ctx, checkpoint_path: str):
    n = make_nodes(ctx)
    g = StateGraph(State)
    for name, fn in n.items():
        g.add_node(name, fn)
    g.set_entry_point("protocol")
    g.add_edge("protocol", "review_protocol")
    g.add_conditional_edges("review_protocol",
                            lambda s: "search_strategy" if not s.get("feedback") else "protocol")
    g.add_edge("search_strategy", "review_strategy")
    g.add_conditional_edges("review_strategy",
                            lambda s: "retrieve" if not s.get("feedback") else "search_strategy")
    g.add_edge("retrieve", "dedup")
    for a, b in [("dedup", "screen_ta"), ("screen_ta", "review_ta"), ("review_ta", "fetch_fulltext"),
                 ("fetch_fulltext", "build_index"), ("build_index", "screen_ft"), ("screen_ft", "review_ft"),
                 ("review_ft", "extract"), ("extract", "synthesize"), ("synthesize", "critique"),
                 ("report", "review_report")]:
        g.add_edge(a, b)
    g.add_conditional_edges("critique", lambda s: "synthesize" if s.get("feedback") else "report")
    g.add_conditional_edges("review_report", lambda s: "synthesize" if s.get("feedback") else END)
    saver = SqliteSaver(sqlite3.connect(checkpoint_path, check_same_thread=False))
    return g.compile(checkpointer=saver)


def drive(graph, thread_id: str, ask=input, show=print):
    """Lance ou reprend le graphe ; gère les pauses humaines jusqu'à la fin."""
    cfg = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(cfg)
    # thread en cours (pause ou coupure) -> reprise ; sinon nouveau passage (les étapes faites sont sautées)
    stream_input = None if snap.next else {"approved": []}
    while True:
        for ev in graph.stream(stream_input, cfg, stream_mode="updates"):
            for node, upd in ev.items():
                if node != "__interrupt__":
                    show(f"✔ {node}")
        snap = graph.get_state(cfg)
        pending = [i for t in snap.tasks for i in t.interrupts]
        if not pending:
            show("— pipeline terminé pour ce lot —")
            return snap.values
        q = pending[0].value
        show(f"\n=== VALIDATION HUMAINE : {q['stage']} ===")
        for k, v in q["summary"].items():
            show(f"\n[{k}]")
            for line in (v if isinstance(v, list) else [f"{a}: {b}" for a, b in v.items()]
                         if isinstance(v, dict) else [v]):
                show(f"  - {line}")
        ans = ask("\n'ok' pour valider, sinon écrivez vos corrections : ")
        stream_input = Command(resume=ans)
