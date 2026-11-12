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
from .agents import protocol_agent, search_strategy_agent
from .tools.sources import SEARCHERS
from .tools.dedup import deduplicate


@dataclass
class Ctx:
    store: RunStore
    cfg: dict                      # protocole YAML
    llm_fast: object
    llm_strong: object
    email: Optional[str] = None
    s2_key: Optional[str] = None
    hitl: bool = True
    extra: dict = field(default_factory=dict)


class State(TypedDict, total=False):
    feedback: Optional[str]
    approved: Annotated[list, operator.add]   # étapes validées par l'humain
    counts: dict


# ---------------------------------------------------------------- helpers
def _review(ctx: Ctx, stage: str, summary: dict) -> dict:
    """Point de contrôle humain. Retourne {'approve': bool, 'feedback': str}."""
    if not ctx.hitl:
        return {"approve": True, "feedback": None}
    ans = interrupt({"stage": stage, "summary": summary})
    if isinstance(ans, str):
        ans = {"approve": ans.strip().lower() in ("", "ok", "y", "yes", "oui"), "feedback": ans}
    ctx.store.log("human", f"review_{stage}", **ans)
    return ans


# ---------------------------------------------------------------- nodes
def make_nodes(ctx: Ctx):
    S = ctx.store

    def protocol(state: State):
        fb = state.get("feedback")
        if S.has("01_protocol.json") and not fb:
            return {}
        p = protocol_agent.run(ctx.llm_strong, ctx.cfg, fb)
        S.save_json("01_protocol.json", p.model_dump())
        S.log("protocol_agent", "protocol_generated", feedback=fb,
              n_inclusion=len(p.inclusion_criteria), n_exclusion=len(p.exclusion_criteria))
        return {"feedback": None}

    def review_protocol(state: State):
        p = S.load_json("01_protocol.json")
        ans = _review(ctx, "protocol", {
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
        ans = _review(ctx, "search_strategy", {
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

    return dict(protocol=protocol, review_protocol=review_protocol, search_strategy=search_strategy,
                review_strategy=review_strategy, retrieve=retrieve, dedup=dedup)


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
    g.add_edge("dedup", END)
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
