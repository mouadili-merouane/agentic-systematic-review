"""Interface de démonstration (Gradio).

Mode « rejouer » : relit tout ce que le pipeline a produit dans runs/<review_id>/ (aucun appel LLM, instantané,
indépendant du quota) -> idéal pour la vidéo. Onglet « Nouveau sujet » : appel LLM en direct pour montrer que
le système s'adapte à n'importe quelle question (protocole + stratégie de recherche générés en quelques secondes).
"""
import json, time, pathlib
import pandas as pd

ICON = {"protocol_agent": "📋", "search_strategy_agent": "🔎", "retrieval_agent": "📚", "screening_agents": "🧪",
        "extraction_agent": "📑", "synthesis_agent": "✍️", "critic_agent": "🛡️", "human": "👤", "coordinator": "🧭",
        "setup": "⚙️"}


def _rd(run_dir):
    run_dir = pathlib.Path(run_dir)
    J = lambda n, d=None: json.loads((run_dir / n).read_text(encoding="utf-8")) if (run_dir / n).exists() else d
    L = lambda n: [json.loads(l) for l in (run_dir / n).read_text(encoding="utf-8").splitlines() if l.strip()] \
        if (run_dir / n).exists() else []
    last = lambda n: {r["id"]: r for r in L(n)}
    return run_dir, J, L, last


def _event_line(e):
    a, act = e["agent"], e["action"]
    extra = {k: v for k, v in e.items() if k not in ("ts", "agent", "action", "query", "queries", "reason")}
    txt = ", ".join(f"{k}={v}" for k, v in extra.items() if not isinstance(v, (dict, list)) and v not in (None, ""))
    return f"`{e['ts'][11:19]}` {ICON.get(a, '•')} **{a}** — {act}" + (f" · {txt[:140]}" if txt else "")


def build_app(run_dir, live_llm=None, root=None):
    import gradio as gr
    run_dir, J, L, last = _rd(run_dir)
    root = pathlib.Path(root or run_dir.parent.parent)
    recs = {r["id"]: r for r in J("04_records.json", [])}
    A, B, F = last("05_ta_A.jsonl"), last("05_ta_B.jsonl"), last("05_ta_final.jsonl")
    FT, E = last("07_ft.jsonl"), last("08_extraction.jsonl")
    proto, strat = J("01_protocol.json", {}), J("02_search_strategy.json", {})
    counts, ta = J("prisma_counts.json", {}), J("05_ta_summary.json", {})
    crit, ev = J("10_critique.json", {}), J("eval/evaluation.json", {})

    # ---------- données des onglets
    screen_df = pd.DataFrame([{
        "id": i, "title": r["title"][:90], "year": r.get("year"),
        "A": A.get(i, {}).get("decision"), "B": B.get(i, {}).get("decision"),
        "final T/A": F.get(i, {}).get("decision"), "by": F.get(i, {}).get("by"),
        "full text": FT.get(i, {}).get("decision", "—")} for i, r in recs.items()])
    incl = [i for i, d in FT.items() if d["decision"] == "include"]
    ext_df = pd.DataFrame([{"id": i, "title": recs[i]["title"][:80], "N": E[i].get("sample_size"),
                            "outcome": E[i].get("outcome"), "best model": E[i].get("best_model"),
                            "AUC": E[i].get("best_auc"), "validation": E[i].get("validation"),
                            "RoB": E[i]["risk_of_bias"]["overall"], "basis": E[i].get("basis")}
                           for i in incl if i in E])
    choices_s = [f"{i} — {recs[i]['title'][:70]}" for i in recs if F.get(i, {}).get("by") == "arbiter"] + \
                [f"{i} — {recs[i]['title'][:70]}" for i in recs if F.get(i, {}).get("by") != "arbiter"]
    choices_e = [f"{i} — {recs[i]['title'][:70]}" for i in incl if i in E]

    def screening_detail(choice):
        if not choice:
            return ""
        i = choice.split(" — ")[0]
        r = recs[i]
        s = f"### {r['title']} ({r.get('year')})\n*{r.get('venue','')}* · trouvé dans : {', '.join(r.get('found_in', []))}\n\n"
        s += f"> {(r.get('abstract') or '')[:1200]}\n\n"
        for lab, d in [("Screener A (sensible)", A.get(i)), ("Screener B (strict)", B.get(i)),
                       ("Décision T/A finale", F.get(i)), ("Texte intégral", FT.get(i))]:
            if d:
                s += (f"**{lab}** : `{d['decision']}`" + (f" · par *{d.get('by')}*" if d.get("by") else "")
                      + (f" · critères non remplis : {', '.join(d.get('criteria_failed') or [])}"
                         if d.get("criteria_failed") else "")
                      + (f" · base : {d.get('basis')}" if d.get("basis") else "") + f"\n\n> {d['rationale']}\n\n")
        return s

    unverified = {(u["id"], u["field"]) for u in crit.get("checks", {}).get("unverified_quotes", [])}

    def extraction_detail(choice):
        if not choice:
            return ""
        i = choice.split(" — ")[0]
        e = E[i]
        rows = "\n".join(f"| {k} | {e.get(k)} |" for k in ["country", "data_source", "study_design", "sample_size",
                                                          "hf_population", "outcome", "ml_models", "best_model",
                                                          "best_auc", "best_auc_ci", "comparator", "comparator_auc",
                                                          "validation", "calibration_reported", "explainability"])
        q = "\n".join(f"- {'⚠️ non retrouvée' if (i, x['field']) in unverified else '✅ vérifiée'} "
                      f"**{x['field']}** : « {x['quote']} »" for x in e.get("evidence", []))
        rob = "\n".join(f"| {d} | {e['risk_of_bias'][d]['rating']} | {e['risk_of_bias'][d]['rationale']} |"
                        for d in ["participants", "predictors", "outcome", "analysis"])
        return (f"### {recs[i]['title']}\n**Base de preuve :** {e.get('basis')} · **Résultat clé :** {e.get('key_finding')}"
                f"\n\n| Champ | Valeur |\n|---|---|\n{rows}\n\n#### Citations justificatives (vérifiées dans le texte source)\n{q}"
                f"\n\n#### Risque de biais (PROBAST) — global : **{e['risk_of_bias']['overall']}**\n"
                f"| Domaine | Note | Justification |\n|---|---|---|\n{rob}")

    def replay(speed):
        log, shown = L("audit_log.jsonl"), []
        log = [e for e in log if e["action"] not in ("session_start",)]
        for e in log:
            shown.append(_event_line(e))
            yield "\n\n".join(shown[-40:])
            time.sleep(max(0.02, 0.6 / speed))
        yield "\n\n".join(shown[-40:]) + "\n\n---\n✅ **Pipeline terminé** — " + \
            f"{counts.get('identified_total')} enregistrements → {counts.get('studies_included')} études incluses"

    def new_topic(question):
        if live_llm is None:
            yield "⚠️ Mode live indisponible (pas de LLM configuré)."
            return
        from .agents import protocol_agent, search_strategy_agent
        yield "📋 Agent Protocole en cours…"
        cfg = {"title": question[:80], "research_question": question, "pico": {}, "study_designs": [],
               "inclusion": [], "exclusion": [], "years": [2015, 2026], "languages": ["English"]}
        p = protocol_agent.run(live_llm, cfg)
        md = (f"### Protocole généré\n**Question :** {p.research_question}\n\n**PICO :** "
              + " · ".join(f"*{k}* : {v}" for k, v in p.pico.items())
              + "\n\n**Inclusion**\n" + "\n".join(f"- {c.id} : {c.text}" for c in p.inclusion_criteria)
              + "\n\n**Exclusion**\n" + "\n".join(f"- {c.id} : {c.text}" for c in p.exclusion_criteria))
        yield md + "\n\n🔎 Agent Stratégie de recherche en cours…"
        s = search_strategy_agent.run(live_llm, p)
        q = search_strategy_agent.build_queries(s, p.years)
        yield md + "\n\n### Stratégie de recherche\n" + "\n".join(
            f"- **{b.name}** : {', '.join(b.keywords)}" + (f" · MeSH : {', '.join(b.mesh_terms)}" if b.mesh_terms else "")
            for b in s.concept_blocks) + f"\n\n**Requête PubMed générée :**\n```\n{q['pubmed']}\n```"

    metrics_md = "### Indicateurs clés\n| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in [
        ("Enregistrements identifiés", counts.get("identified_total")),
        ("Doublons supprimés", counts.get("duplicates_removed")),
        ("Criblés titre/résumé", counts.get("records_screened")),
        ("κ screener A vs B", ta.get("kappa_A_B")), ("Accord brut A/B", ta.get("agreement")),
        ("Désaccords arbitrés", ta.get("arbitrated")),
        ("Textes intégraux récupérés", counts.get("fulltext_retrieved")),
        ("Études incluses", counts.get("studies_included")),
        ("Citations vérifiées dans la source", crit.get("checks", {}).get("evidence_verification_rate")),
        ("Révisions demandées par le critique", crit.get("rounds"))])
    if ev.get("accuracy_screening_vs_human"):
        a = ev["accuracy_screening_vs_human"]
        metrics_md += (f"\n\n**Vs. relecteur humain en aveugle (n={a['n']})** : exactitude {a['accuracy']} · "
                       f"sensibilité {a['sensitivity_recall']} · spécificité {a['specificity']} · κ {a['kappa_ai_vs_human']}")
    if ev.get("time"):
        t = ev["time"]
        metrics_md += (f"\n\n**Temps** : {t.get('active_hours_pipeline')} h de traitement actif vs ≈ "
                       f"{t.get('manual_estimate_hours_screening_and_extraction')} h estimées en manuel "
                       f"(×{t.get('speedup_vs_manual_estimate')})")
    critic_md = ""
    for k, c in enumerate(J("10_critique_history.json", []) or [], 1):
        critic_md += (f"**Passe {k}** — révision demandée : {c['revision_requested']} · citations invalides : "
                      f"{len(c['checks']['invalid_citations'])}\n\n" +
                      "\n".join(f"- [{x['severity']}] {x['section']} : {x['issue']}" for x in c["review"]["issues"][:6])
                      + "\n\n")
    manuscript = (run_dir / "report" / "manuscript.md")
    manu = manuscript.read_text(encoding="utf-8") if manuscript.exists() else "*Rapport non généré*"
    manu = manu.replace("![Figure 1. PRISMA 2020 flow diagram](prisma_flow.png)", "*(Figure 1 : voir le diagramme PRISMA)*")

    with gr.Blocks(title="Agentic Systematic Review") as app:
        gr.Markdown(f"# 🤖 Agentic AI Pipeline for Systematic Reviews\n**{proto.get('title', '')}** — "
                    f"PRISMA 2020 · LangGraph · RAG hybride · validation humaine")
        with gr.Tab("1 · Architecture & exécution"):
            arch = root / "docs" / "architecture.png"
            if arch.exists():
                gr.Image(str(arch), show_label=False)
            with gr.Row():
                speed = gr.Slider(1, 20, value=6, step=1, label="Vitesse de relecture")
                btn = gr.Button("▶ Rejouer l'exécution du pipeline (piste d'audit)", variant="primary")
            log_md = gr.Markdown()
            btn.click(replay, speed, log_md)
        with gr.Tab("2 · Protocole & recherche"):
            gr.Markdown(f"### Question\n{proto.get('research_question','')}\n\n**Inclusion**\n" +
                        "\n".join(f"- {c['id']} : {c['text']}" for c in proto.get("inclusion_criteria", [])) +
                        "\n\n**Exclusion**\n" + "\n".join(f"- {c['id']} : {c['text']}" for c in proto.get("exclusion_criteria", [])))
            gr.Markdown("### Requêtes générées par l'agent\n" + "\n".join(
                f"**{k}**\n```\n{v}\n```" for k, v in strat.get("queries", {}).items()))
            gr.Markdown("### Enregistrements par source\n" + " · ".join(
                f"**{k}** : {v}" for k, v in counts.get("identified_per_source", {}).items()))
        with gr.Tab("3 · Screening (double + arbitre)"):
            gr.Markdown(f"**κ (A vs B) = {ta.get('kappa_A_B')}** · accord {ta.get('agreement')} · "
                        f"{ta.get('arbitrated')} désaccords arbitrés · {ta.get('included')} retenus / {ta.get('n')}")
            gr.Dataframe(screen_df, interactive=False, max_height=320)
            dd = gr.Dropdown(choices_s, label="Voir le raisonnement des agents (les cas arbitrés sont en tête de liste)")
            det = gr.Markdown()
            dd.change(screening_detail, dd, det)
        with gr.Tab("4 · Extraction & risque de biais"):
            gr.Dataframe(ext_df, interactive=False, max_height=300)
            dd2 = gr.Dropdown(choices_e, label="Détail d'une étude : champs, citations vérifiées, PROBAST")
            det2 = gr.Markdown()
            dd2.change(extraction_detail, dd2, det2)
        with gr.Tab("5 · Résultats & qualité"):
            with gr.Row():
                pf = run_dir / "report" / "prisma_flow.png"
                if pf.exists():
                    gr.Image(str(pf), show_label=False)
                gr.Markdown(metrics_md)
            gr.Markdown("### Agent critique — passes de contrôle\n" + (critic_md or "*—*"))
        with gr.Tab("6 · Manuscrit généré"):
            gr.Markdown(manu)
        with gr.Tab("7 · Nouveau sujet (live)"):
            gr.Markdown("Le système est générique : saisissez n'importe quelle question de recherche.")
            q = gr.Textbox(label="Question de recherche",
                           value="Does deep learning on chest X-rays accurately detect pulmonary tuberculosis in adults?")
            go = gr.Button("Générer protocole + stratégie de recherche", variant="primary")
            out = gr.Markdown()
            go.click(new_topic, q, out)
    return app
