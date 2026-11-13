"""Exécution en ligne de commande (hors Kaggle).

    export GROQ_API_KEY=...            # ou GOOGLE_API_KEY / OPENAI_API_KEY selon --provider
    python run.py --protocol configs/protocol_demo.yaml --email you@example.com
    python run.py --protocol configs/protocol_demo.yaml --evaluate      # métriques d'évaluation
    python run.py --protocol configs/protocol_demo.yaml --demo          # interface Gradio

La reprise est automatique : relancer la même commande continue là où l'exécution s'était arrêtée.
"""
import argparse, os, pathlib, yaml
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from sr_agent.llm import FallbackLLM
from sr_agent.utils.checkpoint import RunStore
from sr_agent.graph import Ctx, build_graph, drive

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", default="configs/protocol_demo.yaml")
ap.add_argument("--provider", default="groq")
ap.add_argument("--fast", default="openai/gpt-oss-20b,qwen/qwen3.6-27b,openai/gpt-oss-120b")
ap.add_argument("--strong", default="openai/gpt-oss-120b,qwen/qwen3.6-27b,openai/gpt-oss-20b")
ap.add_argument("--email", default=os.environ.get("CONTACT_EMAIL"))
ap.add_argument("--no-hitl", action="store_true", help="désactive les pauses de validation humaine")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--demo", action="store_true")
a = ap.parse_args()

root = pathlib.Path(__file__).parent
cfg = yaml.safe_load(open(a.protocol))
run_dir = root / "runs" / cfg["review_id"]
store = RunStore(run_dir)
set_llm_cache(SQLiteCache(database_path=str(root / "runs" / "llm_cache.db")))
sw = lambda o, n, e: store.log("coordinator", "model_switch", old=o, new=n, reason=e)
fast = FallbackLLM(a.fast.split(","), a.provider, sw)
strong = FallbackLLM(a.strong.split(","), a.provider, sw)

if a.evaluate:
    from sr_agent import evaluation as EV
    EV.evaluate(store)
    print((run_dir / "eval" / "evaluation_report.md").read_text())
elif a.demo:
    from sr_agent.demo_app import build_app
    build_app(run_dir, live_llm=strong, root=root).launch()
else:
    ctx = Ctx(store=store, cfg=cfg, llm_fast=fast, llm_strong=strong, llm_b=strong, email=a.email,
              s2_key=os.environ.get("S2_API_KEY"), hitl=not a.no_hitl,
              extra={"models": {"screener_a": fast.names[0], "screener_b": strong.names[0],
                                "strong": " / ".join(strong.names)}})
    drive(build_graph(ctx, str(run_dir / "graph_state.sqlite")), thread_id=cfg["review_id"])
