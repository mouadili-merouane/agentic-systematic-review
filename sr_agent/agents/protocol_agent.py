"""Agent Protocole : transforme la configuration brute en protocole PRISMA opérationnel."""
import json
from ..llm import call_json
from ..schemas import Protocol

SYSTEM = """You are an expert systematic-review methodologist (PRISMA 2020, Cochrane Handbook).
Turn the draft review configuration into an operational protocol:
- keep the research question faithful to the draft (you may sharpen wording);
- rewrite inclusion/exclusion criteria so each one is a single, checkable condition that a screener
  can apply to a title/abstract; number them I1..In and E1..En;
- do not invent new scope; keep years, languages and study designs from the draft."""


def run(llm, cfg: dict, feedback: str | None = None) -> Protocol:
    draft = {k: cfg[k] for k in ["title", "research_question", "pico", "study_designs",
                                 "inclusion", "exclusion", "years", "languages"]}
    user = "Draft configuration:\n" + json.dumps(draft, ensure_ascii=False, indent=2)
    if feedback:
        user += f"\n\nHuman reviewer feedback to apply:\n{feedback}"
    return call_json(llm, SYSTEM, user, Protocol)
