# Agentic AI Pipeline for Systematic Reviews (PRISMA 2020)

Système multi-agents (LangGraph + RAG) qui conduit une revue systématique de bout en bout à partir d'une question de recherche.

> 🚧 Documentation complète en cours — voir `docs/` (installation, configuration, exécution, évaluation).

## Structure
```
configs/            protocole de revue (YAML) — seul fichier à modifier pour un nouveau sujet
sr_agent/
  graph.py          coordinateur LangGraph (états, validations humaines, reprise après coupure)
  llm.py            accès LLM : débit limité, relances, sortie JSON validée
  schemas.py        schémas Pydantic échangés entre agents
  agents/           agents : protocole, stratégie de recherche, (screening, extraction, synthèse, critique)
  tools/            connecteurs OpenAlex / PubMed / Semantic Scholar / arXiv, dédoublonnage
  utils/            persistance (checkpoints, piste d'audit, compteurs PRISMA)
notebooks/          notebook d'exécution Kaggle
runs/<review_id>/   toutes les sorties et la piste d'audit de chaque revue
```
