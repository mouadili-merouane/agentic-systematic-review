"""Métriques d'évaluation : accord inter-agents, performance du screening."""


def cohen_kappa(a: list, b: list) -> float:
    """Kappa de Cohen entre deux listes de labels (mêmes positions)."""
    n = len(a)
    if n == 0:
        return float("nan")
    labels = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def binary(decision: str) -> str:
    """'uncertain' compte comme 'include' (il part en texte intégral) pour l'accord binaire."""
    return "exclude" if decision == "exclude" else "include"
