"""Récupération du texte intégral en accès libre.

Ordre : Europe PMC (XML structuré, via PMCID) -> arXiv -> PDF open access (OpenAlex / S2) -> Unpaywall (DOI).
Renvoie (texte, source) ou (None, raison).
"""
import io, re, xml.etree.ElementTree as ET
import requests

UA = {"User-Agent": "agentic-systematic-review/0.1 (academic project)"}
MIN_CHARS = 3000  # en dessous : probablement une page d'accueil, pas un article


def _pdf_text(content: bytes) -> str:
    import fitz  # PyMuPDF
    with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
    text = re.sub(r"-\n(\w)", r"\1", text)          # césures
    return re.sub(r"[ \t]+", " ", text)


def _cut_references(text: str) -> str:
    m = re.search(r"\n\s*(references|bibliography|literature cited)\s*\n", text, re.I)
    return text[:m.start()] if m and m.start() > len(text) * 0.5 else text


def _get_pdf(url):
    r = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
    if r.status_code != 200 or not r.content[:5] == b"%PDF-":
        return None
    return _pdf_text(r.content)


def from_europepmc(pmcid):
    r = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML", headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    root = ET.fromstring(r.content)
    parts = []
    for sec in root.iter("sec"):
        title = sec.findtext("title") or ""
        paras = [" ".join("".join(p.itertext()).split()) for p in sec.findall("p")]
        if paras:
            parts.append((f"## {title}\n" if title else "") + "\n".join(paras))
    abstract = " ".join(" ".join(a.itertext()).split() for a in root.iter("abstract"))
    return ("ABSTRACT: " + abstract + "\n\n" if abstract else "") + "\n\n".join(parts)


def from_unpaywall(doi, email):
    r = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email}, headers=UA, timeout=30)
    if r.status_code != 200:
        return None
    locs = [r.json().get("best_oa_location") or {}] + (r.json().get("oa_locations") or [])
    for loc in locs:
        if loc.get("url_for_pdf"):
            t = _get_pdf(loc["url_for_pdf"])
            if t and len(t) > MIN_CHARS:
                return t
    return None


def fetch(record: dict, email: str | None = None):
    attempts = []
    if record.get("pmcid"):
        attempts.append(("europepmc", lambda: from_europepmc(record["pmcid"])))
    if record.get("arxiv_id"):
        attempts.append(("arxiv", lambda: _get_pdf(f"https://arxiv.org/pdf/{record['arxiv_id']}")))
    if record.get("pdf_url"):
        attempts.append(("oa_pdf", lambda: _get_pdf(record["pdf_url"])))
    if record.get("doi") and email:
        attempts.append(("unpaywall", lambda: from_unpaywall(record["doi"], email)))
    errors = []
    for name, fn in attempts:
        try:
            t = fn()
            if t and len(t) > MIN_CHARS:
                return _cut_references(t), name
            errors.append(f"{name}:empty")
        except Exception as e:
            errors.append(f"{name}:{type(e).__name__}")
    return None, ("no_open_access" if not attempts else ";".join(errors))
