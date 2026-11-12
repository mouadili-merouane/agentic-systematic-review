"""Connecteurs aux bases académiques -> liste de Record normalisés.

OpenAlex, PubMed (E-utilities), Semantic Scholar (search/bulk), arXiv. Relances automatiques sur erreurs réseau.
"""
import time, re, xml.etree.ElementTree as ET
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from ..schemas import Record
from ..utils.checkpoint import stable_id

UA = {"User-Agent": "agentic-systematic-review/0.1 (academic project)"}


@retry(wait=wait_exponential(min=3, max=60), stop=stop_after_attempt(6), reraise=True)
def _get(url, params=None, headers=None, timeout=40):
    r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
    if r.status_code in (429, 500, 502, 503, 504):
        raise requests.HTTPError(f"{r.status_code} on {url}")
    r.raise_for_status()
    return r


def _rid(doi, title):
    return stable_id(doi) if doi else stable_id(re.sub(r"[^a-z0-9]", "", (title or "").lower()))


def _clean_doi(d):
    return (d or "").lower().replace("https://doi.org/", "").strip()


# ---------------- OpenAlex ----------------
def _oa_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def search_openalex(query, years, max_results=100, mailto=None):
    out, cursor = [], "*"
    while len(out) < max_results and cursor:
        params = {"search": query, "filter": f"publication_year:{years[0]}-{years[1]}",
                  "per-page": min(100, max_results - len(out)), "cursor": cursor}
        if mailto:
            params["mailto"] = mailto
        data = _get("https://api.openalex.org/works", params).json()
        for w in data.get("results", []):
            doi = _clean_doi(w.get("doi"))
            loc = w.get("best_oa_location") or {}
            ids = w.get("ids") or {}
            out.append(Record(
                id=_rid(doi, w.get("title")), title=w.get("title") or "",
                abstract=_oa_abstract(w.get("abstract_inverted_index")),
                year=w.get("publication_year"),
                authors=[a["author"]["display_name"] for a in w.get("authorships", [])[:10]],
                venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                doi=doi, pmid=(ids.get("pmid") or "").split("/")[-1],
                pmcid=(ids.get("pmcid") or "").split("/")[-1],
                url=w.get("id", ""), pdf_url=loc.get("pdf_url") or "", source="openalex"))
        cursor = data.get("meta", {}).get("next_cursor")
    return out[:max_results]


# ---------------- PubMed ----------------
def search_pubmed(query, years, max_results=100, email=None):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    p = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"}
    if email:
        p["email"] = email
    ids = _get(base + "esearch.fcgi", p).json()["esearchresult"]["idlist"]
    out = []
    for i in range(0, len(ids), 50):
        time.sleep(0.4)  # < 3 req/s sans clé NCBI
        xml = _get(base + "efetch.fcgi", {"db": "pubmed", "id": ",".join(ids[i:i + 50]),
                                          "retmode": "xml"}).text
        for art in ET.fromstring(xml).findall(".//PubmedArticle"):
            out.append(_parse_pubmed(art))
    return out


def _parse_pubmed(art):
    t = lambda path: "".join(art.find(path).itertext()).strip() if art.find(path) is not None else ""
    abstract = " ".join(
        ((a.get("Label") + ": ") if a.get("Label") else "") + "".join(a.itertext()).strip()
        for a in art.findall(".//Abstract/AbstractText"))
    idmap = {i.get("IdType"): (i.text or "") for i in art.findall(".//PubmedData/ArticleIdList/ArticleId")}
    year = t(".//Article/Journal/JournalIssue/PubDate/Year") or t(".//Article/Journal/JournalIssue/PubDate/MedlineDate")[:4]
    authors = [f"{a.findtext('LastName', '')} {a.findtext('Initials', '')}".strip()
               for a in art.findall(".//AuthorList/Author")[:10]]
    doi = _clean_doi(idmap.get("doi"))
    title = t(".//ArticleTitle")
    pmid = art.findtext(".//MedlineCitation/PMID") or ""
    return Record(id=_rid(doi, title), title=title, abstract=abstract,
                  year=int(year) if year.isdigit() else None, authors=authors,
                  venue=t(".//Article/Journal/Title"), doi=doi, pmid=pmid, pmcid=idmap.get("pmc", ""),
                  url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", source="pubmed")


# ---------------- Semantic Scholar ----------------
def search_semantic_scholar(query, years, max_results=60, api_key=None):
    fields = "title,abstract,year,authors,venue,externalIds,openAccessPdf,url"
    params = {"query": query, "fields": fields, "year": f"{years[0]}-{years[1]}"}
    headers = {"x-api-key": api_key} if api_key else {}
    out, token = [], None
    while len(out) < max_results:
        if token:
            params["token"] = token
        data = _get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params, headers).json()
        for p in data.get("data", []):
            ext = p.get("externalIds") or {}
            doi = _clean_doi(ext.get("DOI"))
            out.append(Record(
                id=_rid(doi, p.get("title")), title=p.get("title") or "", abstract=p.get("abstract") or "",
                year=p.get("year"), authors=[a.get("name", "") for a in (p.get("authors") or [])[:10]],
                venue=p.get("venue") or "", doi=doi, pmid=str(ext.get("PubMed") or ""),
                arxiv_id=ext.get("ArXiv") or "", url=p.get("url") or "",
                pdf_url=(p.get("openAccessPdf") or {}).get("url") or "", source="semantic_scholar"))
        token = data.get("token")
        if not token:
            break
        time.sleep(1.1)
    return out[:max_results]


# ---------------- arXiv ----------------
def search_arxiv(query, years, max_results=25):
    ns = {"a": "http://www.w3.org/2005/Atom", "ax": "http://arxiv.org/schemas/atom"}
    xml = _get("http://export.arxiv.org/api/query",
               {"search_query": query, "max_results": max_results, "sortBy": "relevance"}).text
    out = []
    for e in ET.fromstring(xml).findall("a:entry", ns):
        year = int((e.findtext("a:published", "", ns) or "0000")[:4])
        if not (years[0] <= year <= years[1]):
            continue
        aid = (e.findtext("a:id", "", ns) or "").split("/abs/")[-1]
        doi = _clean_doi(e.findtext("ax:doi", "", ns))
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        out.append(Record(
            id=_rid(doi, title), title=title, abstract=" ".join((e.findtext("a:summary", "", ns) or "").split()),
            year=year, authors=[a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)[:10]],
            venue="arXiv", doi=doi, arxiv_id=aid, url=f"https://arxiv.org/abs/{aid}",
            pdf_url=f"https://arxiv.org/pdf/{aid}", source="arxiv"))
    return out


SEARCHERS = {"openalex": search_openalex, "pubmed": search_pubmed,
             "semantic_scholar": search_semantic_scholar, "arxiv": search_arxiv}
